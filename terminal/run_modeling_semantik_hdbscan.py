# terminal/run_modeling_semantik_hdbscan.py
# ============================================================
# Offline Runner — Clustering Semantik (HDBSCAN)
# FINAL (Hardcoded DB + Auto-create tables + Safe test limit default)
#
# Source (embedding):
#   - lasis_djp.semantik_embedding_runs
#   - lasis_djp.semantik_embedding_vectors  (embedding_json JSONB)
#
# Optional enrich (metadata/text):
#   - lasis_djp.incident_semantik (text_semantic, tgl_semantik, dsb)
#
# Output:
#   - lasis_djp.modeling_semantik_hdbscan_runs
#   - lasis_djp.modeling_semantik_hdbscan_clusters
#   - lasis_djp.modeling_semantik_hdbscan_members
#
# Default: limit 5000 untuk test cepat. Setelah sukses → --limit 0 (full).
#
# Usage:
#   python terminal/run_modeling_semantik_hdbscan.py --limit 5000
#   python terminal/run_modeling_semantik_hdbscan.py --embedding-run-id <UUID> --limit 0
#   python terminal/run_modeling_semantik_hdbscan.py --min-cluster-size 30 --min-samples 10
#
# Catatan:
# - Pastikan package "hdbscan" terinstall:
#     pip install hdbscan
# ============================================================

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from psycopg2.extras import Json

# sklearn optional metrics
from sklearn.metrics import silhouette_score, davies_bouldin_score


# ============================================================
# 🔐 DB CONFIG (HARDCODED)
# ============================================================
DB = {
    "host": "localhost",
    "port": 5432,
    "database": "incident_djp",
    "user": "postgres",
    "password": "admin*123",
}

SCHEMA = "lasis_djp"

T_EMB_RUNS = "semantik_embedding_runs"
T_EMB_VECS = "semantik_embedding_vectors"
T_SEMANTIK = "incident_semantik"

T_RUNS = "modeling_semantik_hdbscan_runs"
T_CLUSTERS = "modeling_semantik_hdbscan_clusters"
T_MEMBERS = "modeling_semantik_hdbscan_members"


# ============================================================
# DB ENGINE
# ============================================================
def get_engine() -> Engine:
    url = (
        f"postgresql+psycopg2://{DB['user']}:{DB['password']}"
        f"@{DB['host']}:{DB['port']}/{DB['database']}"
    )
    return create_engine(url, pool_pre_ping=True, future=True)


# ============================================================
# CLI
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser("Offline Modeling Semantik — HDBSCAN")
    ap.add_argument("--embedding-run-id", default=None, help="run_id dari semantik_embedding_runs (uuid). Default: latest")
    ap.add_argument("--limit", type=int, default=5000, help="Jumlah baris embedding untuk test. 0=full.")
    ap.add_argument("--seed", type=int, default=42)

    # HDBSCAN params
    ap.add_argument("--min-cluster-size", type=int, default=30)
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--metric", default="euclidean", help="euclidean (recommended when embeddings are L2-normalized)")
    ap.add_argument("--cluster-selection-method", choices=["eom", "leaf"], default="eom")

    # Optional dimensionality reduction (PCA) for speed/noise control
    ap.add_argument("--pca-dim", type=int, default=0, help="0=off. contoh 50 untuk percepat HDBSCAN")

    # Save behavior
    ap.add_argument("--notes", default="")
    ap.add_argument("--replace-members", action="store_true", help="Jika ON, hapus dulu hasil untuk modeling_id ini (members+clusters+runs).")

    return ap.parse_args()


# ============================================================
# Ensure tables
# ============================================================
def ensure_tables(engine: Engine) -> None:
    ddl_runs = f"""
    CREATE TABLE IF NOT EXISTS {SCHEMA}.{T_RUNS} (
        modeling_id uuid PRIMARY KEY,
        run_time timestamp without time zone NOT NULL,
        embedding_run_id uuid NOT NULL,
        n_rows integer NOT NULL,
        n_clusters integer NOT NULL,
        n_noise integer NOT NULL,
        silhouette double precision,
        dbi double precision,
        params_json jsonb NOT NULL,
        notes text
    );
    """

    ddl_clusters = f"""
    CREATE TABLE IF NOT EXISTS {SCHEMA}.{T_CLUSTERS} (
        modeling_id uuid NOT NULL,
        cluster_id integer NOT NULL,
        cluster_size integer NOT NULL,
        avg_prob double precision,
        avg_outlier_score double precision,
        PRIMARY KEY (modeling_id, cluster_id)
    );
    """

    ddl_members = f"""
    CREATE TABLE IF NOT EXISTS {SCHEMA}.{T_MEMBERS} (
        modeling_id uuid NOT NULL,
        cluster_id integer NOT NULL,
        is_noise boolean NOT NULL,
        incident_number text NOT NULL,
        tgl_submit timestamp without time zone,
        site text,
        modul text,
        sub_modul text,
        prob double precision,
        outlier_score double precision,
        PRIMARY KEY (modeling_id, incident_number)
    );
    """

    ddl_idx = f"""
    CREATE INDEX IF NOT EXISTS idx_{T_MEMBERS}_modeling
        ON {SCHEMA}.{T_MEMBERS}(modeling_id);
    CREATE INDEX IF NOT EXISTS idx_{T_MEMBERS}_cluster
        ON {SCHEMA}.{T_MEMBERS}(modeling_id, cluster_id);
    """

    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};"))
        conn.execute(text(ddl_runs))
        conn.execute(text(ddl_clusters))
        conn.execute(text(ddl_members))
        conn.execute(text(ddl_idx))


# ============================================================
# Helpers
# ============================================================
def get_latest_embedding_run_id(engine: Engine) -> str:
    sql = f"""
    SELECT run_id::text
    FROM {SCHEMA}.{T_EMB_RUNS}
    ORDER BY run_time DESC
    LIMIT 1
    """
    with engine.connect() as conn:
        row = conn.execute(text(sql)).fetchone()
    if not row:
        raise RuntimeError(f"Tidak ada data di {SCHEMA}.{T_EMB_RUNS}. Jalankan embedding dulu.")
    return str(row[0])


def maybe_delete_existing(engine: Engine, modeling_id: uuid.UUID) -> None:
    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {SCHEMA}.{T_MEMBERS} WHERE modeling_id = :mid"), {"mid": modeling_id})
        conn.execute(text(f"DELETE FROM {SCHEMA}.{T_CLUSTERS} WHERE modeling_id = :mid"), {"mid": modeling_id})
        conn.execute(text(f"DELETE FROM {SCHEMA}.{T_RUNS} WHERE modeling_id = :mid"), {"mid": modeling_id})


def load_embeddings(
    engine: Engine,
    embedding_run_id: str,
    limit: int,
) -> pd.DataFrame:
    limit_sql = f"LIMIT {int(limit)}" if int(limit) > 0 else ""
    sql = f"""
    SELECT
        v.incident_number,
        v.tgl_submit,
        v.embedding_dim,
        v.embedding_json
    FROM {SCHEMA}.{T_EMB_VECS} v
    WHERE v.run_id = :rid
    {limit_sql}
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params={"rid": embedding_run_id})
    if df.empty:
        raise RuntimeError("Embedding vectors kosong untuk run_id tersebut.")
    return df


def enrich_members_with_semantik(engine: Engine, df_members: pd.DataFrame) -> pd.DataFrame:
    """
    Tambahkan site/modul/sub_modul jika ada di incident_semantik.
    Tidak wajib; kalau tabel/kolom tidak ada, tetap jalan.
    """
    # cek kolom tersedia
    chk_sql = f"""
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = :schema AND table_name = :table
    """
    with engine.connect() as conn:
        cols = [r[0] for r in conn.execute(text(chk_sql), {"schema": SCHEMA, "table": T_SEMANTIK}).fetchall()]

    want = ["incident_number", "site", "modul", "sub_modul", "tgl_semantik"]
    has = [c for c in want if c in cols]

    if "incident_number" not in has:
        return df_members

    select_cols = ", ".join(has)
    sql = f"SELECT {select_cols} FROM {SCHEMA}.{T_SEMANTIK}"
    with engine.connect() as conn:
        df_meta = pd.read_sql(text(sql), conn)

    # rename tgl_semantik -> tgl_submit if members has nulls
    if "tgl_semantik" in df_meta.columns and "tgl_submit" in df_members.columns:
        df_meta = df_meta.rename(columns={"tgl_semantik": "tgl_submit_meta"})

    out = df_members.merge(df_meta, on="incident_number", how="left")

    # if members tgl_submit null, fill from tgl_submit_meta
    if "tgl_submit_meta" in out.columns:
        out["tgl_submit"] = out["tgl_submit"].fillna(out["tgl_submit_meta"])
        out = out.drop(columns=["tgl_submit_meta"])

    # ensure columns exist
    for c in ["site", "modul", "sub_modul"]:
        if c not in out.columns:
            out[c] = None

    return out


def to_matrix_from_json(df: pd.DataFrame) -> np.ndarray:
    """
    embedding_json disimpan JSONB list[float].
    Konversi ke numpy matrix (n x dim).
    """
    # df.embedding_json bisa sudah jadi list atau string json tergantung driver
    arrs: List[np.ndarray] = []
    for v in df["embedding_json"].tolist():
        if v is None:
            arrs.append(np.array([], dtype=np.float32))
            continue
        if isinstance(v, str):
            vec = json.loads(v)
        else:
            vec = v
        arrs.append(np.asarray(vec, dtype=np.float32))

    X = np.vstack(arrs)
    return X


def apply_pca(X: np.ndarray, pca_dim: int, seed: int) -> np.ndarray:
    if pca_dim <= 0:
        return X
    from sklearn.decomposition import PCA
    d = min(int(pca_dim), X.shape[1])
    pca = PCA(n_components=d, random_state=int(seed))
    return pca.fit_transform(X).astype(np.float32)


# ============================================================
# HDBSCAN
# ============================================================
def run_hdbscan(
    X: np.ndarray,
    *,
    min_cluster_size: int,
    min_samples: int,
    metric: str,
    cluster_selection_method: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import hdbscan  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Package 'hdbscan' belum terinstall. Jalankan: pip install hdbscan"
        ) from e

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        metric=str(metric),
        cluster_selection_method=str(cluster_selection_method),
        prediction_data=False,
    )
    labels = clusterer.fit_predict(X)  # -1 = noise
    probs = getattr(clusterer, "probabilities_", None)
    outlier = getattr(clusterer, "outlier_scores_", None)

    if probs is None:
        probs = np.zeros(X.shape[0], dtype=np.float32)
    if outlier is None:
        outlier = np.zeros(X.shape[0], dtype=np.float32)

    return labels.astype(int), np.asarray(probs, dtype=np.float32), np.asarray(outlier, dtype=np.float32)


# ============================================================
# Save results
# ============================================================
def save_runs_row(
    engine: Engine,
    modeling_id: uuid.UUID,
    embedding_run_id: str,
    n_rows: int,
    n_clusters: int,
    n_noise: int,
    silhouette: Optional[float],
    dbi: Optional[float],
    params: Dict[str, Any],
    notes: str,
) -> None:
    WIB = timezone(timedelta(hours=7))
    run_time = datetime.now(WIB).replace(tzinfo=None)

    with engine.begin() as conn:
        conn.execute(
            text(f"""
            INSERT INTO {SCHEMA}.{T_RUNS}
            (modeling_id, run_time, embedding_run_id, n_rows, n_clusters, n_noise, silhouette, dbi, params_json, notes)
            VALUES
            (:mid, :rt, :erid, :n_rows, :n_clusters, :n_noise, :sil, :dbi, :params, :notes)
            """),
            {
                "mid": modeling_id,
                "rt": run_time,
                "erid": embedding_run_id,
                "n_rows": int(n_rows),
                "n_clusters": int(n_clusters),
                "n_noise": int(n_noise),
                "sil": silhouette,
                "dbi": dbi,
                "params": Json(params),
                "notes": notes,
            }
        )


def save_clusters(engine: Engine, df_clusters: pd.DataFrame) -> None:
    if df_clusters.empty:
        return
    records = df_clusters.to_dict(orient="records")
    for r in records:
        # ensure numeric
        r["cluster_id"] = int(r["cluster_id"])
        r["cluster_size"] = int(r["cluster_size"])
    with engine.begin() as conn:
        conn.execute(
            text(f"""
            INSERT INTO {SCHEMA}.{T_CLUSTERS}
            (modeling_id, cluster_id, cluster_size, avg_prob, avg_outlier_score)
            VALUES
            (:modeling_id, :cluster_id, :cluster_size, :avg_prob, :avg_outlier_score)
            ON CONFLICT (modeling_id, cluster_id) DO UPDATE SET
              cluster_size = EXCLUDED.cluster_size,
              avg_prob = EXCLUDED.avg_prob,
              avg_outlier_score = EXCLUDED.avg_outlier_score
            """),
            records
        )


def save_members(engine: Engine, df_members: pd.DataFrame) -> None:
    if df_members.empty:
        return
    # convert to records & ensure Json already handled upstream (not needed here)
    records = df_members.to_dict(orient="records")
    with engine.begin() as conn:
        conn.execute(
            text(f"""
            INSERT INTO {SCHEMA}.{T_MEMBERS}
            (modeling_id, cluster_id, is_noise, incident_number, tgl_submit, site, modul, sub_modul, prob, outlier_score)
            VALUES
            (:modeling_id, :cluster_id, :is_noise, :incident_number, :tgl_submit, :site, :modul, :sub_modul, :prob, :outlier_score)
            ON CONFLICT (modeling_id, incident_number) DO UPDATE SET
              cluster_id = EXCLUDED.cluster_id,
              is_noise = EXCLUDED.is_noise,
              tgl_submit = EXCLUDED.tgl_submit,
              site = EXCLUDED.site,
              modul = EXCLUDED.modul,
              sub_modul = EXCLUDED.sub_modul,
              prob = EXCLUDED.prob,
              outlier_score = EXCLUDED.outlier_score
            """),
            records
        )


# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()
    np.random.seed(int(args.seed))

    engine = get_engine()
    print("[INFO] Ensuring modeling tables ...")
    ensure_tables(engine)

    embedding_run_id = args.embedding_run_id or get_latest_embedding_run_id(engine)
    print(f"[INFO] Using embedding_run_id = {embedding_run_id}")

    print(f"[INFO] Loading embeddings (limit={args.limit}) ...")
    df_emb = load_embeddings(engine, embedding_run_id, args.limit)
    n_rows = len(df_emb)
    print(f"[INFO] Loaded {n_rows:,} rows")

    # matrix
    X = to_matrix_from_json(df_emb)
    if X.ndim != 2 or X.shape[0] != n_rows:
        raise RuntimeError("Gagal membentuk matrix embedding.")
    print(f"[INFO] Matrix shape = {X.shape[0]:,} x {X.shape[1]}")

    # optional PCA
    if int(args.pca_dim) > 0:
        print(f"[INFO] Applying PCA -> {args.pca_dim} dims ...")
        X = apply_pca(X, int(args.pca_dim), int(args.seed))
        print(f"[INFO] After PCA shape = {X.shape[0]:,} x {X.shape[1]}")

    # run hdbscan
    print("[INFO] Running HDBSCAN ...")
    t0 = time.time()
    labels, probs, outlier = run_hdbscan(
        X,
        min_cluster_size=int(args.min_cluster_size),
        min_samples=int(args.min_samples),
        metric=str(args.metric),
        cluster_selection_method=str(args.cluster_selection_method),
    )
    elapsed = time.time() - t0
    print(f"[INFO] HDBSCAN done in {elapsed:,.1f}s")

    # cluster stats
    n_noise = int(np.sum(labels == -1))
    cluster_ids = sorted([int(c) for c in set(labels.tolist()) if int(c) != -1])
    n_clusters = int(len(cluster_ids))
    print(f"[INFO] n_clusters={n_clusters} | n_noise={n_noise:,} ({(n_noise/n_rows*100):.1f}%)")

    # evaluation (internal) - exclude noise
    sil = None
    dbi = None
    mask = labels != -1
    if n_clusters >= 2 and int(np.sum(mask)) >= 10:
        try:
            sil = float(silhouette_score(X[mask], labels[mask], metric="euclidean"))
        except Exception:
            sil = None
        try:
            dbi = float(davies_bouldin_score(X[mask], labels[mask]))
        except Exception:
            dbi = None

    # new modeling_id
    modeling_id = uuid.uuid4()
    print(f"[INFO] modeling_id = {modeling_id}")

    if args.replace_members:
        print("[WARN] replace-members ON: deleting previous rows for this modeling_id (should be none) ...")
        maybe_delete_existing(engine, modeling_id)

    # members df
    df_members = pd.DataFrame(
        {
            "incident_number": df_emb["incident_number"].astype(str),
            "tgl_submit": df_emb["tgl_submit"],
            "cluster_id": labels.astype(int),
            "is_noise": (labels == -1),
            "prob": probs.astype(float),
            "outlier_score": outlier.astype(float),
        }
    )
    df_members.insert(0, "modeling_id", modeling_id)

    # enrich from incident_semantik (best effort)
    try:
        df_members = enrich_members_with_semantik(engine, df_members)
    except Exception:
        # tetap jalan walau enrich gagal
        for c in ["site", "modul", "sub_modul"]:
            if c not in df_members.columns:
                df_members[c] = None

    # clusters df
    if n_clusters > 0:
        df_clusters = (
            df_members[~df_members["is_noise"]]
            .groupby(["modeling_id", "cluster_id"], as_index=False)
            .agg(
                cluster_size=("incident_number", "count"),
                avg_prob=("prob", "mean"),
                avg_outlier_score=("outlier_score", "mean"),
            )
            .sort_values(["cluster_size", "cluster_id"], ascending=[False, True])
        )
    else:
        df_clusters = pd.DataFrame(columns=["modeling_id", "cluster_id", "cluster_size", "avg_prob", "avg_outlier_score"])

    # save runs row
    params = {
        "embedding_run_id": embedding_run_id,
        "limit": int(args.limit),
        "min_cluster_size": int(args.min_cluster_size),
        "min_samples": int(args.min_samples),
        "metric": str(args.metric),
        "cluster_selection_method": str(args.cluster_selection_method),
        "pca_dim": int(args.pca_dim),
        "seed": int(args.seed),
    }
    notes = (args.notes or "").strip()
    if not notes:
        notes = f"HDBSCAN semantik | limit={args.limit} | mcs={args.min_cluster_size} ms={args.min_samples}"

    print("[INFO] Saving results to DB ...")
    save_runs_row(
        engine,
        modeling_id=modeling_id,
        embedding_run_id=embedding_run_id,
        n_rows=n_rows,
        n_clusters=n_clusters,
        n_noise=n_noise,
        silhouette=sil,
        dbi=dbi,
        params=params,
        notes=notes,
    )
    save_clusters(engine, df_clusters)
    save_members(engine, df_members[[
        "modeling_id","cluster_id","is_noise","incident_number","tgl_submit","site","modul","sub_modul","prob","outlier_score"
    ]])

    print("[DONE] Saved:")
    print(f"  runs    : {SCHEMA}.{T_RUNS}")
    print(f"  clusters: {SCHEMA}.{T_CLUSTERS}")
    print(f"  members : {SCHEMA}.{T_MEMBERS}")
    print()
    print("[NEXT] Quick checks:")
    print(f"  SELECT * FROM {SCHEMA}.{T_RUNS} ORDER BY run_time DESC LIMIT 5;")
    print(f"  SELECT cluster_id, cluster_size FROM {SCHEMA}.{T_CLUSTERS} WHERE modeling_id='{modeling_id}' ORDER BY cluster_size DESC LIMIT 20;")
    print(f"  SELECT COUNT(*) FROM {SCHEMA}.{T_MEMBERS} WHERE modeling_id='{modeling_id}';")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
