# terminal/run_modeling_semantik_hdbscan.py
# ============================================================
# Offline Runner — Clustering Semantik (HDBSCAN)
# FINAL PATCHED (Hardcoded DB + Auto-create tables + Safe test limit default)
#
# Source (embedding):
#   - lasis_djp.semantik_embedding_runs
#   - lasis_djp.semantik_embedding_vectors  (embedding_json JSONB)
#
# Optional enrich (metadata/text):
#   - lasis_djp.incident_semantik (site, modul, sub_modul, tgl_submit/tgl_semantik, dll)
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
# Notes:
# - Pastikan package "hdbscan" terinstall: pip install hdbscan
# - Jika ingin UMAP, itu untuk viewer saja, bukan clustering.
# ============================================================

from __future__ import annotations

import argparse
import json
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
    ap.add_argument(
        "--embedding-run-id",
        default=None,
        help="run_id dari semantik_embedding_runs (uuid). Default: latest",
    )
    ap.add_argument("--limit", type=int, default=5000, help="Jumlah baris embedding untuk test. 0=full.")
    ap.add_argument("--seed", type=int, default=42)

    # (opsional) pakai modeling_id tertentu agar bisa re-run & replace
    ap.add_argument("--modeling-id", default=None, help="Opsional: pakai modeling_id tertentu (uuid).")
    ap.add_argument(
        "--replace-members",
        action="store_true",
        help="Jika ON dan modeling-id diberikan, hapus dulu hasil untuk modeling_id tsb (members+clusters+runs).",
    )

    # HDBSCAN params
    ap.add_argument("--min-cluster-size", type=int, default=5)
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument(
        "--metric",
        default="euclidean",
        help="euclidean (recommended if embeddings are L2-normalized). cosine juga bisa.",
    )
    ap.add_argument("--cluster-selection-method", choices=["eom", "leaf"], default="eom")

    # Optional dimensionality reduction (PCA) for speed/noise control
    ap.add_argument("--pca-dim", type=int, default=0, help="0=off. contoh 50 untuk percepat HDBSCAN")

    ap.add_argument("--notes", default="")
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

    CREATE INDEX IF NOT EXISTS idx_{T_MEMBERS}_incident
        ON {SCHEMA}.{T_MEMBERS}(incident_number);

    CREATE INDEX IF NOT EXISTS idx_{T_RUNS}_runtime_desc
        ON {SCHEMA}.{T_RUNS}(run_time DESC);
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


def load_embeddings(engine: Engine, embedding_run_id: str, limit: int) -> pd.DataFrame:
    """
    PATCH: jangan ambil tgl_submit dari vectors (sering tidak ada).
    Ambil minimal: incident_number, embedding_dim, embedding_json, (opsional n_chars).
    """
    limit_sql = f"LIMIT {int(limit)}" if int(limit) > 0 else ""
    sql = f"""
    SELECT
        v.incident_number,
        v.embedding_dim,
        v.embedding_json
    FROM {SCHEMA}.{T_EMB_VECS} v
    WHERE v.run_id = :rid
    ORDER BY v.incident_number
    {limit_sql}
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params={"rid": embedding_run_id})
    if df.empty:
        raise RuntimeError("Embedding vectors kosong untuk run_id tersebut.")
    return df


def enrich_members_with_semantik(engine: Engine, df_members: pd.DataFrame) -> pd.DataFrame:
    """
    Tambahkan tgl_submit/site/modul/sub_modul dari incident_semantik (best-effort).
    Jika tabel/kolom tidak ada, tetap jalan.
    """
    chk_sql = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = :schema AND table_name = :table
    """

    with engine.connect() as conn:
        cols = [r[0] for r in conn.execute(text(chk_sql), {"schema": SCHEMA, "table": T_SEMANTIK}).fetchall()]

    if "incident_number" not in cols:
        return df_members

    # Beberapa skema menyimpan tgl_submit asli di incident_semantik; sebagian pakai tgl_semantik.
    time_col = "tgl_submit" if "tgl_submit" in cols else ("tgl_semantik" if "tgl_semantik" in cols else None)

    select_cols = ["incident_number"]
    for c in ["site", "modul", "sub_modul"]:
        if c in cols:
            select_cols.append(c)
    if time_col:
        select_cols.append(time_col)

    sql = f"SELECT {', '.join(select_cols)} FROM {SCHEMA}.{T_SEMANTIK}"
    with engine.connect() as conn:
        df_meta = pd.read_sql(text(sql), conn)

    if time_col and time_col != "tgl_submit":
        df_meta = df_meta.rename(columns={time_col: "tgl_submit"})

    out = df_members.merge(df_meta, on="incident_number", how="left")

    # ensure columns exist
    for c in ["tgl_submit", "site", "modul", "sub_modul"]:
        if c not in out.columns:
            out[c] = None

    return out


def _parse_vec(v) -> Optional[np.ndarray]:
    if v is None:
        return None
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return None
    if not isinstance(v, (list, tuple)):
        return None
    if len(v) == 0:
        return None
    try:
        arr = np.asarray(v, dtype=np.float32)
    except Exception:
        return None
    if arr.ndim != 1:
        return None
    return arr


def to_matrix_from_json(df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame, int]:
    """
    PATCH: validasi embedding_json:
    - drop NULL/invalid
    - drop dim mismatch
    Return:
      X, df_valid (aligned), n_dropped
    """
    target_dim = int(df["embedding_dim"].dropna().iloc[0])
    keep_rows = []
    vectors: List[np.ndarray] = []
    dropped = 0

    for idx, row in df.iterrows():
        vec = _parse_vec(row["embedding_json"])
        if vec is None:
            dropped += 1
            continue
        if int(vec.shape[0]) != target_dim:
            dropped += 1
            continue
        keep_rows.append(idx)
        vectors.append(vec)

    if not vectors:
        raise RuntimeError("Semua embedding_json invalid / kosong. Tidak bisa membentuk matrix.")

    df_valid = df.loc[keep_rows].copy()
    X = np.vstack(vectors).astype(np.float32)
    return X, df_valid, dropped


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
        raise RuntimeError("Package 'hdbscan' belum terinstall. Jalankan: pip install hdbscan") from e

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
            ON CONFLICT (modeling_id) DO UPDATE SET
              run_time = EXCLUDED.run_time,
              embedding_run_id = EXCLUDED.embedding_run_id,
              n_rows = EXCLUDED.n_rows,
              n_clusters = EXCLUDED.n_clusters,
              n_noise = EXCLUDED.n_noise,
              silhouette = EXCLUDED.silhouette,
              dbi = EXCLUDED.dbi,
              params_json = EXCLUDED.params_json,
              notes = EXCLUDED.notes
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
            },
        )


def save_clusters(engine: Engine, df_clusters: pd.DataFrame) -> None:
    if df_clusters.empty:
        return
    records = df_clusters.to_dict(orient="records")
    for r in records:
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
            records,
        )


def save_members(engine: Engine, df_members: pd.DataFrame) -> None:
    if df_members.empty:
        return
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
            records,
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

    # modeling_id (baru atau user-specified)
    if args.modeling_id:
        modeling_id = uuid.UUID(str(args.modeling_id))
    else:
        modeling_id = uuid.uuid4()
    print(f"[INFO] modeling_id = {modeling_id}")

    if args.replace_members and args.modeling_id:
        print("[WARN] replace-members ON: deleting existing rows for provided modeling_id ...")
        maybe_delete_existing(engine, modeling_id)

    print(f"[INFO] Loading embeddings (limit={args.limit}) ...")
    df_emb_raw = load_embeddings(engine, embedding_run_id, args.limit)
    print(f"[INFO] Loaded {len(df_emb_raw):,} rows (raw)")

    print("[INFO] Building matrix from embedding_json (with validation) ...")
    X, df_emb, dropped = to_matrix_from_json(df_emb_raw)
    if dropped:
        print(f"[WARN] Dropped {dropped:,} rows due to invalid/null/mismatched embeddings.")
    n_rows = len(df_emb)
    print(f"[INFO] Using {n_rows:,} rows (valid)")
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
    sil: Optional[float] = None
    dbi: Optional[float] = None
    mask = labels != -1

    if n_clusters >= 2 and int(np.sum(mask)) >= 10:
        # Silhouette: coba pakai metric yang sama jika didukung; fallback euclidean
        try:
            sil = float(silhouette_score(X[mask], labels[mask], metric=str(args.metric)))
        except Exception:
            try:
                sil = float(silhouette_score(X[mask], labels[mask], metric="euclidean"))
            except Exception:
                sil = None

        # DBI: defensible hanya di euclidean space
        if str(args.metric).lower() == "euclidean":
            try:
                dbi = float(davies_bouldin_score(X[mask], labels[mask]))
            except Exception:
                dbi = None
        else:
            dbi = None

    # members df (align with df_emb valid)
    df_members = pd.DataFrame(
        {
            "incident_number": df_emb["incident_number"].astype(str),
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
    except Exception as e:
        print(f"[WARN] Enrich incident_semantik failed: {e}")
        for c in ["tgl_submit", "site", "modul", "sub_modul"]:
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
        df_clusters = pd.DataFrame(
            columns=["modeling_id", "cluster_id", "cluster_size", "avg_prob", "avg_outlier_score"]
        )

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
        "dropped_invalid_embeddings": int(dropped),
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

    df_members_out = df_members[
        ["modeling_id", "cluster_id", "is_noise", "incident_number", "tgl_submit", "site", "modul", "sub_modul", "prob", "outlier_score"]
    ].copy()
    save_members(engine, df_members_out)

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
