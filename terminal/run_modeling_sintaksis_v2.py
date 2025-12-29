# run_modeling_sintaksis_offline_final_SELFHEAL.py
# ============================================================
# Offline Modeling Sintaksis (SELF-HEALING FINAL)
# TF-IDF + Cosine kNN + Threshold + Temporal Window + NOISE policy
#
# Fix utama untuk error:
# - Auto ALTER TABLE modeling_sintaksis_runs untuk menambahkan kolom baru
#   (tfidf_run_id, threshold, window_days, knn_k, dst) jika belum ada.
#
# Output:
# - lasis_djp.modeling_sintaksis_runs
# - lasis_djp.modeling_sintaksis_clusters
# - lasis_djp.modeling_sintaksis_members
# ============================================================

from __future__ import annotations

import time
import uuid
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from psycopg2.extras import Json


# ============================================================
# ✅ DB SETTING (ISI SEKALI DI SINI)
# ============================================================
DB = {
    "host": "localhost",
    "port": 5432,
    "database": "incident_djp",
    "user": "postgres",
    "password": "admin*123",
}

NOISE_ID = -1  # cluster_id untuk non-recurring / singleton


# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    schema: str = "lasis_djp"
    source_table: str = "incident_tfidf_vectors"
    runs_table: str = "incident_tfidf_runs"  # untuk auto run_id terbaru

    out_runs: str = "modeling_sintaksis_runs"
    out_clusters: str = "modeling_sintaksis_clusters"
    out_members: str = "modeling_sintaksis_members"

    run_id_tfidf: str = ""  # bisa kosong -> auto
    threshold: float = 0.80
    window_days: int = 30
    knn_k: int = 25
    min_cluster_size: int = 10  # konsisten dengan default CLI

    # behavior
    save_noise_members: bool = True     # simpan noise di members (cluster_id=-1)
    save_noise_clusters: bool = False   # noise TIDAK masuk tabel clusters

    # opsional untuk tes
    limit: Optional[int] = None


# ============================================================
# DB
# ============================================================

def get_engine() -> Engine:
    url = (
        f"postgresql+psycopg2://{DB['user']}:{DB['password']}"
        f"@{DB['host']}:{DB['port']}/{DB['database']}"
    )
    return create_engine(url, pool_pre_ping=True)


def ensure_base_tables(engine: Engine, cfg: Config) -> None:
    """
    Pastikan schema & tabel output ada (minimal).
    Kolom tambahan akan di-ALTER via ensure_runs_columns().
    """
    ddl = f"""
    CREATE SCHEMA IF NOT EXISTS {cfg.schema};

    CREATE TABLE IF NOT EXISTS {cfg.schema}.{cfg.out_runs} (
        modeling_id      uuid PRIMARY KEY,
        run_time         timestamptz NOT NULL,
        approach         text NOT NULL,
        params_json      jsonb NOT NULL,
        notes            text NULL
    );

    CREATE TABLE IF NOT EXISTS {cfg.schema}.{cfg.out_clusters} (
        modeling_id     uuid NOT NULL,
        cluster_id      bigint NOT NULL,
        cluster_size    integer NOT NULL,
        is_recurring    smallint NOT NULL,
        min_time        timestamp NULL,
        max_time        timestamp NULL,
        span_days       integer NULL,
        PRIMARY KEY (modeling_id, cluster_id)
    );

    CREATE TABLE IF NOT EXISTS {cfg.schema}.{cfg.out_members} (
        modeling_id     uuid NOT NULL,
        cluster_id      bigint NOT NULL,
        incident_number text NOT NULL,
        tgl_submit      timestamp NULL,
        site            text NULL,
        assignee        text NULL,
        modul           text NULL,
        sub_modul       text NULL,
        is_recurring    smallint NOT NULL,
        PRIMARY KEY (modeling_id, incident_number)
    );

    CREATE INDEX IF NOT EXISTS idx_{cfg.out_members}_model_cluster
      ON {cfg.schema}.{cfg.out_members} (modeling_id, cluster_id);

    CREATE INDEX IF NOT EXISTS idx_{cfg.out_members}_model_time
      ON {cfg.schema}.{cfg.out_members} (modeling_id, tgl_submit);

    CREATE INDEX IF NOT EXISTS idx_{cfg.out_clusters}_model_rec
      ON {cfg.schema}.{cfg.out_clusters} (modeling_id, is_recurring, cluster_size);
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def ensure_runs_columns(engine: Engine, cfg: Config) -> None:
    """
    SELF-HEAL:
    Tambahkan kolom-kolom baru pada modeling_sintaksis_runs jika belum ada.
    Ini yang memperbaiki error:
      psycopg2.errors.UndefinedColumn: column "tfidf_run_id" does not exist
    """
    alter = f"""
    ALTER TABLE {cfg.schema}.{cfg.out_runs}
      ADD COLUMN IF NOT EXISTS tfidf_run_id uuid NULL,
      ADD COLUMN IF NOT EXISTS threshold double precision NULL,
      ADD COLUMN IF NOT EXISTS window_days integer NULL,
      ADD COLUMN IF NOT EXISTS knn_k integer NULL,
      ADD COLUMN IF NOT EXISTS min_cluster_size integer NULL,
      ADD COLUMN IF NOT EXISTS n_rows bigint NULL,
      ADD COLUMN IF NOT EXISTS n_clusters_all bigint NULL,
      ADD COLUMN IF NOT EXISTS n_clusters_recurring bigint NULL,
      ADD COLUMN IF NOT EXISTS n_noise_tickets bigint NULL,
      ADD COLUMN IF NOT EXISTS vocab_size bigint NULL,
      ADD COLUMN IF NOT EXISTS nnz bigint NULL,
      ADD COLUMN IF NOT EXISTS elapsed_sec double precision NULL;
    """
    with engine.begin() as conn:
        conn.execute(text(alter))

    # OPTIONAL index untuk evaluasi
    idx = f"""
    CREATE INDEX IF NOT EXISTS idx_model_sintaksis_runs_thr_win_time
    ON {cfg.schema}.{cfg.out_runs} (threshold, window_days, run_time DESC);

    CREATE INDEX IF NOT EXISTS idx_model_sintaksis_runs_tfidf_run
    ON {cfg.schema}.{cfg.out_runs} (tfidf_run_id);
    """
    with engine.begin() as conn:
        conn.execute(text(idx))


def get_latest_tfidf_run_id(engine: Engine, cfg: Config) -> str:
    sql = f"""
    SELECT run_id
    FROM {cfg.schema}.{cfg.runs_table}
    ORDER BY run_time DESC
    LIMIT 1
    """
    df = pd.read_sql(text(sql), engine)
    if df.empty or df.loc[0, "run_id"] is None:
        raise RuntimeError(f"Tidak ditemukan run_id pada {cfg.schema}.{cfg.runs_table}.")
    return str(df.loc[0, "run_id"])


# ============================================================
# LOAD DATA
# ============================================================

def load_data(engine: Engine, cfg: Config) -> pd.DataFrame:
    lim = f"LIMIT {int(cfg.limit)}" if cfg.limit else ""
    sql = f"""
    SELECT
        run_id,
        incident_number,
        tgl_submit,
        site,
        assignee,
        modul,
        sub_modul,
        tfidf_json
    FROM {cfg.schema}.{cfg.source_table}
    WHERE run_id = :run_id
      AND tfidf_json IS NOT NULL
    ORDER BY tgl_submit NULLS LAST
    {lim}
    """
    df = pd.read_sql(text(sql), engine, params={"run_id": cfg.run_id_tfidf})
    if df.empty:
        raise RuntimeError(f"Data kosong untuk run_id={cfg.run_id_tfidf}.")
    df["tgl_submit"] = pd.to_datetime(df["tgl_submit"], errors="coerce")

    # jika tz-aware, samakan ke UTC lalu tz-naive
    try:
        if getattr(df["tgl_submit"].dt, "tz", None) is not None:
            df["tgl_submit"] = df["tgl_submit"].dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        pass

    return df


# ============================================================
# BUILD TF-IDF MATRIX (CSR) from tfidf_json (term->weight)
# ============================================================

def build_csr_matrix(tfidf_json_list: List[Any]) -> Tuple[csr_matrix, Dict[str, int]]:
    vocab: Dict[str, int] = {}
    docs: List[Dict[str, float]] = []

    for x in tfidf_json_list:
        d = x if isinstance(x, dict) else {}
        docs.append(d)
        for term in d.keys():
            if term not in vocab:
                vocab[term] = len(vocab)

    indptr = [0]
    indices: List[int] = []
    data: List[float] = []

    for d in docs:
        for term, val in d.items():
            j = vocab.get(term)
            if j is None:
                continue
            try:
                fv = float(val)
            except Exception:
                continue
            if fv == 0.0:
                continue
            indices.append(j)
            data.append(fv)
        indptr.append(len(indices))

    X = csr_matrix(
        (np.array(data, dtype=np.float32),
         np.array(indices, dtype=np.int32),
         np.array(indptr, dtype=np.int32)),
        shape=(len(docs), len(vocab)),
        dtype=np.float32
    )
    return X, vocab


# ============================================================
# UNION FIND
# ============================================================

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


# ============================================================
# CLUSTERING: Cosine Threshold + Temporal Window via kNN
# ============================================================

def run_clustering(df: pd.DataFrame, X: csr_matrix, cfg: Config) -> pd.DataFrame:
    n = X.shape[0]
    out = df.copy()

    if n < 2:
        out["cluster_id"] = 0
        return out

    # Normalisasi L2 (recommended)
    normalize(X, norm="l2", axis=1, copy=False)

    # waktu -> hari (aman)
    t_floor = out["tgl_submit"].dt.floor("D")
    t_days = np.full(len(t_floor), -10**18, dtype=np.int64)
    mask = t_floor.notna().to_numpy()
    if mask.any():
        t_days[mask] = t_floor.to_numpy(dtype="datetime64[D]")[mask].astype(np.int64)

    uf = UnionFind(n)

    k = int(max(2, cfg.knn_k))
    k = int(min(k, n))

    nn = NearestNeighbors(
        n_neighbors=k,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1
    )
    nn.fit(X)

    dist, idx = nn.kneighbors(X, return_distance=True)
    sim = 1.0 - dist

    thr = float(cfg.threshold)
    win = int(cfg.window_days)

    for i in range(n):
        ti = int(t_days[i])
        for pos in range(1, k):
            j = int(idx[i, pos])
            s = float(sim[i, pos])
            if s < thr:
                continue

            tj = int(t_days[j])
            if ti < -10**17 or tj < -10**17:
                continue

            if abs(ti - tj) > win:
                continue

            uf.union(i, j)

    roots_arr = np.asarray([uf.find(i) for i in range(n)], dtype=np.int64)
    out["cluster_id"] = pd.factorize(roots_arr)[0].astype(np.int64)
    return out


def apply_noise_policy(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Cluster kecil (< min_cluster_size) menjadi NOISE (-1),
    sehingga jumlah cluster bermakna tidak meledak.
    """
    df = df.copy()
    df["cluster_size"] = df.groupby("cluster_id")["incident_number"].transform("count").astype(int)
    df["is_recurring"] = (df["cluster_size"] >= int(cfg.min_cluster_size)).astype(np.int16)
    df.loc[df["is_recurring"] == 0, "cluster_id"] = NOISE_ID
    return df


# ============================================================
# SAVE OUTPUT
# ============================================================

def save_results(
    engine: Engine,
    df: pd.DataFrame,
    cfg: Config,
    modeling_id: str,
    vocab_size: int,
    nnz: int,
    elapsed_sec: float,
) -> None:
    df = df.copy()

    # MEMBERS
    members_all = df[[
        "cluster_id", "incident_number", "tgl_submit", "site", "assignee",
        "modul", "sub_modul", "is_recurring"
    ]].copy()
    members_all.insert(0, "modeling_id", modeling_id)

    if cfg.save_noise_members:
        members = members_all
    else:
        members = members_all[members_all["cluster_id"] != NOISE_ID].copy()

    members = members.drop_duplicates(subset=["modeling_id", "incident_number"])

    # CLUSTERS (recurring only by default)
    df_for_clusters = df if cfg.save_noise_clusters else df[df["cluster_id"] != NOISE_ID]

    clusters = (
        df_for_clusters.groupby("cluster_id", as_index=False)
        .agg(
            cluster_size=("incident_number", "count"),
            min_time=("tgl_submit", "min"),
            max_time=("tgl_submit", "max"),
        )
    )
    if not clusters.empty:
        clusters["span_days"] = (clusters["max_time"] - clusters["min_time"]).dt.days
        clusters["is_recurring"] = (clusters["cluster_size"] >= int(cfg.min_cluster_size)).astype(np.int16)
        clusters.insert(0, "modeling_id", modeling_id)

    # RUN METRICS
    n_rows = int(len(df))
    n_noise = int((df["cluster_id"] == NOISE_ID).sum())
    n_clusters_all = int(df["cluster_id"].nunique())
    n_clusters_rec = int(df[df["cluster_id"] != NOISE_ID]["cluster_id"].nunique())

    run_time = pd.Timestamp.now(tz="UTC")
    params_obj = asdict(cfg)

    runs_row = {
        "modeling_id": modeling_id,
        "run_time": run_time,
        "approach": "tfidf_cosine_threshold_temporal",

        "tfidf_run_id": cfg.run_id_tfidf,
        "threshold": float(cfg.threshold),
        "window_days": int(cfg.window_days),
        "knn_k": int(cfg.knn_k),
        "min_cluster_size": int(cfg.min_cluster_size),

        "n_rows": n_rows,
        "n_clusters_all": n_clusters_all,
        "n_clusters_recurring": n_clusters_rec,
        "n_noise_tickets": n_noise,
        "vocab_size": int(vocab_size),
        "nnz": int(nnz),
        "elapsed_sec": float(elapsed_sec),

        "params_json": params_obj,
        "notes": f"tfidf_run_id={cfg.run_id_tfidf} | noise_id={NOISE_ID}"
    }

    # IMPORTANT: kolom bisa belum ada -> pastikan ALTER sudah dilakukan
    ensure_runs_columns(engine, cfg)

    with engine.begin() as conn:
        conn.execute(
            text(f"""
                INSERT INTO {cfg.schema}.{cfg.out_runs}
                (modeling_id, run_time, approach,
                 tfidf_run_id, threshold, window_days, knn_k, min_cluster_size,
                 n_rows, n_clusters_all, n_clusters_recurring, n_noise_tickets,
                 vocab_size, nnz, elapsed_sec,
                 params_json, notes)
                VALUES
                (:modeling_id, :run_time, :approach,
                 CAST(:tfidf_run_id AS uuid), :threshold, :window_days, :knn_k, :min_cluster_size,
                 :n_rows, :n_clusters_all, :n_clusters_recurring, :n_noise_tickets,
                 :vocab_size, :nnz, :elapsed_sec,
                 :params_json, :notes)
            """),
            {**runs_row, "params_json": Json(runs_row["params_json"])}
        )

        members.to_sql(cfg.out_members, conn, schema=cfg.schema, if_exists="append",
                       index=False, method="multi", chunksize=5000)

        if not clusters.empty:
            clusters.to_sql(cfg.out_clusters, conn, schema=cfg.schema, if_exists="append",
                            index=False, method="multi", chunksize=5000)


# ============================================================
# MAIN
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default=None, help="Opsional. Jika kosong -> auto ambil run TF-IDF terbaru.")
    p.add_argument("--schema", default="lasis_djp")
    p.add_argument("--threshold", type=float, default=0.80)
    p.add_argument("--window-days", type=int, default=15)
    p.add_argument("--knn-k", type=int, default=25)

    # konsisten dengan Config
    p.add_argument("--min-cluster-size", type=int, default=10)

    p.add_argument("--limit", type=int, default=None, help="Untuk uji coba.")
    p.add_argument("--save-noise-members", action="store_true", help="Jika set, tiket noise disimpan di members.")
    p.add_argument("--save-noise-clusters", action="store_true", help="Jika set, noise juga masuk tabel clusters.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config(
        schema=args.schema,
        threshold=args.threshold,
        window_days=args.window_days,
        knn_k=args.knn_k,
        min_cluster_size=args.min_cluster_size,
        limit=args.limit,
        save_noise_members=bool(args.save_noise_members),
        save_noise_clusters=bool(args.save_noise_clusters),
    )

    t0 = time.time()
    modeling_id = str(uuid.uuid4())

    engine = get_engine()

    # base table minimal + self-heal columns
    ensure_base_tables(engine, cfg)
    ensure_runs_columns(engine, cfg)

    # auto pick run_id jika tidak diberikan
    if args.run_id:
        cfg.run_id_tfidf = args.run_id
        print(f"[INFO] run_id dipilih manual: {cfg.run_id_tfidf}")
    else:
        cfg.run_id_tfidf = get_latest_tfidf_run_id(engine, cfg)
        print(f"[AUTO] run_id TF-IDF terbaru dipilih: {cfg.run_id_tfidf}")

    print(f"[INFO] modeling_id={modeling_id}")
    print(f"[INFO] threshold={cfg.threshold} | window_days={cfg.window_days} | knn_k={cfg.knn_k} | min_cluster_size={cfg.min_cluster_size}")
    print(f"[INFO] noise_policy: NOISE_ID={NOISE_ID} | save_noise_members={cfg.save_noise_members} | save_noise_clusters={cfg.save_noise_clusters}")

    df = load_data(engine, cfg)
    print(f"[INFO] rows={len(df):,}")

    X, vocab = build_csr_matrix(df["tfidf_json"].tolist())
    print(f"[INFO] X shape={X.shape} | nnz={X.nnz:,} | vocab={len(vocab):,}")

    df2 = run_clustering(df, X, cfg)
    df3 = apply_noise_policy(df2, cfg)

    n_noise = int((df3["cluster_id"] == NOISE_ID).sum())
    n_clusters_all = int(df3["cluster_id"].nunique())
    n_clusters_rec = int(df3[df3["cluster_id"] != NOISE_ID]["cluster_id"].nunique())
    print(f"[INFO] clusters_all(groups incl noise)={n_clusters_all:,} | clusters_recurring={n_clusters_rec:,} | noise_tickets={n_noise:,}")

    elapsed = time.time() - t0

    save_results(
        engine=engine,
        df=df3,
        cfg=cfg,
        modeling_id=modeling_id,
        vocab_size=len(vocab),
        nnz=int(X.nnz),
        elapsed_sec=float(elapsed),
    )

    print(f"[DONE] saved to {cfg.schema}.{cfg.out_members} / {cfg.out_clusters} / {cfg.out_runs}")
    print(f"[DONE] elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
