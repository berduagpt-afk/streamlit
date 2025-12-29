# run_modeling_sintaksis_offline_final.py
# ============================================================
# Offline Modeling Sintaksis (FINAL – DB SETTING DI SCRIPT)
# TF-IDF + Cosine Similarity Threshold + Temporal Window
#
# Source  : lasis_djp.incident_tfidf_vectors
# Auto run_id:
#   - Jika --run-id tidak diisi -> ambil run_id terbaru dari lasis_djp.incident_tfidf_runs
#
# Output  : lasis_djp.modeling_sintaksis_*
#   - modeling_sintaksis_runs
#   - modeling_sintaksis_clusters
#   - modeling_sintaksis_members
#
# Perbaikan Warning Pandas:
#   1) Series.view deprecated  -> ganti ke .astype("int64")
#   2) pd.factorize input list -> ganti input ke np.asarray(...)
# ============================================================

from __future__ import annotations

import json
import uuid
import time
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


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
    min_cluster_size: int = 10

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


def ensure_output_tables(engine: Engine, cfg: Config) -> None:
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

    uf = UnionFind(n)

    # ========================================================
    # FIX WARNING #1: Series.view deprecated -> use astype("int64")
    # ========================================================
    t = out["tgl_submit"]
    t_floor = t.dt.floor("D")
    # NaT sentinel agar selalu gagal temporal window
    # NOTE: astype("int64") aman untuk datetime64[ns] pada pandas baru
    t_days = np.where(
        t_floor.notna(),
        (t_floor.astype("int64") // 86_400_000_000_000).astype(np.int64),
        np.int64(-10**18)
    )

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
        for pos in range(1, k):  # skip self
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

    roots_list = [uf.find(i) for i in range(n)]

    # ========================================================
    # FIX WARNING #2: pd.factorize(list) deprecated
    # -> wrap with np.asarray(...) so it's ndarray
    # ========================================================
    roots_arr = np.asarray(roots_list, dtype=np.int64)
    out["cluster_id"] = pd.factorize(roots_arr)[0].astype(np.int64)

    return out


# ============================================================
# SAVE OUTPUT
# ============================================================

def save_results(engine: Engine, df: pd.DataFrame, cfg: Config, modeling_id: str) -> None:
    df = df.copy()

    df["cluster_size"] = df.groupby("cluster_id")["incident_number"].transform("count").astype(int)
    df["is_recurring"] = (df["cluster_size"] >= int(cfg.min_cluster_size)).astype(np.int16)

    # MEMBERS
    members = df[[
        "cluster_id", "incident_number", "tgl_submit", "site", "assignee",
        "modul", "sub_modul", "is_recurring"
    ]].copy()
    members.insert(0, "modeling_id", modeling_id)

    # CLUSTERS
    clusters = (
        df.groupby("cluster_id", as_index=False)
        .agg(
            cluster_size=("incident_number", "count"),
            min_time=("tgl_submit", "min"),
            max_time=("tgl_submit", "max"),
        )
    )
    clusters["span_days"] = (clusters["max_time"] - clusters["min_time"]).dt.days
    clusters["is_recurring"] = (clusters["cluster_size"] >= int(cfg.min_cluster_size)).astype(np.int16)
    clusters.insert(0, "modeling_id", modeling_id)

    # RUNS
    runs = pd.DataFrame([{
        "modeling_id": modeling_id,
        "run_time": pd.Timestamp.utcnow(),
        "approach": "tfidf_cosine_threshold_temporal",
        "params_json": json.dumps(asdict(cfg)),
        "notes": f"tfidf_run_id={cfg.run_id_tfidf}"
    }])

    with engine.begin() as conn:
        conn.execute(
            text(f"""
                INSERT INTO {cfg.schema}.{cfg.out_runs}
                (modeling_id, run_time, approach, params_json, notes)
                VALUES (:modeling_id, :run_time, :approach, CAST(:params_json AS jsonb), :notes)
            """),
            runs.to_dict(orient="records")[0]
        )

        members.to_sql(cfg.out_members, conn, schema=cfg.schema, if_exists="append",
                       index=False, method="multi", chunksize=5000)
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
    p.add_argument("--window-days", type=int, default=30)
    p.add_argument("--knn-k", type=int, default=25)
    p.add_argument("--min-cluster-size", type=int, default=2)
    p.add_argument("--limit", type=int, default=None, help="Untuk uji coba.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config(
        schema=args.schema,
        threshold=args.threshold,
        window_days=args.window_days,
        knn_k=args.knn_k,
        min_cluster_size=args.min_cluster_size,
        limit=args.limit
    )

    t0 = time.time()
    modeling_id = str(uuid.uuid4())

    engine = get_engine()
    ensure_output_tables(engine, cfg)

    # auto pick run_id jika tidak diberikan
    if args.run_id:
        cfg.run_id_tfidf = args.run_id
        print(f"[INFO] run_id dipilih manual: {cfg.run_id_tfidf}")
    else:
        cfg.run_id_tfidf = get_latest_tfidf_run_id(engine, cfg)
        print(f"[AUTO] run_id TF-IDF terbaru dipilih: {cfg.run_id_tfidf}")

    print(f"[INFO] modeling_id={modeling_id}")
    print(f"[INFO] threshold={cfg.threshold} | window_days={cfg.window_days} | knn_k={cfg.knn_k} | min_cluster_size={cfg.min_cluster_size}")

    df = load_data(engine, cfg)
    print(f"[INFO] rows={len(df):,}")

    X, vocab = build_csr_matrix(df["tfidf_json"].tolist())
    print(f"[INFO] X shape={X.shape} | nnz={X.nnz:,} | vocab={len(vocab):,}")

    df2 = run_clustering(df, X, cfg)

    n_clusters = df2["cluster_id"].nunique()
    rec_mask = (df2.groupby("cluster_id")["incident_number"].transform("count") >= cfg.min_cluster_size)
    print(f"[INFO] clusters={n_clusters:,} | recurring_tickets={int(rec_mask.sum()):,}")

    save_results(engine, df2, cfg, modeling_id)

    print(f"[DONE] saved to {cfg.schema}.{cfg.out_members} / {cfg.out_clusters} / {cfg.out_runs}")
    print(f"[DONE] elapsed={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
