# ============================================================
# Offline Modeling Sintaksis — FINAL PATCHED + Cosine Similarity Distribution (Pra-Threshold)
# TF-IDF (DB) + Cosine kNN + Threshold Grid + Sim Distribution
#
# Output tables:
# - lasis_djp.modeling_sintaksis_runs
# - lasis_djp.modeling_sintaksis_clusters
# - lasis_djp.modeling_sintaksis_members
# - lasis_djp.modeling_sintaksis_cosine_dist   (distribusi similarity pra-threshold)
#
# PATCHES (FINAL):
# 1) ensure_tables() aman untuk multi-statement DDL:
#    - eksekusi 1 statement per execute (split ';' + exec_driver_sql)
# 2) Hilangkan warning buffer dtype mismatch:
#    - CSR matrix float64 (dtype=np.float64) + guard cast sebelum kNN
# 3) n_singletons dihitung sebagai jumlah cluster singleton (bukan jumlah baris via transform)
# 4) build_csr() robust:
#    - handle tfidf_json bertipe dict / str(JSON) / None
#    - filter NaN/Inf
# 5) Konsistensi kNN:
#    - Distribusi dan clustering memakai definisi neighbor yang sama:
#      k_eff = min(knn_k + 1, n) lalu skip self (jpos=0)
#      => neighbor efektif = knn_k
# 6) CSR dirapikan:
#    - X.sum_duplicates()
# 7) Normalisasi L2 dilakukan SEKALI di main (in-place)
# 8) tfidf_run_id disimpan sebagai TEXT (bukan UUID) agar kompatibel dengan run_id string
# ============================================================

from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from psycopg2.extras import Json
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# ============================================================
# DB CONFIG (sesuaikan)
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
    runs_table: str = "incident_tfidf_runs"

    out_runs: str = "modeling_sintaksis_runs"
    out_clusters: str = "modeling_sintaksis_clusters"
    out_members: str = "modeling_sintaksis_members"

    # cosine distribution (pra-threshold)
    out_cosine_dist: str = "modeling_sintaksis_cosine_dist"

    run_id_tfidf: str = ""
    knn_k: int = 25
    window_days: int = 30          # metadata
    min_cluster_size: int = 10     # metadata

    # distribusi
    dist_bins: int = 50
    dist_sample_max: int = 1_000_000  # batasi sampel similarity utk hemat RAM (opsional)

    limit: Optional[int] = None


# ============================================================
# DB HELPERS
# ============================================================

def get_engine() -> Engine:
    url = (
        f"postgresql+psycopg2://{DB['user']}:{DB['password']}"
        f"@{DB['host']}:{DB['port']}/{DB['database']}"
    )
    return create_engine(url, pool_pre_ping=True)


def ensure_tables(engine: Engine, cfg: Config) -> None:
    """
    Membuat tabel output jika belum ada.
    FIX WAJIB: DDL multi-statement harus dieksekusi per-statement (split ';').
    FINAL: tfidf_run_id bertipe TEXT agar kompatibel dengan run_id TF-IDF yang bisa non-UUID.
    """
    ddl = f"""
    CREATE SCHEMA IF NOT EXISTS {cfg.schema};

    CREATE TABLE IF NOT EXISTS {cfg.schema}.{cfg.out_runs}
    (
        job_id uuid,
        modeling_id uuid NOT NULL,
        run_time timestamp with time zone NOT NULL,
        approach text NOT NULL,
        params_json jsonb NOT NULL,
        notes text,
        tfidf_run_id text,
        threshold double precision,
        window_days integer,
        knn_k integer,
        min_cluster_size integer,
        n_rows bigint,
        n_clusters_all bigint,
        n_singletons bigint,
        vocab_size bigint,
        nnz bigint,
        elapsed_sec double precision,
        CONSTRAINT {cfg.out_runs}_pkey PRIMARY KEY (modeling_id)
    );

    CREATE TABLE IF NOT EXISTS {cfg.schema}.{cfg.out_members}
    (
        job_id uuid,
        modeling_id uuid NOT NULL,
        cluster_id bigint NOT NULL,
        incident_number text NOT NULL,
        tgl_submit timestamp without time zone,
        site text,
        assignee text,
        modul text,
        sub_modul text,
        CONSTRAINT {cfg.out_members}_pkey PRIMARY KEY (modeling_id, incident_number)
    );

    CREATE TABLE IF NOT EXISTS {cfg.schema}.{cfg.out_clusters}
    (
        job_id uuid,
        modeling_id uuid NOT NULL,
        cluster_id bigint NOT NULL,
        cluster_size integer NOT NULL,
        min_time timestamp without time zone,
        max_time timestamp without time zone,
        span_days integer,
        CONSTRAINT {cfg.out_clusters}_pkey PRIMARY KEY (modeling_id, cluster_id)
    );

    -- Distribusi cosine similarity pra-threshold (berbasis kNN candidate pairs)
    CREATE TABLE IF NOT EXISTS {cfg.schema}.{cfg.out_cosine_dist}
    (
        job_id uuid NOT NULL,
        tfidf_run_id text NOT NULL,
        run_time timestamp with time zone NOT NULL,
        knn_k integer NOT NULL,
        n_rows bigint NOT NULL,
        n_pairs bigint NOT NULL,
        stats_json jsonb NOT NULL,
        hist_json jsonb NOT NULL,
        sample_note text,
        CONSTRAINT {cfg.out_cosine_dist}_pkey PRIMARY KEY (job_id)
    );

    CREATE INDEX IF NOT EXISTS idx_{cfg.out_runs}_job_time
      ON {cfg.schema}.{cfg.out_runs} (job_id, run_time DESC);

    CREATE INDEX IF NOT EXISTS idx_{cfg.out_members}_job_model_cluster
      ON {cfg.schema}.{cfg.out_members} (job_id, modeling_id, cluster_id);

    CREATE INDEX IF NOT EXISTS idx_{cfg.out_clusters}_job_model_cluster
      ON {cfg.schema}.{cfg.out_clusters} (job_id, modeling_id, cluster_id);

    CREATE INDEX IF NOT EXISTS idx_{cfg.out_cosine_dist}_tfidf
      ON {cfg.schema}.{cfg.out_cosine_dist} (tfidf_run_id, run_time DESC);
    """

    statements = [s.strip() for s in ddl.split(";") if s.strip()]
    with engine.begin() as conn:
        for stmt in statements:
            conn.exec_driver_sql(stmt)


def get_latest_tfidf_run_id(engine: Engine, cfg: Config) -> str:
    sql = f"""
    SELECT run_id
    FROM {cfg.schema}.{cfg.runs_table}
    ORDER BY run_time DESC
    LIMIT 1
    """
    df = pd.read_sql(text(sql), engine)
    if df.empty:
        raise RuntimeError("Tidak ditemukan run TF-IDF di tabel incident_tfidf_runs.")
    return str(df.loc[0, "run_id"])


# ============================================================
# LOAD DATA
# ============================================================

def load_data(engine: Engine, cfg: Config) -> pd.DataFrame:
    lim = f"LIMIT {int(cfg.limit)}" if cfg.limit else ""
    sql = f"""
    SELECT
        incident_number, tgl_submit, site, assignee, modul, sub_modul, tfidf_json
    FROM {cfg.schema}.{cfg.source_table}
    WHERE run_id = :run_id
      AND tfidf_json IS NOT NULL
    ORDER BY tgl_submit NULLS LAST
    {lim}
    """
    df = pd.read_sql(text(sql), engine, params={"run_id": cfg.run_id_tfidf})
    df["tgl_submit"] = pd.to_datetime(df["tgl_submit"], errors="coerce")
    return df


# ============================================================
# BUILD CSR MATRIX (PATCHED ROBUST)
# ============================================================

def _parse_tfidf_obj(doc: Any) -> Dict[str, float]:
    """
    Parse tfidf_json dari DB yang bisa berupa:
    - dict (jsonb)
    - str JSON
    - None / lainnya
    Return: dict term->float
    """
    if doc is None:
        return {}
    if isinstance(doc, dict):
        d = doc
    elif isinstance(doc, str):
        try:
            d = json.loads(doc)
        except Exception:
            return {}
        if not isinstance(d, dict):
            return {}
    else:
        return {}

    out: Dict[str, float] = {}
    for t, v in d.items():
        try:
            fv = float(v)
        except Exception:
            continue
        if fv == 0.0 or (not np.isfinite(fv)):
            continue
        out[str(t)] = fv
    return out


def build_csr(tfidf_list: List[Any]) -> Tuple[csr_matrix, int]:
    """
    Build CSR matrix dari list tfidf_json (robust).
    FIX WAJIB: gunakan float64 untuk menghindari warning buffer dtype mismatch.
    """
    vocab: Dict[str, int] = {}
    indptr: List[int] = [0]
    indices: List[int] = []
    data: List[float] = []

    for doc in tfidf_list:
        d = _parse_tfidf_obj(doc)

        for t, fv in d.items():
            if t not in vocab:
                vocab[t] = len(vocab)
            indices.append(vocab[t])
            data.append(float(fv))

        indptr.append(len(indices))

    X = csr_matrix((data, indices, indptr), dtype=np.float64)
    return X, len(vocab)


# ============================================================
# Cosine similarity distribution (pra-threshold) — berbasis kNN
# ============================================================

def compute_cosine_distribution_knn(
    X_normed: csr_matrix,
    knn_k: int,
    bins: int = 50,
    sample_max: int = 1_000_000,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
    """
    Distribusi cosine similarity pra-threshold dari pasangan kandidat kNN (DIRECTED).
    - X_normed diasumsikan SUDAH L2-normalized
    - Definisi neighbor konsisten dengan clustering:
      k_eff = min(knn_k + 1, n) lalu drop self => efektif knn_k
    """
    n = int(X_normed.shape[0])
    if n == 0:
        stats = {"mean": None, "std": None, "min": None, "max": None, "quantiles": {}}
        hist = {"bin_edges": [], "counts": []}
        return stats, hist, 0

    k = int(max(1, knn_k))
    k_eff = min(k + 1, n)  # +1 utk self

    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=k_eff, n_jobs=-1)
    nn.fit(X_normed)

    dist, _idx = nn.kneighbors(X_normed, return_distance=True)
    sim = 1.0 - dist  # (n, k_eff)

    sim_vals = sim[:, 1:].reshape(-1).astype(np.float64)  # drop self
    sim_vals = sim_vals[np.isfinite(sim_vals)]
    n_pairs_total = int(sim_vals.size)

    if n_pairs_total == 0:
        stats = {"mean": None, "std": None, "min": None, "max": None, "quantiles": {}}
        hist = {"bin_edges": [], "counts": []}
        return stats, hist, 0

    sample_note = "all_pairs"
    sim_used = sim_vals
    if sample_max and n_pairs_total > int(sample_max):
        rng = np.random.default_rng(int(random_state))
        take = int(sample_max)
        pick = rng.choice(n_pairs_total, size=take, replace=False)
        sim_used = sim_vals[pick]
        sample_note = f"subsample_{take}_of_{n_pairs_total}"

    q_points = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    q_vals = np.quantile(sim_used, q_points).tolist()

    stats = {
        "mean": float(np.mean(sim_used)),
        "std": float(np.std(sim_used)),
        "min": float(np.min(sim_used)),
        "max": float(np.max(sim_used)),
        "quantiles": {str(q): float(v) for q, v in zip(q_points, q_vals)},
        "pair_mode": "knn_directed_candidate_pairs",
        "knn_k_requested": int(knn_k),
        "knn_k_effective": int(k_eff - 1),
        "n_pairs_total": int(n_pairs_total),
        "n_pairs_used": int(sim_used.size),
        "sample_note": sample_note,
        "note_method": "Distribution computed over kNN candidate pairs (directed), not full pairwise O(N^2).",
    }

    counts, bin_edges = np.histogram(sim_used, bins=int(bins), range=(0.0, 1.0))
    hist = {
        "bin_edges": [float(x) for x in bin_edges.tolist()],
        "counts": [int(x) for x in counts.tolist()],
        "bins": int(bins),
        "range": [0.0, 1.0],
    }

    return stats, hist, n_pairs_total


def save_cosine_distribution(
    engine: Engine,
    cfg: Config,
    job_id: str,
    tfidf_run_id: str,
    stats_json: Dict[str, Any],
    hist_json: Dict[str, Any],
    n_rows: int,
    n_pairs_total: int,
) -> None:
    run_time = pd.Timestamp.now(tz="UTC").to_pydatetime()
    sample_note = str(stats_json.get("sample_note", ""))

    with engine.begin() as conn:
        conn.execute(
            text(f"""
                INSERT INTO {cfg.schema}.{cfg.out_cosine_dist}
                (job_id, tfidf_run_id, run_time, knn_k, n_rows, n_pairs, stats_json, hist_json, sample_note)
                VALUES
                (CAST(:job_id AS uuid), :tfidf_run_id, :run_time, :knn_k, :n_rows, :n_pairs,
                 :stats_json, :hist_json, :sample_note)
                ON CONFLICT (job_id) DO UPDATE SET
                    run_time = EXCLUDED.run_time,
                    tfidf_run_id = EXCLUDED.tfidf_run_id,
                    knn_k = EXCLUDED.knn_k,
                    n_rows = EXCLUDED.n_rows,
                    n_pairs = EXCLUDED.n_pairs,
                    stats_json = EXCLUDED.stats_json,
                    hist_json = EXCLUDED.hist_json,
                    sample_note = EXCLUDED.sample_note
            """),
            {
                "job_id": str(job_id),
                "tfidf_run_id": str(tfidf_run_id),
                "run_time": run_time,
                "knn_k": int(cfg.knn_k),
                "n_rows": int(n_rows),
                "n_pairs": int(n_pairs_total),
                "stats_json": Json(stats_json),
                "hist_json": Json(hist_json),
                "sample_note": sample_note,
            }
        )


# ============================================================
# UNION-FIND
# ============================================================

class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


# ============================================================
# CLUSTERING (PATCHED CONSISTENT kNN)
# ============================================================

def cluster_text(df: pd.DataFrame, X_normed: csr_matrix, thr: float, knn_k: int) -> pd.DataFrame:
    """
    Clustering via cosine kNN -> edges where similarity >= thr -> connected components.
    Definisi neighbor konsisten:
    k_eff = min(knn_k + 1, n) lalu skip self => efektif knn_k
    """
    df = df.copy()
    n = int(X_normed.shape[0])
    if n == 0:
        df["cluster_id"] = pd.Series(dtype="int64")
        return df

    k = int(max(1, knn_k))
    k_eff = min(k + 1, n)  # +1 utk self
    if k_eff < 2:
        df["cluster_id"] = np.arange(n, dtype=np.int64)
        return df

    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=k_eff, n_jobs=-1)
    nn.fit(X_normed)

    dist, idx = nn.kneighbors(X_normed, return_distance=True)
    sim = 1.0 - dist

    uf = UnionFind(n)
    for i in range(n):
        for jpos in range(1, k_eff):  # skip self
            if float(sim[i, jpos]) >= float(thr):
                uf.union(i, int(idx[i, jpos]))

    roots = np.fromiter((uf.find(i) for i in range(n)), dtype=np.int64, count=n)
    df["cluster_id"] = pd.factorize(roots)[0].astype(np.int64)
    return df


# ============================================================
# SAVE
# ============================================================

def save_all(
    engine: Engine,
    df: pd.DataFrame,
    cfg: Config,
    job_id: str,
    modeling_id: str,
    thr: float,
    vocab_size: int,
    nnz: int,
    elapsed: float,
) -> None:
    run_time = pd.Timestamp.now(tz="UTC").to_pydatetime()

    clusters = (
        df.groupby("cluster_id", as_index=False)
          .agg(
              cluster_size=("incident_number", "count"),
              min_time=("tgl_submit", "min"),
              max_time=("tgl_submit", "max"),
          )
    )

    if not clusters.empty:
        clusters["span_days"] = (clusters["max_time"] - clusters["min_time"]).dt.days
        clusters.insert(0, "job_id", job_id)
        clusters.insert(1, "modeling_id", modeling_id)

        clusters = clusters[[
            "job_id", "modeling_id", "cluster_id",
            "cluster_size", "min_time", "max_time", "span_days"
        ]].copy()

        clusters["cluster_id"] = clusters["cluster_id"].astype("int64")
        clusters["cluster_size"] = clusters["cluster_size"].astype("int32")
        clusters["span_days"] = clusters["span_days"].fillna(0).astype("int32")
    else:
        clusters = pd.DataFrame(columns=[
            "job_id", "modeling_id", "cluster_id",
            "cluster_size", "min_time", "max_time", "span_days"
        ])

    members = df.copy()
    members.insert(0, "job_id", job_id)
    members.insert(1, "modeling_id", modeling_id)

    members = members[[
        "job_id",
        "modeling_id",
        "cluster_id",
        "incident_number",
        "tgl_submit",
        "site",
        "assignee",
        "modul",
        "sub_modul",
    ]].copy()
    members["cluster_id"] = members["cluster_id"].astype("int64")

    n_rows = int(len(df))
    n_clusters_all = int(df["cluster_id"].nunique())

    # FIX WAJIB: jumlah cluster singleton (bukan hitung baris via transform)
    sizes = df.groupby("cluster_id")["incident_number"].size()
    n_singletons = int((sizes == 1).sum())

    run_row = {
        "job_id": str(job_id),
        "modeling_id": str(modeling_id),
        "run_time": run_time,
        "approach": "tfidf_cosine_threshold_grid_job",
        "params_json": asdict(cfg),
        "notes": (
            "Clustering murni: output cluster_id tanpa labeling recurring/noise. "
            "Distribusi cosine pra-threshold dihitung dari kandidat kNN (directed)."
        ),
        "tfidf_run_id": str(cfg.run_id_tfidf),
        "threshold": float(thr),
        "window_days": int(cfg.window_days),
        "knn_k": int(cfg.knn_k),
        "min_cluster_size": int(cfg.min_cluster_size),
        "n_rows": int(n_rows),
        "n_clusters_all": int(n_clusters_all),
        "n_singletons": int(n_singletons),
        "vocab_size": int(vocab_size),
        "nnz": int(nnz),
        "elapsed_sec": float(elapsed),
    }

    with engine.begin() as conn:
        conn.execute(
            text(f"""
                INSERT INTO {cfg.schema}.{cfg.out_runs}
                (
                    job_id, modeling_id, run_time, approach, params_json, notes,
                    tfidf_run_id, threshold, window_days, knn_k, min_cluster_size,
                    n_rows, n_clusters_all, n_singletons, vocab_size, nnz, elapsed_sec
                )
                VALUES
                (
                    CAST(:job_id AS uuid),
                    CAST(:modeling_id AS uuid),
                    :run_time,
                    :approach,
                    :params_json,
                    :notes,
                    :tfidf_run_id,
                    :threshold,
                    :window_days,
                    :knn_k,
                    :min_cluster_size,
                    :n_rows,
                    :n_clusters_all,
                    :n_singletons,
                    :vocab_size,
                    :nnz,
                    :elapsed_sec
                )
            """),
            {
                "job_id": run_row["job_id"],
                "modeling_id": run_row["modeling_id"],
                "run_time": run_row["run_time"],
                "approach": run_row["approach"],
                "params_json": Json(run_row["params_json"]),
                "notes": run_row["notes"],
                "tfidf_run_id": run_row["tfidf_run_id"],
                "threshold": run_row["threshold"],
                "window_days": run_row["window_days"],
                "knn_k": run_row["knn_k"],
                "min_cluster_size": run_row["min_cluster_size"],
                "n_rows": run_row["n_rows"],
                "n_clusters_all": run_row["n_clusters_all"],
                "n_singletons": run_row["n_singletons"],
                "vocab_size": run_row["vocab_size"],
                "nnz": run_row["nnz"],
                "elapsed_sec": run_row["elapsed_sec"],
            }
        )

        members.to_sql(
            name=cfg.out_members,
            con=conn,
            schema=cfg.schema,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=5000,
        )

        if not clusters.empty:
            clusters.to_sql(
                name=cfg.out_clusters,
                con=conn,
                schema=cfg.schema,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=5000,
            )


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", default="0.60,0.70,0.80,0.90")
    ap.add_argument("--window-days", type=int, default=30)
    ap.add_argument("--knn-k", type=int, default=25)
    ap.add_argument("--min-cluster-size", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)

    # cosine distribution controls
    ap.add_argument("--dist-bins", type=int, default=50)
    ap.add_argument("--dist-sample-max", type=int, default=1_000_000)

    args = ap.parse_args()

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    if not thresholds:
        thresholds = [0.6, 0.7, 0.8, 0.9]

    cfg = Config(
        knn_k=int(args.knn_k),
        window_days=int(args.window_days),
        min_cluster_size=int(args.min_cluster_size),
        limit=args.limit,
        dist_bins=int(args.dist_bins),
        dist_sample_max=int(args.dist_sample_max),
    )

    engine = get_engine()
    ensure_tables(engine, cfg)

    # latest tfidf run
    cfg.run_id_tfidf = get_latest_tfidf_run_id(engine, cfg)

    df = load_data(engine, cfg)
    if df.empty:
        raise RuntimeError("Data kosong. Cek run_id TF-IDF / isi tabel incident_tfidf_vectors.")

    # CSR
    X, vocab_size = build_csr(df["tfidf_json"].tolist())
    if X.shape[0] != len(df):
        raise RuntimeError("CSR rows != df rows. Ada masalah saat build_csr.")

    X.sum_duplicates()

    # FIX WAJIB: pastikan float64 untuk menghindari warning dtype mismatch
    if X.dtype != np.float64:
        X = X.astype(np.float64)

    normalize(X, norm="l2", axis=1, copy=False)

    job_id = str(uuid.uuid4())
    print(f"[JOB START] job_id={job_id} | tfidf_run_id={cfg.run_id_tfidf} | rows={len(df)}")

    # cosine dist pra-threshold (sekali per job)
    tdist0 = time.time()
    stats_json, hist_json, n_pairs_total = compute_cosine_distribution_knn(
        X_normed=X,
        knn_k=int(cfg.knn_k),
        bins=int(cfg.dist_bins),
        sample_max=int(cfg.dist_sample_max),
        random_state=42,
    )
    save_cosine_distribution(
        engine=engine,
        cfg=cfg,
        job_id=job_id,
        tfidf_run_id=str(cfg.run_id_tfidf),
        stats_json=stats_json,
        hist_json=hist_json,
        n_rows=int(len(df)),
        n_pairs_total=int(n_pairs_total),
    )
    print(
        f"[COSINE DIST SAVED] job_id={job_id} | pairs_total={stats_json.get('n_pairs_total')} "
        f"| used={stats_json.get('n_pairs_used')} | note={stats_json.get('sample_note')} "
        f"| elapsed={time.time()-tdist0:.2f}s"
    )

    # threshold grid modeling
    for thr in thresholds:
        t0 = time.time()
        modeling_id = str(uuid.uuid4())

        dfc = cluster_text(df, X_normed=X, thr=float(thr), knn_k=int(cfg.knn_k))

        save_all(
            engine=engine,
            df=dfc,
            cfg=cfg,
            job_id=job_id,
            modeling_id=modeling_id,
            thr=float(thr),
            vocab_size=int(vocab_size),
            nnz=int(X.nnz),
            elapsed=float(time.time() - t0),
        )

        print(f"[DONE] job_id={job_id} | threshold={thr} | modeling_id={modeling_id}")

    print(f"[JOB DONE] job_id={job_id} | thresholds={thresholds}")


if __name__ == "__main__":
    main()
