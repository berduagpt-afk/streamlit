# ============================================================
# Offline Modeling Sintaksis — FINAL (THRESHOLD GRID + JOB ID)
# TF-IDF (sudah ada di DB) + Cosine kNN + Threshold EXPERIMENT
#
# Output tables:
# - lasis_djp.modeling_sintaksis_runs      (job_id + modeling_id)
# - lasis_djp.modeling_sintaksis_clusters  (NO is_recurring)
# - lasis_djp.modeling_sintaksis_members   (NO is_recurring)
#
# Tambahan (BOOTSTRAP TF-IDF TABLES + INDEX):
# - Auto create/ensure schema + tabel:
#   - lasis_djp.incident_tfidf_runs
#   - lasis_djp.incident_tfidf_vectors
# - Auto-add kolom yang kurang (ALTER TABLE) sehingga tidak perlu manual.
# - Tambahkan index untuk percepat:
#   - latest run (run_time desc)
#   - filter vectors by run_id
#   - order by tgl_submit
#   - filter by modul/sub_modul (opsional tapi sering kepakai)
#
# Fix:
# - Tidak menyimpan dict ke kolom non-json (hindari "can't adapt type 'dict'")
# - params_json dibungkus psycopg2.extras.Json
# - to_sql args keyword-only (aman pandas >= 2.x dan future pandas 3.0)
# ============================================================

from __future__ import annotations

import time
import uuid
import argparse
from dataclasses import dataclass, asdict
from typing import List, Optional, Any, Tuple, Dict

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from psycopg2.extras import Json


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

    run_id_tfidf: str = ""
    knn_k: int = 25
    window_days: int = 30          # metadata saja di tahap clustering ini
    min_cluster_size: int = 10     # metadata saja di tahap clustering ini

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


# ============================================================
# BOOTSTRAP TF-IDF TABLES (runs & vectors) + auto ALTER columns
# + INDEX tambahan
# ============================================================

def _existing_columns(engine: Engine, schema: str, table: str) -> set[str]:
    sql = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = :schema AND table_name = :table
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql), {"schema": schema, "table": table}).fetchall()
    return {r[0] for r in rows}


def _ensure_columns(engine: Engine, schema: str, table: str, desired_cols: Dict[str, str]) -> None:
    existing = _existing_columns(engine, schema, table)
    if not existing:
        return

    alters = []
    for col, coldef in desired_cols.items():
        if col not in existing:
            alters.append(f"ALTER TABLE {schema}.{table} ADD COLUMN {col} {coldef};")

    if alters:
        with engine.begin() as conn:
            for stmt in alters:
                conn.execute(text(stmt))


def bootstrap_tfidf_tables(engine: Engine, cfg: Config) -> None:
    """
    Membuat / melengkapi tabel:
      - lasis_djp.incident_tfidf_runs
      - lasis_djp.incident_tfidf_vectors
    sesuai DDL final user, + auto add missing columns, + indexes untuk performa.
    """
    schema = cfg.schema
    runs = cfg.runs_table
    vecs = cfg.source_table

    ddl_runs = f"""
    CREATE SCHEMA IF NOT EXISTS {schema};

    CREATE TABLE IF NOT EXISTS {schema}.{runs}
    (
        run_id text NOT NULL,
        run_time timestamp without time zone NOT NULL,
        approach text NOT NULL,
        params_json jsonb NOT NULL,
        data_range jsonb,
        notes text,
        idf_json jsonb,
        feature_names_json jsonb,
        CONSTRAINT incident_tfidf_runs_pkey PRIMARY KEY (run_id)
    );
    """

    ddl_vecs = f"""
    CREATE SCHEMA IF NOT EXISTS {schema};

    CREATE TABLE IF NOT EXISTS {schema}.{vecs}
    (
        run_id text NOT NULL,
        incident_number text NOT NULL,
        tgl_submit timestamp without time zone,
        site text,
        assignee text,
        modul text,
        sub_modul text,
        subject_tokens_json jsonb,
        tf_json jsonb,
        idf_json jsonb,
        tfidf_json jsonb NOT NULL,
        tfidf_vec_json jsonb,
        tokens_sintaksis_json jsonb,
        text_sintaksis text,
        CONSTRAINT incident_tfidf_vectors_pkey PRIMARY KEY (run_id, incident_number)
    );
    """

    # Indexes: dibuat IF NOT EXISTS (aman dipanggil berulang)
    ddl_indexes = f"""
    -- incident_tfidf_runs: cepat ambil latest run
    CREATE INDEX IF NOT EXISTS idx_{runs}_run_time_desc
      ON {schema}.{runs} (run_time DESC);

    -- incident_tfidf_runs: sering juga filter by approach + waktu (opsional, tapi murah)
    CREATE INDEX IF NOT EXISTS idx_{runs}_approach_run_time_desc
      ON {schema}.{runs} (approach, run_time DESC);

    -- incident_tfidf_vectors: cepat filter WHERE run_id=...
    CREATE INDEX IF NOT EXISTS idx_{vecs}_run_id
      ON {schema}.{vecs} (run_id);

    -- incident_tfidf_vectors: cepat untuk ORDER BY tgl_submit (setelah filter run_id)
    CREATE INDEX IF NOT EXISTS idx_{vecs}_run_id_tgl_submit
      ON {schema}.{vecs} (run_id, tgl_submit);

    -- incident_tfidf_vectors: sering dipakai untuk analisis per modul/sub_modul (opsional tapi berguna)
    CREATE INDEX IF NOT EXISTS idx_{vecs}_run_id_modul
      ON {schema}.{vecs} (run_id, modul);

    CREATE INDEX IF NOT EXISTS idx_{vecs}_run_id_modul_submodul
      ON {schema}.{vecs} (run_id, modul, sub_modul);
    """

    with engine.begin() as conn:
        conn.execute(text(ddl_runs))
        conn.execute(text(ddl_vecs))
        conn.execute(text(ddl_indexes))

    desired_runs = {
        "run_id": "text NOT NULL",
        "run_time": "timestamp without time zone NOT NULL",
        "approach": "text NOT NULL",
        "params_json": "jsonb NOT NULL",
        "data_range": "jsonb",
        "notes": "text",
        "idf_json": "jsonb",
        "feature_names_json": "jsonb",
    }
    desired_vecs = {
        "run_id": "text NOT NULL",
        "incident_number": "text NOT NULL",
        "tgl_submit": "timestamp without time zone",
        "site": "text",
        "assignee": "text",
        "modul": "text",
        "sub_modul": "text",
        "subject_tokens_json": "jsonb",
        "tf_json": "jsonb",
        "idf_json": "jsonb",
        "tfidf_json": "jsonb NOT NULL",
        "tfidf_vec_json": "jsonb",
        "tokens_sintaksis_json": "jsonb",
        "text_sintaksis": "text",
    }
    _ensure_columns(engine, schema, runs, desired_runs)
    _ensure_columns(engine, schema, vecs, desired_vecs)


# ============================================================
# OUTPUT TABLES (modeling_*)
# ============================================================

def ensure_tables(engine: Engine, cfg: Config) -> None:
    """
    Membuat tabel output jika belum ada (schema final TANPA is_recurring di members/clusters).
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

    CREATE INDEX IF NOT EXISTS idx_{cfg.out_runs}_job_time
      ON {cfg.schema}.{cfg.out_runs} (job_id, run_time DESC);

    CREATE INDEX IF NOT EXISTS idx_{cfg.out_members}_job_model_cluster
      ON {cfg.schema}.{cfg.out_members} (job_id, modeling_id, cluster_id);

    CREATE INDEX IF NOT EXISTS idx_{cfg.out_clusters}_job_model_cluster
      ON {cfg.schema}.{cfg.out_clusters} (job_id, modeling_id, cluster_id);

    -- tambahan: mempercepat join/filter per modeling_id
    CREATE INDEX IF NOT EXISTS idx_{cfg.out_members}_modeling_id
      ON {cfg.schema}.{cfg.out_members} (modeling_id);

    CREATE INDEX IF NOT EXISTS idx_{cfg.out_clusters}_modeling_id
      ON {cfg.schema}.{cfg.out_clusters} (modeling_id);
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def get_latest_tfidf_run_id(engine: Engine, cfg: Config) -> str:
    chk = "SELECT to_regclass(:tbl) AS t"
    tbl = f"{cfg.schema}.{cfg.runs_table}"
    with engine.connect() as conn:
        t = conn.execute(text(chk), {"tbl": tbl}).scalar()
    if t is None:
        raise RuntimeError(
            f"Tabel {tbl} tidak ada. Bootstrap gagal atau schema/table name berbeda."
        )

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
# BUILD CSR MATRIX
# ============================================================

def build_csr(tfidf_list: List[Any]) -> Tuple[csr_matrix, int]:
    """
    Build CSR matrix dari list dict term->value.
    """
    vocab: dict[str, int] = {}
    indptr = [0]
    indices: list[int] = []
    data: list[float] = []

    for doc in tfidf_list:
        d = doc if isinstance(doc, dict) else {}
        for t, v in d.items():
            if t not in vocab:
                vocab[t] = len(vocab)
            try:
                fv = float(v)
            except Exception:
                continue
            if fv == 0.0:
                continue
            indices.append(vocab[t])
            data.append(fv)
        indptr.append(len(indices))

    X = csr_matrix((data, indices, indptr), dtype=np.float32)
    return X, len(vocab)


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
# CLUSTERING (murni)
# ============================================================

def cluster_text(df: pd.DataFrame, X: csr_matrix, thr: float, k: int) -> pd.DataFrame:
    """
    Clustering via kNN cosine -> edges where similarity >= thr -> connected components (Union-Find).
    Output: df["cluster_id"] untuk SEMUA ticket (tanpa noise/recurring).
    """
    df = df.copy()
    if X.shape[0] == 0:
        df["cluster_id"] = pd.Series(dtype="int64")
        return df

    normalize(X, norm="l2", axis=1, copy=False)

    k = int(max(2, k))
    k = int(min(k, X.shape[0]))

    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=k, n_jobs=-1)
    nn.fit(X)

    dist, idx = nn.kneighbors(X, return_distance=True)
    sim = 1.0 - dist

    uf = UnionFind(len(df))
    for i in range(len(df)):
        for jpos in range(1, k):
            if float(sim[i, jpos]) >= float(thr):
                uf.union(i, int(idx[i, jpos]))

    roots = np.fromiter((uf.find(i) for i in range(len(df))), dtype=np.int64, count=len(df))
    df["cluster_id"] = pd.factorize(roots)[0].astype(np.int64)
    return df


# ============================================================
# SAVE (job_id + NO is_recurring)
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
    n_singletons = int((df.groupby("cluster_id")["incident_number"].transform("count") == 1).sum())

    run_row = {
        "job_id": str(job_id),
        "modeling_id": str(modeling_id),
        "run_time": run_time,
        "approach": "tfidf_cosine_threshold_grid_job",
        "params_json": asdict(cfg),
        "notes": "Clustering murni: output cluster_id tanpa labeling recurring/noise. window_days & min_cluster_size disimpan sebagai metadata.",
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
                    job_id,
                    modeling_id,
                    run_time,
                    approach,
                    params_json,
                    notes,
                    tfidf_run_id,
                    threshold,
                    window_days,
                    knn_k,
                    min_cluster_size,
                    n_rows,
                    n_clusters_all,
                    n_singletons,
                    vocab_size,
                    nnz,
                    elapsed_sec
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
    ap.add_argument("--window-days", type=int, default=30)       # metadata
    ap.add_argument("--knn-k", type=int, default=25)
    ap.add_argument("--min-cluster-size", type=int, default=10)  # metadata
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]

    cfg = Config(
        knn_k=int(args.knn_k),
        window_days=int(args.window_days),
        min_cluster_size=int(args.min_cluster_size),
        limit=args.limit,
    )

    engine = get_engine()

    # ✅ 1) Bootstrap TF-IDF tables + indexes
    bootstrap_tfidf_tables(engine, cfg)

    # ✅ 2) Ensure output modeling tables (+ indexes)
    ensure_tables(engine, cfg)

    # ✅ 3) Ambil latest TF-IDF run_id
    cfg.run_id_tfidf = get_latest_tfidf_run_id(engine, cfg)

    df = load_data(engine, cfg)
    if df.empty:
        raise RuntimeError("Data kosong. Cek run_id TF-IDF / isi tabel incident_tfidf_vectors.")

    # Build CSR sekali
    X, vocab_size = build_csr(df["tfidf_json"].tolist())
    if X.shape[0] != len(df):
        raise RuntimeError("CSR rows != df rows. Ada masalah saat build_csr.")

    # JOB ID (sekali per eksekusi script)
    job_id = str(uuid.uuid4())
    print(f"[JOB START] job_id={job_id} | tfidf_run_id={cfg.run_id_tfidf} | rows={len(df)}")

    for thr in thresholds:
        t0 = time.time()
        modeling_id = str(uuid.uuid4())

        dfc = cluster_text(df, X, thr=float(thr), k=int(cfg.knn_k))

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
