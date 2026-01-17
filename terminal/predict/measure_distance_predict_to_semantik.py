from __future__ import annotations

import os
import sys
import json
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# ============================================================
# DB SETTINGS (JANGAN DIUBAH)
# ============================================================
def make_engine() -> Engine:
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")
    db   = os.getenv("PGDATABASE", "incident_djp")
    user = os.getenv("PGUSER", "postgres")
    pw   = os.getenv("PGPASSWORD", "admin*123")
    url = f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}"
    return create_engine(url, pool_pre_ping=True)


# ============================================================
# KONFIG
# ============================================================
SCHEMA = "lasis_djp"

# Predict embeddings (SBERT) - FLOAT8[] embedding
PRED_TABLE = "incident_predict_sbert_embeddings"
PRED_INC_COL = "incident_number"
PRED_MODEL_COL = "model_name"
PRED_EMB_COL = "embedding"

# Semantik embeddings (JSON) - embedding_json
SEM_VEC_TABLE = "semantik_embedding_vectors"
SEM_INC_COL = "incident_number"
SEM_EMBJSON_COL = "embedding_json"
SEM_DIM_COL = "embedding_dim"

# Cluster membership (incident_number -> cluster_id)
# Kalau tabel kamu beda, ubah via ENV:
# setx MEM_TABLE "nama_table"
MEM_TABLE = os.getenv("MEM_TABLE", "modeling_semantik_hdbscan_members")
MEM_INC_COL = os.getenv("MEM_INC_COL", "incident_number")
MEM_CLUSTER_COL = os.getenv("MEM_CLUSTER_COL", "cluster_id")
MEM_IS_NOISE_COL = os.getenv("MEM_IS_NOISE_COL", "is_noise")  # opsional

# Output
OUT_TABLE = "predict_to_semantik_distance"

TOP_K = int(os.getenv("TOP_K", "5"))
BATCH = int(os.getenv("BATCH", "256"))

# exclude noise: cluster_id = -1 / is_noise true
EXCLUDE_NOISE = os.getenv("EXCLUDE_NOISE", "1") == "1"


# ============================================================
# OUTPUT TABLE
# ============================================================
def ensure_out_table(engine: Engine) -> None:
    ddl = f"""
    CREATE SCHEMA IF NOT EXISTS {SCHEMA};

    CREATE TABLE IF NOT EXISTS {SCHEMA}.{OUT_TABLE} (
        run_id BIGSERIAL PRIMARY KEY,
        incident_number TEXT NOT NULL,
        model_name TEXT NOT NULL,
        topk_rank INTEGER NOT NULL,
        target_cluster_id BIGINT,
        target_incident_number TEXT,
        cosine_sim DOUBLE PRECISION NOT NULL,
        cosine_dist DOUBLE PRECISION NOT NULL,
        created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT now()
    );

    CREATE INDEX IF NOT EXISTS idx_{OUT_TABLE}_inc
    ON {SCHEMA}.{OUT_TABLE} (incident_number);

    CREATE INDEX IF NOT EXISTS idx_{OUT_TABLE}_cluster
    ON {SCHEMA}.{OUT_TABLE} (target_cluster_id);

    CREATE INDEX IF NOT EXISTS idx_{OUT_TABLE}_model
    ON {SCHEMA}.{OUT_TABLE} (model_name);
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


# ============================================================
# HELPERS
# ============================================================
def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)


def parse_embedding_json(val) -> Optional[List[float]]:
    """
    embedding_json bisa berupa:
    - python list (kalau driver decode)
    - string JSON '[]'
    """
    if val is None:
        return None

    if isinstance(val, list):
        return val

    if isinstance(val, (bytes, bytearray)):
        val = val.decode("utf-8", errors="ignore")

    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return obj
        except Exception:
            return None

    return None


def safe_fetch_columns(engine: Engine, schema: str, table: str) -> List[str]:
    q = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = :s AND table_name = :t
    ORDER BY ordinal_position
    """
    with engine.connect() as conn:
        rows = conn.execute(text(q), {"s": schema, "t": table}).fetchall()
    return [r[0] for r in rows]


def to_matrix_float8array(series: pd.Series) -> np.ndarray:
    # untuk kolom FLOAT8[] dari Postgres
    arr = np.array(series.tolist(), dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Predict embedding invalid. ndim={arr.ndim}")
    return arr


def filter_semantik_by_dim(df_sem: pd.DataFrame, expected_dim: int) -> pd.DataFrame:
    # keep hanya embedding yang list dan panjangnya = expected_dim
    mask = df_sem["embedding"].apply(lambda x: isinstance(x, list) and len(x) == expected_dim)
    bad = int((~mask).sum())
    if bad > 0:
        # tampilkan beberapa contoh dimensi yang salah
        wrong_dims = (
            df_sem.loc[~mask, "embedding"]
            .apply(lambda x: len(x) if isinstance(x, list) else None)
            .value_counts(dropna=False)
            .head(10)
            .to_dict()
        )
        print(f"⚠️  Drop {bad:,} baris semantik karena dimensi embedding != {expected_dim}. Contoh distribusi dimensi salah: {wrong_dims}")
    return df_sem.loc[mask].reset_index(drop=True)


def to_matrix_from_list(series: pd.Series, expected_dim: int) -> np.ndarray:
    # series berisi list float dengan dimensi seragam
    lst = series.tolist()
    arr = np.array(lst, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Semantik embedding invalid. ndim={arr.ndim}")
    if arr.shape[1] != expected_dim:
        raise ValueError(f"Semantik dim mismatch. got={arr.shape[1]} expected={expected_dim}")
    return arr


# ============================================================
# LOAD DATA
# ============================================================
def load_predict(engine: Engine) -> pd.DataFrame:
    cols = safe_fetch_columns(engine, SCHEMA, PRED_TABLE)
    need = {PRED_INC_COL, PRED_MODEL_COL, PRED_EMB_COL}
    missing = need - set(cols)
    if missing:
        raise KeyError(f"Kolom PRED hilang: {sorted(missing)}. Kolom tersedia: {cols}")

    q = f"""
    SELECT
        {PRED_INC_COL} AS incident_number,
        {PRED_MODEL_COL} AS model_name,
        {PRED_EMB_COL} AS embedding
    FROM {SCHEMA}.{PRED_TABLE}
    WHERE {PRED_EMB_COL} IS NOT NULL
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(q), conn)

    df.columns = [c.strip().lower() for c in df.columns]
    df = df.drop_duplicates(subset=["incident_number", "model_name"], keep="first").reset_index(drop=True)
    return df


def load_semantik_join_members(engine: Engine) -> pd.DataFrame:
    # cek kolom membership (buat noise filter yang aman)
    mem_cols = set(safe_fetch_columns(engine, SCHEMA, MEM_TABLE))
    if MEM_CLUSTER_COL not in mem_cols or MEM_INC_COL not in mem_cols:
        raise KeyError(
            f"Tabel membership {SCHEMA}.{MEM_TABLE} harus punya kolom "
            f"{MEM_INC_COL} dan {MEM_CLUSTER_COL}. Kolom tersedia: {sorted(mem_cols)}"
        )

    where_noise = ""
    if EXCLUDE_NOISE:
        if MEM_IS_NOISE_COL in mem_cols:
            where_noise = f"AND COALESCE(m.{MEM_IS_NOISE_COL}, FALSE) = FALSE AND m.{MEM_CLUSTER_COL} <> -1"
        else:
            where_noise = f"AND m.{MEM_CLUSTER_COL} <> -1"

    # cek kolom vector table
    sem_cols = set(safe_fetch_columns(engine, SCHEMA, SEM_VEC_TABLE))
    need = {SEM_INC_COL, SEM_EMBJSON_COL}
    missing = need - sem_cols
    if missing:
        raise KeyError(f"Kolom SEM_VEC hilang: {sorted(missing)}. Kolom tersedia: {sorted(sem_cols)}")

    q = f"""
    SELECT
        v.{SEM_INC_COL} AS incident_number,
        v.{SEM_DIM_COL} AS embedding_dim,
        v.{SEM_EMBJSON_COL} AS embedding_json,
        m.{MEM_CLUSTER_COL} AS cluster_id
    FROM {SCHEMA}.{SEM_VEC_TABLE} v
    JOIN {SCHEMA}.{MEM_TABLE} m
      ON m.{MEM_INC_COL} = v.{SEM_INC_COL}
    WHERE v.{SEM_EMBJSON_COL} IS NOT NULL
      AND m.{MEM_CLUSTER_COL} IS NOT NULL
      {where_noise}
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(q), conn)

    df.columns = [c.strip().lower() for c in df.columns]

    # parse json -> list
    df["embedding"] = df["embedding_json"].apply(parse_embedding_json)
    df = df[df["embedding"].notna()].reset_index(drop=True)

    # keep kolom penting saja
    return df[["incident_number", "cluster_id", "embedding"]]


# ============================================================
# CLUSTER CENTROIDS + TOPK
# ============================================================
def build_cluster_centroids(df_sem: pd.DataFrame, expected_dim: int) -> Tuple[pd.DataFrame, np.ndarray]:
    # filter dimensi dulu agar tidak error shape
    df_sem = filter_semantik_by_dim(df_sem, expected_dim=expected_dim)
    if df_sem.empty:
        raise ValueError("Semua embedding semantik terfilter habis (tidak ada yang berdimensi sama dengan predict).")

    X = to_matrix_from_list(df_sem["embedding"], expected_dim=expected_dim)

    cluster_ids = df_sem["cluster_id"].astype(np.int64).to_numpy()
    uniq = np.unique(cluster_ids)

    centroids = np.zeros((len(uniq), expected_dim), dtype=np.float64)
    repr_inc = []

    for i, cid in enumerate(uniq):
        idx = np.where(cluster_ids == cid)[0]
        centroids[i] = X[idx].mean(axis=0)
        repr_inc.append(df_sem.iloc[idx[0]]["incident_number"])

    dfc = pd.DataFrame({"cluster_id": uniq, "repr_incident_number": repr_inc})
    return dfc, centroids


def topk_cosine(pred_mat: np.ndarray, cent_mat: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    P = normalize_rows(pred_mat)
    C = normalize_rows(cent_mat)
    sims = P @ C.T

    k = min(k, sims.shape[1])
    idx_part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    row_idx = np.arange(sims.shape[0])[:, None]
    cand_sims = sims[row_idx, idx_part]
    order = np.argsort(-cand_sims, axis=1)
    top_idx = idx_part[row_idx, order]
    top_sims = cand_sims[row_idx, order]
    return top_idx, top_sims


def insert_results(engine: Engine, rows: List[Dict]) -> None:
    sql = f"""
    INSERT INTO {SCHEMA}.{OUT_TABLE}
    (incident_number, model_name, topk_rank, target_cluster_id, target_incident_number, cosine_sim, cosine_dist)
    VALUES
    (:incident_number, :model_name, :topk_rank, :target_cluster_id, :target_incident_number, :cosine_sim, :cosine_dist)
    """
    with engine.begin() as conn:
        conn.execute(text(sql), rows)


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    engine = make_engine()
    ensure_out_table(engine)

    print(f"✅ Load predict embeddings: {SCHEMA}.{PRED_TABLE}")
    df_pred = load_predict(engine)
    print(f"   predict rows: {len(df_pred):,}")

    if df_pred.empty:
        print("⚠️ predict kosong. Stop.")
        return

    # matrix predict
    pred_mat_all = to_matrix_float8array(df_pred["embedding"])
    pred_dim = pred_mat_all.shape[1]
    print(f"✅ pred_dim = {pred_dim}")

    print(f"✅ Load semantik vectors + membership join: {SCHEMA}.{SEM_VEC_TABLE} + {SCHEMA}.{MEM_TABLE}")
    df_sem = load_semantik_join_members(engine)
    print(f"   semantik rows (joined, before dim-filter): {len(df_sem):,}")

    if df_sem.empty:
        print("⚠️ semantik kosong (hasil join 0). Stop.")
        return

    print("✅ Build semantik cluster centroids (dimension-aligned)...")
    df_cent, cent_mat = build_cluster_centroids(df_sem, expected_dim=pred_dim)
    print(f"   clusters: {len(df_cent):,}, dim: {cent_mat.shape[1]}")

    # cek final dim
    if cent_mat.shape[1] != pred_dim:
        raise ValueError(f"Dim mismatch: pred={pred_dim} sem_centroid={cent_mat.shape[1]}")

    n = len(df_pred)
    print(f"✅ Compute top-{TOP_K} cosine similarity per predict ticket...")

    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        chunk = df_pred.iloc[start:end].reset_index(drop=True)
        pred_mat = pred_mat_all[start:end]

        top_idx, top_sims = topk_cosine(pred_mat, cent_mat, TOP_K)

        rows_out: List[Dict] = []
        for i in range(len(chunk)):
            inc = chunk.loc[i, "incident_number"]
            model_name = chunk.loc[i, "model_name"]

            for r in range(min(TOP_K, top_idx.shape[1])):
                j = int(top_idx[i, r])
                sim = float(top_sims[i, r])
                dist = 1.0 - sim

                rows_out.append(
                    {
                        "incident_number": inc,
                        "model_name": model_name,
                        "topk_rank": r + 1,
                        "target_cluster_id": int(df_cent.loc[j, "cluster_id"]),
                        "target_incident_number": df_cent.loc[j, "repr_incident_number"],
                        "cosine_sim": sim,
                        "cosine_dist": dist,
                    }
                )

        insert_results(engine, rows_out)
        print(f"✅ inserted results for {end:,}/{n:,}")

    print(f"🎉 Selesai. Hasil tersimpan di: {SCHEMA}.{OUT_TABLE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("❌ ERROR:", repr(e), file=sys.stderr)
        raise
