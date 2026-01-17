from __future__ import annotations

import os
import sys
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from sentence_transformers import SentenceTransformer


# =========================
# KONFIGURASI
# =========================
SCHEMA_SRC = os.getenv("SCHEMA_SRC", "lasis_djp")
TABLE_SRC = os.getenv("TABLE_SRC", "incident_semantic_labels")

SCHEMA_OUT = os.getenv("SCHEMA_OUT", "lasis_djp")
TABLE_OUT = os.getenv("TABLE_OUT", "incident_predict_sbert_embeddings")

MODEL_NAME = os.getenv("SBERT_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "256"))
LIMIT_ROWS: Optional[int] = None  # isi angka kalau mau test kecil dulu


# =========================
# DB ENGINE
# =========================
def make_engine() -> Engine:
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")
    db = os.getenv("PGDATABASE", "incident_djp")
    user = os.getenv("PGUSER", "postgres")
    pw = os.getenv("PGPASSWORD", "admin*123")

    url = f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}"
    return create_engine(url, pool_pre_ping=True)


def ensure_schema(engine: Engine, schema: str) -> None:
    if not schema.replace("_", "").isalnum():
        raise ValueError(f"Invalid schema name: {schema}")
    with engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}";'))


def ensure_table_out(engine: Engine) -> None:
    ensure_schema(engine, SCHEMA_OUT)

    ddl = f"""
    CREATE TABLE IF NOT EXISTS "{SCHEMA_OUT}"."{TABLE_OUT}" (
        embed_id BIGSERIAL PRIMARY KEY,
        incident_number TEXT NOT NULL,
        label INTEGER NOT NULL,
        model_name TEXT NOT NULL,
        embedding_dim INTEGER NOT NULL,
        embedding FLOAT8[] NOT NULL,
        run_time TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT now(),
        CONSTRAINT uq_incident_predict_sbert UNIQUE (incident_number, model_name)
    );

    CREATE INDEX IF NOT EXISTS idx_predict_sbert_label
    ON "{SCHEMA_OUT}"."{TABLE_OUT}" (label);

    CREATE INDEX IF NOT EXISTS idx_predict_sbert_model
    ON "{SCHEMA_OUT}"."{TABLE_OUT}" (model_name);
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


# =========================
# LOAD SOURCE (sesuai DDL)
# =========================
def load_source(engine: Engine) -> pd.DataFrame:
    q = f"""
    SELECT
        incident_number,
        text_semantic,
        label
    FROM "{SCHEMA_SRC}"."{TABLE_SRC}"
    WHERE text_semantic IS NOT NULL
      AND trim(text_semantic) <> ''
    """
    if LIMIT_ROWS is not None:
        q += f" LIMIT {int(LIMIT_ROWS)}"

    with engine.connect() as conn:
        df = pd.read_sql(text(q), conn)

    # Normalisasi nama kolom
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"incident_number", "text_semantic", "label"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Kolom wajib tidak ditemukan: {sorted(missing)}. "
            f"Kolom tersedia: {list(df.columns)}. "
            f"Pastikan DDL tabel sumber sesuai."
        )

    # Dedup berdasarkan incident_number
    df = df.drop_duplicates(subset=["incident_number"], keep="first").reset_index(drop=True)
    return df


def already_embedded(engine: Engine) -> set[str]:
    q = f"""
    SELECT incident_number
    FROM "{SCHEMA_OUT}"."{TABLE_OUT}"
    WHERE model_name = :m
    """
    with engine.connect() as conn:
        rows = conn.execute(text(q), {"m": MODEL_NAME}).fetchall()
    return {r[0] for r in rows}


def insert_batch(engine: Engine, rows: List[Dict]) -> None:
    # ON CONFLICT agar aman jika ada duplikasi incident_number+model_name
    sql = f"""
    INSERT INTO "{SCHEMA_OUT}"."{TABLE_OUT}"
    (incident_number, label, model_name, embedding_dim, embedding)
    VALUES
    (:incident_number, :label, :model_name, :embedding_dim, :embedding)
    ON CONFLICT (incident_number, model_name)
    DO UPDATE SET
        label = EXCLUDED.label,
        embedding_dim = EXCLUDED.embedding_dim,
        embedding = EXCLUDED.embedding,
        run_time = now()
    """
    with engine.begin() as conn:
        conn.execute(text(sql), rows)


# =========================
# MAIN
# =========================
def main() -> None:
    engine = make_engine()
    ensure_table_out(engine)

    print(f"✅ Load source: {SCHEMA_SRC}.{TABLE_SRC}")
    df = load_source(engine)
    print(f"   rows loaded (dedup): {len(df):,}")

    done = already_embedded(engine)
    if done:
        before = len(df)
        df = df[~df["incident_number"].isin(done)].reset_index(drop=True)
        print(f"✅ Skip already-embedded (model sama): {before - len(df):,}")

    if df.empty:
        print("✅ Tidak ada data baru untuk embedding.")
        return

    print(f"✅ Load SBERT model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    n = len(df)
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        chunk = df.iloc[start:end]

        texts = chunk["text_semantic"].astype(str).tolist()
        emb = model.encode(
            texts,
            batch_size=min(BATCH_SIZE, 64),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        dim = int(emb.shape[1])
        emb_list = emb.astype(np.float64).tolist()

        rows: List[Dict] = []
        for i, r in enumerate(chunk.itertuples(index=False)):
            rows.append(
                {
                    "incident_number": getattr(r, "incident_number"),
                    "label": int(getattr(r, "label")),
                    "model_name": MODEL_NAME,
                    "embedding_dim": dim,
                    "embedding": emb_list[i],
                }
            )

        insert_batch(engine, rows)
        print(f"✅ Inserted/Upserted {end:,}/{n:,} (dim={dim})")

    print(f"🎉 Done. Output table: {SCHEMA_OUT}.{TABLE_OUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("❌ ERROR:", repr(e), file=sys.stderr)
        raise
