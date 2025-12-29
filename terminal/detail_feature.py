from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer


# ======================================================
# Config
# ======================================================
DB_URL = "postgresql+psycopg2://postgres:admin*123@localhost:5432/incident_djp"

SCHEMA = "lasis_djp"
SOURCE_TABLE = "incident_clean"
TEXT_COL = "text_sintaksis"
MODUL_COL = "modul"

# TF-IDF params (samakan dengan eksperimen Anda)
VEC_PARAMS = dict(
    min_df=2,
    max_df=0.95,
    # tambahkan jika perlu:
    # ngram_range=(1,2),
    # token_pattern=r"(?u)\b\w+\b",
)

# Output tables
T_FEATURES = "tfidf_features"         # 1 tabel detail feature (utama)
T_FEATURES_SUM = "tfidf_features_summary"  # opsional (ringkas per modul)


# ======================================================
# Helpers
# ======================================================
def utc_now():
    return datetime.now(timezone.utc)


def ensure_tables(engine):
    """Create output tables if not exists."""
    ddl_features = f"""
    CREATE TABLE IF NOT EXISTS {SCHEMA}.{T_FEATURES} (
        run_id      TEXT NOT NULL,
        created_at  TIMESTAMPTZ NOT NULL,
        params_json JSONB NOT NULL,

        modul       TEXT NOT NULL,
        term        TEXT NOT NULL,

        n_docs      INTEGER NOT NULL,
        df          INTEGER NOT NULL,
        df_ratio    DOUBLE PRECISION NOT NULL,
        idf         DOUBLE PRECISION NOT NULL,

        n_features  INTEGER NOT NULL,

        PRIMARY KEY (run_id, modul, term)
    );
    CREATE INDEX IF NOT EXISTS idx_{T_FEATURES}_modul ON {SCHEMA}.{T_FEATURES} (modul);
    CREATE INDEX IF NOT EXISTS idx_{T_FEATURES}_term  ON {SCHEMA}.{T_FEATURES} (term);
    """

    ddl_summary = f"""
    CREATE TABLE IF NOT EXISTS {SCHEMA}.{T_FEATURES_SUM} (
        run_id      TEXT NOT NULL,
        created_at  TIMESTAMPTZ NOT NULL,
        params_json JSONB NOT NULL,

        modul       TEXT NOT NULL,
        n_docs      INTEGER NOT NULL,
        n_features  INTEGER NOT NULL,

        PRIMARY KEY (run_id, modul)
    );
    CREATE INDEX IF NOT EXISTS idx_{T_FEATURES_SUM}_modul ON {SCHEMA}.{T_FEATURES_SUM} (modul);
    """

    with engine.begin() as conn:
        conn.execute(text(ddl_features))
        conn.execute(text(ddl_summary))


def load_source(engine) -> pd.DataFrame:
    sql = f"""
    SELECT {MODUL_COL} AS modul, {TEXT_COL} AS text_sintaksis
    FROM {SCHEMA}.{SOURCE_TABLE}
    WHERE {TEXT_COL} IS NOT NULL
      AND NULLIF(TRIM({TEXT_COL}), '') IS NOT NULL
      AND {MODUL_COL} IS NOT NULL
      AND NULLIF(TRIM({MODUL_COL}), '') IS NOT NULL
    """
    return pd.read_sql(text(sql), engine)


def compute_features_per_modul(df: pd.DataFrame, run_id: str, created_at: datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return:
    - df_features: detail per term per modul
    - df_summary : ringkas per modul (opsional tapi berguna)
    """
    params_json = json.dumps(
        {
            "source": f"{SCHEMA}.{SOURCE_TABLE}",
            "text_col": TEXT_COL,
            "group_by": MODUL_COL,
            "tfidf_params": VEC_PARAMS,
        },
        ensure_ascii=False,
    )

    rows = []
    sums = []

    for modul, g in df.groupby("modul", dropna=True):
        texts = g["text_sintaksis"].astype(str).tolist()
        n_docs = len(texts)

        # Skip modul terlalu kecil (TF-IDF min_df=2 but needs at least 2 docs to be meaningful)
        if n_docs < 2:
            continue

        vectorizer = TfidfVectorizer(**VEC_PARAMS)
        X = vectorizer.fit_transform(texts)  # shape: (n_docs, n_terms)

        terms = vectorizer.get_feature_names_out()
        idf = vectorizer.idf_.astype(float)  # length: n_terms

        # df(term) = jumlah dokumen yang memiliki term (non-zero)
        # (X > 0).sum(axis=0) -> matrix 1 x n_terms
        df_counts = np.asarray((X > 0).sum(axis=0)).ravel().astype(int)

        n_features = len(terms)
        sums.append(
            {
                "run_id": run_id,
                "created_at": created_at,
                "params_json": params_json,
                "modul": modul,
                "n_docs": n_docs,
                "n_features": n_features,
            }
        )

        df_ratio = (df_counts / float(n_docs)).astype(float)

        # Build rows per term
        for term, dfc, dfr, idfv in zip(terms, df_counts, df_ratio, idf):
            rows.append(
                {
                    "run_id": run_id,
                    "created_at": created_at,
                    "params_json": params_json,
                    "modul": modul,
                    "term": term,
                    "n_docs": n_docs,
                    "df": int(dfc),
                    "df_ratio": float(dfr),
                    "idf": float(idfv),
                    "n_features": n_features,  # disimpan supaya mudah agregasi tanpa join
                }
            )

        print(f"{modul:30s} → {n_features:6d} features | docs={n_docs}")

    df_features = pd.DataFrame(rows)
    df_summary = pd.DataFrame(sums)
    return df_features, df_summary


def save_to_db(engine, df_features: pd.DataFrame, df_summary: pd.DataFrame, mode: str = "append"):
    """
    mode:
      - "append": tambah data run baru
      - "replace": hapus isi tabel lalu isi lagi (hati-hati)
    """
    if df_features.empty:
        raise ValueError("Tidak ada fitur yang terbentuk. Cek data & parameter TF-IDF (min_df/max_df)")

    with engine.begin() as conn:
        # Insert detail features
        df_features.to_sql(
            T_FEATURES,
            con=conn,
            schema=SCHEMA,
            if_exists=mode,
            index=False,
            method="multi",
            chunksize=5000,
        )

        # Insert summary (opsional)
        if not df_summary.empty:
            df_summary.to_sql(
                T_FEATURES_SUM,
                con=conn,
                schema=SCHEMA,
                if_exists=mode,
                index=False,
                method="multi",
                chunksize=2000,
            )


# ======================================================
# Main
# ======================================================
def main():
    engine = create_engine(DB_URL, pool_pre_ping=True)

    # 1) pastikan tabel output ada
    ensure_tables(engine)

    # 2) ambil data sumber
    df = load_source(engine)
    print("rows loaded:", len(df))

    # 3) hitung fitur TF-IDF per modul
    run_id = str(uuid.uuid4())
    created_at = utc_now()

    df_features, df_summary = compute_features_per_modul(df, run_id, created_at)

    # 4) simpan ke DB (append run baru)
    save_to_db(engine, df_features, df_summary, mode="append")

    print("\nDONE")
    print("run_id:", run_id)
    print("features rows:", len(df_features))
    print("summary rows :", len(df_summary))


if __name__ == "__main__":
    main()
