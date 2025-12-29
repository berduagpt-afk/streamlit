# run_tfidf_unigram_kelayakan.py (FINAL - NATIVE DICT -> JSONB via psycopg2.extras.Json)
# Offline TF-IDF Unigram untuk incident_kelayakan -> simpan ke tabel baru
#
# Input : lasis_djp.incident_kelayakan
#   (tgl_submit, incident_number, site, assignee, modul, sub_modul, detailed_decription)
#
# Output (tabel BARU):
#   - lasis_djp.incident_clean_tfidf_runs
#   - lasis_djp.incident_clean_tfidf_vectors
#
# Catatan:
# - TF-IDF disimpan numerik dalam JSONB {feature: weight} (sparse, hanya non-zero).
# - UNIGRAM: ngram_range=(1,1)
# - Penyimpanan native dict dilakukan via psycopg2.extras.Json agar psycopg2 bisa adapt JSONB.

from __future__ import annotations

import os
import sys
import time
import uuid
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sklearn.feature_extraction.text import TfidfVectorizer
from psycopg2.extras import Json


# =============================================================================
# Config
# =============================================================================

@dataclass
class TfidfConfig:
    # source
    schema: str = "lasis_djp"
    source_table: str = "incident_clean"

    # columns (sesuai DDL incident_clean)
    col_date: str = "tgl_submit"
    col_incident: str = "incident_number"
    col_site: str = "site"
    col_assignee: str = "assignee"
    col_modul: str = "modul"
    col_sub_modul: str = "sub_modul"
    col_text: str = "text_sintaksis"

    # filters (opsional)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    modul: Optional[str] = None
    site: Optional[str] = None

    # tfidf params (UNIGRAM)
    min_df: int = 2
    max_df: float = 0.95
    max_features: Optional[int] = None  # None = full vocab
    lowercase: bool = True
    ngram_min: int = 1
    ngram_max: int = 1  # ✅ UNIGRAM

    # store controls
    store_topn: Optional[int] = None  # None = simpan semua non-zero; isi angka untuk batasi top-N
    insert_chunksize: int = 2000
    insert_method: str = "multi"

    # output tables (baru)
    output_schema: str = "lasis_djp"
    runs_table: str = "incident_modeling_tfidf_runs"
    vectors_table: str = "incident_modeling_tfidf_vectors"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--db-url", default=None)

    p.add_argument("--schema", default="lasis_djp")
    p.add_argument("--source-table", default="incident_clean")

    p.add_argument("--start", dest="start_date", default=None)
    p.add_argument("--end", dest="end_date", default=None)
    p.add_argument("--modul", default=None)
    p.add_argument("--site", default=None)

    p.add_argument("--min-df", type=int, default=2)
    p.add_argument("--max-df", type=float, default=0.95)
    p.add_argument("--max-features", type=int, default=None)
    p.add_argument("--lowercase", action="store_true", default=True)
    p.add_argument("--no-lowercase", dest="lowercase", action="store_false")

    p.add_argument("--store-topn", type=int, default=None)
    p.add_argument("--insert-chunksize", type=int, default=2000)
    p.add_argument("--insert-method", default="multi")
    p.add_argument("--output-schema", default="lasis_djp")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def build_config(ns: argparse.Namespace) -> TfidfConfig:
    return TfidfConfig(
        schema=ns.schema,
        source_table=ns.source_table,
        start_date=ns.start_date,
        end_date=ns.end_date,
        modul=ns.modul,
        site=ns.site,
        min_df=ns.min_df,
        max_df=ns.max_df,
        max_features=ns.max_features,
        lowercase=ns.lowercase,
        store_topn=ns.store_topn,
        insert_chunksize=ns.insert_chunksize,
        insert_method=ns.insert_method,
        output_schema=ns.output_schema,
    )


# =============================================================================
# DB helpers
# =============================================================================

def get_engine(db_url: Optional[str]) -> Engine:
    """
    Prioritas:
    1) --db-url (disarankan)
    2) env PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD
    """
    if db_url:
        return create_engine(db_url, pool_pre_ping=True)

    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")
    db = os.getenv("PGDATABASE", "incident_djp")
    user = os.getenv("PGUSER", "postgres")
    pw = os.getenv("PGPASSWORD", "admin*123")

    if not pw:
        raise RuntimeError("PGPASSWORD tidak ditemukan. Set env PGPASSWORD atau gunakan --db-url.")

    return create_engine(f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}", pool_pre_ping=True)


def _exec_ddl(conn, ddl: str) -> None:
    parts = [s.strip() for s in ddl.split(";") if s.strip()]
    for stmt in parts:
        conn.execute(text(stmt))


def ensure_output_tables(engine: Engine, cfg: TfidfConfig) -> None:
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {cfg.output_schema}.{cfg.runs_table} (
        run_id       TEXT PRIMARY KEY,
        run_time     TIMESTAMP NOT NULL,
        approach     TEXT NOT NULL,
        params_json  JSONB NOT NULL,
        data_range   JSONB,
        notes        TEXT
    );

    CREATE TABLE IF NOT EXISTS {cfg.output_schema}.{cfg.vectors_table} (
        run_id              TEXT NOT NULL,
        incident_number     TEXT NOT NULL,
        tgl_submit          TIMESTAMP,
        site                TEXT,
        assignee            TEXT,
        modul               TEXT,
        sub_modul           TEXT,
        text_sintaksis TEXT,
        tfidf_json          JSONB NOT NULL,
        PRIMARY KEY (run_id, incident_number)
    );

    CREATE INDEX IF NOT EXISTS idx_{cfg.vectors_table}_run_modul
        ON {cfg.output_schema}.{cfg.vectors_table} (run_id, modul, tgl_submit);
    """
    with engine.begin() as conn:
        _exec_ddl(conn, ddl)


def fetch_source(engine: Engine, cfg: TfidfConfig) -> pd.DataFrame:
    where = []
    params: Dict[str, object] = {}

    if cfg.start_date:
        where.append(f"{cfg.col_date} >= :start_date")
        params["start_date"] = pd.to_datetime(cfg.start_date)

    if cfg.end_date:
        where.append(f"{cfg.col_date} < :end_date_plus")
        params["end_date_plus"] = pd.to_datetime(cfg.end_date) + pd.Timedelta(days=1)

    if cfg.modul:
        where.append(f"{cfg.col_modul} = :modul")
        params["modul"] = cfg.modul

    if cfg.site:
        where.append(f"{cfg.col_site} = :site")
        params["site"] = cfg.site

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    sql = f"""
    SELECT
        {cfg.col_date}      AS tgl_submit,
        {cfg.col_incident}  AS incident_number,
        {cfg.col_site}      AS site,
        {cfg.col_assignee}  AS assignee,
        {cfg.col_modul}     AS modul,
        {cfg.col_sub_modul} AS sub_modul,
        {cfg.col_text}      AS text_sintaksis
    FROM {cfg.schema}.{cfg.source_table}
    {where_sql}
    ORDER BY {cfg.col_date} ASC
    """

    df = pd.read_sql(text(sql), engine, params=params)

    df["tgl_submit"] = pd.to_datetime(df["tgl_submit"], errors="coerce")

    for c in ["incident_number", "site", "assignee", "modul", "sub_modul"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str)

    if "text_sintaksis" not in df.columns:
        df["text_sintaksis"] = ""
    df["text_sintaksis"] = df["text_sintaksis"].fillna("").astype(str)

    df = df.dropna(subset=["tgl_submit"])
    df = df[df["text_sintaksis"].str.len() > 0].copy()
    return df


# =============================================================================
# TF-IDF helpers
# =============================================================================

def row_sparse_to_dict(
    vectorizer: TfidfVectorizer,
    X_row,
    topn: Optional[int] = None
) -> Dict[str, float]:
    """
    Convert 1xF sparse TF-IDF row to native Python dict {feature: weight}.
    """
    row = X_row.tocsr()
    idx = row.indices
    val = row.data

    if val.size == 0:
        return {}

    if topn is not None and val.size > topn:
        top_pos = np.argpartition(-val, topn - 1)[:topn]
        idx = idx[top_pos]
        val = val[top_pos]

    feature_names = vectorizer.get_feature_names_out()
    d = {str(feature_names[i]): float(v) for i, v in zip(idx, val)}
    return dict(sorted(d.items(), key=lambda kv: -kv[1]))


# =============================================================================
# Main
# =============================================================================

def run_tfidf(engine: Engine, cfg: TfidfConfig, dry_run: bool = False) -> str:
    t0 = time.time()

    df = fetch_source(engine, cfg)
    if df.empty:
        raise RuntimeError("Tidak ada data yang memenuhi filter / kolom teks kosong.")

    texts = df["text_sintaksis"].tolist()

    vectorizer = TfidfVectorizer(
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        max_features=cfg.max_features,
        ngram_range=(cfg.ngram_min, cfg.ngram_max),  # ✅ (1,1)
        lowercase=cfg.lowercase,
        dtype=np.float32,
    )

    X = vectorizer.fit_transform(texts)

    run_id = str(uuid.uuid4())
    run_time = datetime.now()

    n_features = int(len(vectorizer.get_feature_names_out()))

    params_json: Dict[str, Any] = {
        "approach": "tfidf_unigram_from_incident_clean",
        "tfidf": {
            "min_df": int(cfg.min_df),
            "max_df": float(cfg.max_df),
            "max_features": None if cfg.max_features is None else int(cfg.max_features),
            "ngram_range": [cfg.ngram_min, cfg.ngram_max],
            "lowercase": bool(cfg.lowercase),
            "n_features": n_features,
        },
        "store": {
            "store_topn": cfg.store_topn,
            "format": "jsonb_feature_to_weight_sparse",
        },
    }

    data_range: Dict[str, Any] = {
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "modul": cfg.modul,
        "site": cfg.site,
        "n_rows": int(len(df)),
    }

    print("[INFO] n_rows:", len(df))
    print("[INFO] n_features (unigram):", n_features)

    # Build rows
    records = df.to_dict("records")
    rows: List[Dict[str, Any]] = []
    for i, r in enumerate(records):
        rows.append({
            "run_id": run_id,
            "incident_number": r["incident_number"],
            "tgl_submit": r["tgl_submit"],
            "site": r["site"],
            "assignee": r["assignee"],
            "modul": r["modul"],
            "sub_modul": r["sub_modul"],
            "text_sintaksis": r["text_sintaksis"],
            "tfidf_json": row_sparse_to_dict(vectorizer, X[i], topn=cfg.store_topn),
        })

    df_runs = pd.DataFrame([{
        "run_id": run_id,
        "run_time": run_time,
        "approach": "tfidf_unigram_from_incident_clean",
        "params_json": params_json,  # dict (native)
        "data_range": data_range,    # dict (native)
        "notes": None,
    }])

    df_vec = pd.DataFrame(rows)

    # IMPORTANT: wrap dict to psycopg2 Json adapter so it can be inserted into JSONB columns
    df_runs["params_json"] = df_runs["params_json"].apply(Json)
    df_runs["data_range"] = df_runs["data_range"].apply(Json)
    df_vec["tfidf_json"] = df_vec["tfidf_json"].apply(Json)

    elapsed = time.time() - t0
    print(f"[OK] run_id={run_id} rows={len(df_vec):,} elapsed={elapsed:.1f}s")

    if dry_run:
        print("[DRY RUN] Tidak menulis ke database.")
        return run_id

    ensure_output_tables(engine, cfg)

    method = cfg.insert_method if cfg.insert_method and str(cfg.insert_method).lower() != "none" else None
    chunksize = int(cfg.insert_chunksize) if cfg.insert_chunksize else None

    with engine.begin() as conn:
        df_runs.to_sql(
            cfg.runs_table,
            conn,
            schema=cfg.output_schema,
            if_exists="append",
            index=False,
            method=method,
            chunksize=chunksize,
        )

        df_vec.to_sql(
            cfg.vectors_table,
            conn,
            schema=cfg.output_schema,
            if_exists="append",
            index=False,
            method=method,
            chunksize=chunksize,
        )

    return run_id


def main() -> int:
    ns = parse_args()
    cfg = build_config(ns)
    engine = get_engine(ns.db_url)

    print("[CONFIG]", asdict(cfg))

    try:
        run_id = run_tfidf(engine, cfg, dry_run=ns.dry_run)
        print(f"[DONE] run_id={run_id}")
        return 0
    except Exception as e:
        print("[ERROR]", str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
