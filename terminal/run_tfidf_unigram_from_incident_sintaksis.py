# run_tfidf_unigram_tokens_sintaksis.py (FINAL - FULL, PATCH: IDF ONLY IN RUNS)
# ============================================================
# Offline TF-IDF Unigram dari tokens_sintaksis_json -> simpan ke:
#   - lasis_djp.incident_tfidf_runs        (IDF disimpan sekali per run)
#   - lasis_djp.incident_tfidf_vectors     (tanpa IDF)
#
# Input : lasis_djp.incident_sintaksis
#   (tgl_submit, incident_number, site, assignee, modul, sub_modul, tokens_sintaksis_json)
#
# Output:
#   RUNS (incident_tfidf_runs):
#     - run_id, run_time, approach, params_json, data_range, notes
#     - idf_json             : JSONB {term: idf} (global)
#     - feature_names_json   : JSONB [term1, term2, ...] urutan fitur untuk rekonstruksi vector
#
#   VECTORS (incident_tfidf_vectors):
#     - subject_tokens_json : JSONB list token
#     - tf_json            : JSONB {term: tf}     (TF normalized: count/len(tokens))
#     - tfidf_json         : JSONB {term: tfidf}  (tf * idf)
#     - tfidf_vec_json     : JSONB list dense vector sesuai urutan feature_names_json (optional)
#
# Catatan:
# - IDF tidak di-duplicate per dokumen untuk hemat storage.
# - Dense vector bisa besar. Matikan dengan --no-dense-vec bila perlu.
# ============================================================

from __future__ import annotations

import os
import sys
import time
import uuid
import json
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from collections import Counter

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
    source_table: str = "incident_sintaksis"

    # columns
    col_date: str = "tgl_submit"
    col_incident: str = "incident_number"
    col_site: str = "site"
    col_assignee: str = "assignee"
    col_modul: str = "modul"
    col_sub_modul: str = "sub_modul"
    col_tokens_json: str = "tokens_sintaksis_json"

    # filters (opsional)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    modul: Optional[str] = None
    site: Optional[str] = None

    # tfidf params (UNIGRAM)
    min_df: int = 2
    max_df: float = 0.95
    max_features: Optional[int] = None
    lowercase: bool = True
    ngram_min: int = 1
    ngram_max: int = 1

    # store controls
    store_topn: Optional[int] = None   # top-n untuk tf/tfidf dict (opsional)
    store_vec_dense: bool = True       # simpan tfidf_vec_json dense
    insert_chunksize: int = 2000
    insert_method: str = "multi"

    # output
    output_schema: str = "lasis_djp"
    runs_table: str = "incident_tfidf_runs"
    vectors_table: str = "incident_tfidf_vectors"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--db-url", default=None)

    p.add_argument("--schema", default="lasis_djp")
    p.add_argument("--source-table", default="incident_sintaksis")

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
    p.add_argument("--no-dense-vec", dest="store_vec_dense", action="store_false")

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
        store_vec_dense=bool(getattr(ns, "store_vec_dense", True)),
        insert_chunksize=ns.insert_chunksize,
        insert_method=ns.insert_method,
        output_schema=ns.output_schema,
    )


# =============================================================================
# DB helpers
# =============================================================================

def get_engine(db_url: Optional[str]) -> Engine:
    if db_url:
        return create_engine(db_url, pool_pre_ping=True)

    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")
    db = os.getenv("PGDATABASE", "incident_djp")
    user = os.getenv("PGUSER", "postgres")
    pw = os.getenv("PGPASSWORD", "admin*123")

    if not pw:
        raise RuntimeError("PGPASSWORD tidak ditemukan. Set env PGPASSWORD atau gunakan --db-url.")

    return create_engine(
        f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}",
        pool_pre_ping=True
    )


def _exec_ddl(conn, ddl: str) -> None:
    parts = [s.strip() for s in ddl.split(";") if s.strip()]
    for stmt in parts:
        conn.execute(text(stmt))


def ensure_output_tables(engine: Engine, cfg: TfidfConfig) -> None:
    """
    Patch DDL:
    - RUNS: tambah idf_json + feature_names_json (sekali per run)
    - VECTORS: TANPA idf_json
    """
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {cfg.output_schema}.{cfg.runs_table}
    (
        run_id TEXT PRIMARY KEY,
        run_time TIMESTAMP NOT NULL,
        approach TEXT NOT NULL,
        params_json JSONB NOT NULL,
        data_range JSONB,
        notes TEXT,

        -- ✅ global per run
        idf_json JSONB,
        feature_names_json JSONB
    );

    -- jika tabel runs sudah ada, tambahkan kolom jika belum ada
    ALTER TABLE {cfg.output_schema}.{cfg.runs_table}
        ADD COLUMN IF NOT EXISTS idf_json JSONB;
    ALTER TABLE {cfg.output_schema}.{cfg.runs_table}
        ADD COLUMN IF NOT EXISTS feature_names_json JSONB;

    CREATE TABLE IF NOT EXISTS {cfg.output_schema}.{cfg.vectors_table}
    (
        run_id TEXT NOT NULL,
        incident_number TEXT NOT NULL,
        tgl_submit TIMESTAMP,
        site TEXT,
        assignee TEXT,
        modul TEXT,
        sub_modul TEXT,

        subject_tokens_json JSONB,
        tf_json JSONB,
        tfidf_json JSONB NOT NULL,
        tfidf_vec_json JSONB,

        tokens_sintaksis_json JSONB,
        text_sintaksis TEXT,

        PRIMARY KEY (run_id, incident_number)
    );

    CREATE INDEX IF NOT EXISTS idx_{cfg.vectors_table}_run_modul
        ON {cfg.output_schema}.{cfg.vectors_table} (run_id, modul, tgl_submit);
    """
    with engine.begin() as conn:
        _exec_ddl(conn, ddl)


def _tokens_to_list(val: Union[str, list, None]) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x) for x in val if str(x).strip()]
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed if str(x).strip()]
            except Exception:
                pass
        return [s] if s else []
    return [str(val)]


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
        {cfg.col_date}        AS tgl_submit,
        {cfg.col_incident}    AS incident_number,
        {cfg.col_site}        AS site,
        {cfg.col_assignee}    AS assignee,
        {cfg.col_modul}       AS modul,
        {cfg.col_sub_modul}   AS sub_modul,
        {cfg.col_tokens_json} AS tokens_sintaksis_json
    FROM {cfg.schema}.{cfg.source_table}
    {where_sql}
    ORDER BY {cfg.col_date} ASC
    """
    df = pd.read_sql(text(sql), engine, params=params)

    df["tgl_submit"] = pd.to_datetime(df["tgl_submit"], errors="coerce")
    for c in ["incident_number", "site", "assignee", "modul", "sub_modul"]:
        df[c] = df.get(c, "").fillna("").astype(str)

    df["tokens_sintaksis_json"] = df.get("tokens_sintaksis_json", None).apply(_tokens_to_list)

    if cfg.lowercase:
        df["tokens_sintaksis_json"] = df["tokens_sintaksis_json"].apply(lambda toks: [t.lower() for t in toks])

    df["text_sintaksis"] = df["tokens_sintaksis_json"].apply(lambda toks: " ".join(toks).strip())

    df = df.dropna(subset=["tgl_submit"])
    df = df[df["text_sintaksis"].str.len() > 0].copy()
    return df


# =============================================================================
# TF / IDF / TFIDF helpers
# =============================================================================

def build_idf_dict(vectorizer: TfidfVectorizer) -> Dict[str, float]:
    feats = vectorizer.get_feature_names_out()
    idf_vals = vectorizer.idf_
    out: Dict[str, float] = {}
    for tok, idfv in zip(feats, idf_vals):
        out[str(tok)] = float(idfv)
    return out


def tf_dict_from_tokens(tokens: List[str], vocab_set: set[str]) -> Dict[str, float]:
    n = len(tokens)
    if n == 0:
        return {}
    cnt = Counter(tok for tok in tokens if tok in vocab_set)
    out: Dict[str, float] = {}
    for tok, c in cnt.items():
        out[str(tok)] = float(c) / float(n)
    return out


def tfidf_from_tf_idf(tf: Dict[str, float], idf: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for tok, tfv in tf.items():
        idfv = idf.get(tok)
        if idfv is None:
            continue
        out[str(tok)] = float(tfv) * float(idfv)
    return out


def maybe_topn(d: Dict[str, float], topn: Optional[int]) -> Dict[str, float]:
    if not d:
        return {}
    items = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
    if topn is not None and int(topn) > 0:
        items = items[: int(topn)]
    out: Dict[str, float] = {}
    for k, v in items:
        out[str(k)] = float(v)
    return out


def dense_vec_from_tfidf(tfidf: Dict[str, float], feat_index: Dict[str, int], n_features: int) -> List[float]:
    vec = np.zeros(n_features, dtype=np.float32)
    for tok, v in tfidf.items():
        j = feat_index.get(tok)
        if j is not None:
            vec[j] = float(v)
    return vec.astype(float).tolist()


# =============================================================================
# Main run
# =============================================================================

def run_tfidf(engine: Engine, cfg: TfidfConfig, dry_run: bool = False) -> str:
    t0 = time.time()

    df = fetch_source(engine, cfg)
    if df.empty:
        raise RuntimeError("Tidak ada data yang memenuhi filter / token kosong.")

    vectorizer = TfidfVectorizer(
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        max_features=cfg.max_features,
        ngram_range=(cfg.ngram_min, cfg.ngram_max),
        lowercase=False,          # tokens sudah ditangani manual
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        dtype=np.float32,
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
        norm=None,               # kita hitung TF-IDF manual
    )

    texts = df["text_sintaksis"].tolist()
    vectorizer.fit(texts)

    features = vectorizer.get_feature_names_out()
    n_features = int(len(features))
    vocab_set = set(str(x) for x in features)

    idf_dict = build_idf_dict(vectorizer)

    feat_index: Dict[str, int] = {}
    feature_names_list: List[str] = []
    for i, tok in enumerate(features):
        s = str(tok)
        feat_index[s] = int(i)
        feature_names_list.append(s)

    run_id = str(uuid.uuid4())
    run_time = datetime.now()

    params_json: Dict[str, Any] = {
        "approach": "tfidf_unigram_manual_tf_x_idf",
        "tf": {"type": "normalized_tf", "formula": "count(term)/len(tokens)"},
        "idf": {"source": "sklearn.TfidfVectorizer.idf_", "smooth_idf": True, "stored_in": "runs.idf_json"},
        "tfidf": {"formula": "tf * idf", "norm": None},
        "tfidf_vec": {
            "dense": bool(cfg.store_vec_dense),
            "feature_order": "runs.feature_names_json",
        },
        "vectorizer": {
            "min_df": int(cfg.min_df),
            "max_df": float(cfg.max_df),
            "max_features": None if cfg.max_features is None else int(cfg.max_features),
            "ngram_range": [cfg.ngram_min, cfg.ngram_max],
            "lowercase_tokens": bool(cfg.lowercase),
            "n_features": n_features,
        },
        "source": {
            "table": f"{cfg.schema}.{cfg.source_table}",
            "tokens_column": cfg.col_tokens_json,
            "doc_build": "join_tokens_with_space",
        },
        "store": {
            "store_topn": cfg.store_topn,
            "idf_stored": "runs_only",
            "tf_stored": "vectors.tf_json",
            "tfidf_stored": "vectors.tfidf_json",
            "tfidf_vec_stored": "vectors.tfidf_vec_json" if cfg.store_vec_dense else "none",
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
    print("[INFO] n_features:", n_features)

    rows: List[Dict[str, Any]] = []
    for r in df.to_dict("records"):
        tokens: List[str] = r["tokens_sintaksis_json"]

        tf_full = tf_dict_from_tokens(tokens, vocab_set=vocab_set)
        tf_json = maybe_topn(tf_full, cfg.store_topn)

        tfidf_full = tfidf_from_tf_idf(tf_full, idf_dict)
        tfidf_json = maybe_topn(tfidf_full, cfg.store_topn)

        vec_json = dense_vec_from_tfidf(tfidf_full, feat_index, n_features) if cfg.store_vec_dense else None

        rows.append({
            "run_id": run_id,
            "incident_number": r["incident_number"],
            "tgl_submit": r["tgl_submit"],
            "site": r["site"],
            "assignee": r["assignee"],
            "modul": r["modul"],
            "sub_modul": r["sub_modul"],

            "subject_tokens_json": tokens,
            "tf_json": tf_json,
            "tfidf_json": tfidf_json,
            "tfidf_vec_json": vec_json,

            "tokens_sintaksis_json": tokens,
            "text_sintaksis": r["text_sintaksis"],
        })

    # RUN ROW (IDF + FEATURE NAMES disimpan sekali)
    df_runs = pd.DataFrame([{
        "run_id": run_id,
        "run_time": run_time,
        "approach": "tfidf_unigram_manual_tf_x_idf",
        "params_json": params_json,
        "data_range": data_range,
        "notes": None,
        "idf_json": idf_dict,
        "feature_names_json": feature_names_list,
    }])

    df_vec = pd.DataFrame(rows)

    # JSONB adapt
    df_runs["params_json"] = df_runs["params_json"].apply(Json)
    df_runs["data_range"] = df_runs["data_range"].apply(Json)
    df_runs["idf_json"] = df_runs["idf_json"].apply(Json)
    df_runs["feature_names_json"] = df_runs["feature_names_json"].apply(Json)

    json_cols = [
        "subject_tokens_json",
        "tf_json",
        "tfidf_json",
        "tfidf_vec_json",
        "tokens_sintaksis_json",
    ]
    for c in json_cols:
        if c in df_vec.columns:
            df_vec[c] = df_vec[c].apply(lambda x: Json(x) if x is not None else None)

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
