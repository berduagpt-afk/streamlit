# run_modeling.py (PATCHED)
# Offline / Batch Modeling — Sintaksis (TF-IDF + Cosine Similarity Threshold)
# Segmentasi: modul + window waktu (sub_modul diabaikan untuk grouping)
# Source: lasis_djp.incident_clean
# Output: lasis_djp.modeling_runs, lasis_djp.cluster_summary, lasis_djp.cluster_members

from __future__ import annotations

import os
import sys
import json
import uuid
import time
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


# =============================================================================
# Config & CLI
# =============================================================================

@dataclass
class ModelingConfig:
    schema: str = "lasis_djp"
    source_table: str = "incident_clean"

    # Kolom sumber (sesuai DDL incident_clean)
    col_incident: str = "incident_number"
    col_date: str = "tgl_submit"
    col_site: str = "site"
    col_assignee: str = "assignee"
    col_modul: str = "modul"
    col_sub_modul: str = "sub_modul"  # tetap dibaca & disimpan (metadata), tapi tidak dipakai grouping
    col_detail: str = "detailed_decription"
    col_text: str = "text_sintaksis"
    col_preprocessed_time: str = "tgl_preprocessed"

    # Filter
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    modul: Optional[str] = None
    site: Optional[str] = None

    # Segmentasi waktu
    window_days: int = 7

    # TF-IDF
    max_features: int = 20000
    min_df: int = 2
    max_df: float = 0.95
    ngram_min: int = 1
    ngram_max: int = 2

    # Cosine threshold + min cluster
    cosine_threshold: float = 0.75
    min_cluster_size: int = 2

    # Performance
    max_rows: Optional[int] = None
    full_matrix_limit: int = 2500
    nn_topk: int = 30

    # Output
    output_schema: str = "lasis_djp"
    runs_table: str = "modeling_runs"
    summary_table: str = "cluster_summary"
    members_table: str = "cluster_members"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--db-url", default=None)
    p.add_argument("--schema", default="lasis_djp")
    p.add_argument("--source-table", default="incident_clean")

    p.add_argument("--start", dest="start_date", default=None)
    p.add_argument("--end", dest="end_date", default=None)
    p.add_argument("--modul", default=None)
    p.add_argument("--site", default=None)

    p.add_argument("--window", dest="window_days", type=int, default=7)
    p.add_argument("--threshold", dest="cosine_threshold", type=float, default=0.75)
    p.add_argument("--min-cluster-size", type=int, default=2)

    p.add_argument("--max-features", type=int, default=20000)
    p.add_argument("--min-df", type=int, default=2)
    p.add_argument("--max-df", type=float, default=0.95)
    p.add_argument("--ngram-min", type=int, default=1)
    p.add_argument("--ngram-max", type=int, default=2)

    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--full-matrix-limit", type=int, default=2500)
    p.add_argument("--nn-topk", type=int, default=30)

    p.add_argument("--output-schema", default="lasis_djp")
    p.add_argument("--dry-run", action="store_true")

    return p.parse_args()


def build_config(ns: argparse.Namespace) -> ModelingConfig:
    return ModelingConfig(
        schema=ns.schema,
        source_table=ns.source_table,
        start_date=ns.start_date,
        end_date=ns.end_date,
        modul=ns.modul,
        site=ns.site,
        window_days=ns.window_days,
        cosine_threshold=ns.cosine_threshold,
        min_cluster_size=ns.min_cluster_size,
        max_features=ns.max_features,
        min_df=ns.min_df,
        max_df=ns.max_df,
        ngram_min=ns.ngram_min,
        ngram_max=ns.ngram_max,
        max_rows=ns.max_rows,
        full_matrix_limit=ns.full_matrix_limit,
        nn_topk=ns.nn_topk,
        output_schema=ns.output_schema,
    )


# =============================================================================
# DB
# =============================================================================

def get_engine(db_url: Optional[str]) -> Engine:
    """
    Prioritas:
    1) --db-url (paling direkomendasikan)
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
        raise RuntimeError(
            "PGPASSWORD tidak ditemukan. "
            "Set env PGPASSWORD atau gunakan --db-url."
        )

    return create_engine(f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}", pool_pre_ping=True)


def _exec_ddl_statements(conn, ddl: str) -> None:
    """
    Jalankan DDL multi-statement dengan aman: eksekusi per statement.
    """
    parts = [s.strip() for s in ddl.split(";") if s.strip()]
    for stmt in parts:
        conn.execute(text(stmt))


def ensure_output_tables(engine: Engine, cfg: ModelingConfig) -> None:
    """
    Membuat tabel bila belum ada.
    Plus: memastikan kolom threshold ada di modeling_runs (ALTER TABLE jika perlu).
    """
    ddl_create = f"""
    CREATE TABLE IF NOT EXISTS {cfg.output_schema}.{cfg.runs_table} (
        run_id      TEXT PRIMARY KEY,
        run_time    TIMESTAMP NOT NULL,
        approach    TEXT NOT NULL,
        threshold   DOUBLE PRECISION,
        params_json JSONB NOT NULL,
        data_range  JSONB,
        notes       TEXT
    );

    CREATE TABLE IF NOT EXISTS {cfg.output_schema}.{cfg.summary_table} (
        run_id                  TEXT NOT NULL,
        cluster_id              TEXT NOT NULL,
        modul                   TEXT,
        sub_modul               TEXT,
        window_start            DATE,
        window_end              DATE,
        n_tickets               INTEGER NOT NULL,
        first_seen              TIMESTAMP,
        last_seen               TIMESTAMP,
        representative_incident TEXT,
        representative_text     TEXT,
        top_terms               TEXT,
        metrics_json            JSONB,
        PRIMARY KEY (run_id, cluster_id)
    );

    CREATE TABLE IF NOT EXISTS {cfg.output_schema}.{cfg.members_table} (
        run_id              TEXT NOT NULL,
        cluster_id          TEXT NOT NULL,
        incident_number     TEXT NOT NULL,
        tgl_submit          TIMESTAMP,
        site                TEXT,
        assignee            TEXT,
        modul               TEXT,
        sub_modul           TEXT,
        detailed_decription TEXT,
        text_sintaksis      TEXT,
        tgl_preprocessed    TIMESTAMPTZ,
        PRIMARY KEY (run_id, cluster_id, incident_number)
    );

    CREATE INDEX IF NOT EXISTS idx_{cfg.members_table}_run_cluster
        ON {cfg.output_schema}.{cfg.members_table} (run_id, cluster_id);

    CREATE INDEX IF NOT EXISTS idx_{cfg.summary_table}_run_modul
        ON {cfg.output_schema}.{cfg.summary_table} (run_id, modul, window_start);
    """

    ddl_alter_threshold = f"""
    ALTER TABLE {cfg.output_schema}.{cfg.runs_table}
    ADD COLUMN IF NOT EXISTS threshold DOUBLE PRECISION;
    """

    with engine.begin() as conn:
        _exec_ddl_statements(conn, ddl_create)
        _exec_ddl_statements(conn, ddl_alter_threshold)


def fetch_source_data(engine: Engine, cfg: ModelingConfig) -> pd.DataFrame:
    where = []
    params: Dict[str, object] = {}

    # tanggal: end_date_plus di Python (hindari cast date di SQL)
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
    limit_sql = "LIMIT :max_rows" if cfg.max_rows else ""
    if cfg.max_rows:
        params["max_rows"] = int(cfg.max_rows)

    sql = f"""
        SELECT
            {cfg.col_date}              AS tgl_submit,
            {cfg.col_incident}          AS incident_number,
            {cfg.col_site}              AS site,
            {cfg.col_assignee}          AS assignee,
            {cfg.col_modul}             AS modul,
            {cfg.col_sub_modul}         AS sub_modul,
            {cfg.col_detail}            AS detailed_decription,
            {cfg.col_text}              AS text_sintaksis,
            {cfg.col_preprocessed_time} AS tgl_preprocessed
        FROM {cfg.schema}.{cfg.source_table}
        {where_sql}
        ORDER BY {cfg.col_date} ASC
        {limit_sql}
    """

    df = pd.read_sql(text(sql), engine, params=params)

    df["tgl_submit"] = pd.to_datetime(df["tgl_submit"], errors="coerce")
    df["tgl_preprocessed"] = pd.to_datetime(df["tgl_preprocessed"], errors="coerce", utc=True)

    # NA-safe: jangan sampai jadi string "nan"
    for c in ["incident_number", "site", "assignee", "modul", "sub_modul"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str)

    df["detailed_decription"] = df.get("detailed_decription", "").fillna("").astype(str)
    df["text_sintaksis"] = df.get("text_sintaksis", "").fillna("").astype(str)

    df = df.dropna(subset=["tgl_submit"])
    df = df[df["text_sintaksis"].str.len() > 0].copy()
    return df


# =============================================================================
# Time window
# =============================================================================

def compute_window_start(ts: pd.Timestamp, window_days: int) -> date:
    d0 = ts.normalize().date()
    epoch = date(1970, 1, 1)
    delta_days = (d0 - epoch).days
    bucket = (delta_days // window_days) * window_days
    return epoch + timedelta(days=bucket)


def add_time_window_columns(df: pd.DataFrame, window_days: int) -> pd.DataFrame:
    df = df.copy()
    df["window_start"] = df["tgl_submit"].apply(lambda x: compute_window_start(x, window_days))
    df["window_end"] = df["window_start"].apply(lambda d: d + timedelta(days=window_days - 1))
    return df


# =============================================================================
# Cosine threshold clustering (connected components)
# =============================================================================

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

    def groups(self) -> Dict[int, List[int]]:
        out: Dict[int, List[int]] = {}
        for i in range(len(self.parent)):
            r = self.find(i)
            out.setdefault(r, []).append(i)
        return out


def _adaptive_topk(n: int, base_topk: int) -> int:
    """
    Top-k adaptif untuk mengurangi risiko cluster "pecah" saat pakai kNN.
    - minimum: base_topk
    - naik mengikuti sqrt(n) (tapi dibatasi)
    """
    if n <= 2:
        return 1
    boost = int(np.sqrt(n) * 5)
    k = max(base_topk, boost)
    k = min(k, 200)           # batas aman (bisa kamu naikkan bila perlu)
    return min(k, n - 1)


def cluster_segment_threshold(X_tfidf, threshold: float, full_matrix_limit: int, nn_topk: int) -> List[int]:
    n = X_tfidf.shape[0]
    uf = UnionFind(n)

    if n <= 1:
        return [0] * n

    if n <= full_matrix_limit:
        S = cosine_similarity(X_tfidf)
        for i in range(n):
            for j in range(i + 1, n):
                if S[i, j] >= threshold:
                    uf.union(i, j)
    else:
        # kNN brute cosine: sim = 1 - distance
        k = _adaptive_topk(n, nn_topk)
        nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
        nn.fit(X_tfidf)
        distances, indices = nn.kneighbors(X_tfidf, return_distance=True)
        for i in range(n):
            for dist, j in zip(distances[i], indices[i]):
                if i == j:
                    continue
                sim = 1.0 - float(dist)
                if sim >= threshold:
                    uf.union(i, int(j))

    groups = uf.groups()
    roots = sorted(groups.keys(), key=lambda r: (-len(groups[r]), r))
    root_to_label = {r: idx for idx, r in enumerate(roots)}
    return [root_to_label[uf.find(i)] for i in range(n)]


# =============================================================================
# Helpers (np.matrix safe)
# =============================================================================

def compute_top_terms(vectorizer: TfidfVectorizer, X_seg, local_positions: List[int], topk: int = 12) -> str:
    if not local_positions:
        return ""
    Xc = X_seg[local_positions]
    mean_vec = np.asarray(Xc.mean(axis=0)).ravel()
    if mean_vec.size == 0:
        return ""
    top_idx = np.argsort(-mean_vec)[:topk]
    terms = vectorizer.get_feature_names_out()
    chosen = [terms[i] for i in top_idx if mean_vec[i] > 0]
    return ", ".join(chosen)


def pick_representative_pos(X_seg, local_positions: List[int]) -> Optional[int]:
    if not local_positions:
        return None
    Xc = X_seg[local_positions]
    centroid = np.asarray(Xc.mean(axis=0))
    sims = cosine_similarity(Xc, centroid).ravel()
    best_local = int(np.argmax(sims))
    return local_positions[best_local]


# =============================================================================
# Main pipeline
# =============================================================================

def run_modeling(engine: Engine, cfg: ModelingConfig, dry_run: bool = False) -> str:
    t0 = time.time()

    df = fetch_source_data(engine, cfg)
    if df.empty:
        raise RuntimeError("Tidak ada data yang memenuhi filter (cek koneksi DB/schema/tanggal/modul).")

    df = add_time_window_columns(df, cfg.window_days)
    df = df.reset_index(drop=True)
    df["_row_id"] = np.arange(len(df))

    vectorizer = TfidfVectorizer(
        max_features=cfg.max_features,
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        ngram_range=(cfg.ngram_min, cfg.ngram_max),
        lowercase=False,
        dtype=np.float32,
    )

    # TF-IDF global, lalu slicing per segmen
    X_all = vectorizer.fit_transform(df["text_sintaksis"].tolist())

    run_id = str(uuid.uuid4())
    run_time = datetime.now()

    params_json: Dict[str, Any] = {
        "approach": "sintaksis_tfidf_cosine_threshold",
        "segmentation": "modul + window_start (sub_modul ignored)",
        "window_days": cfg.window_days,
        "cosine_threshold": cfg.cosine_threshold,
        "min_cluster_size": cfg.min_cluster_size,
        "tfidf": {
            "max_features": cfg.max_features,
            "min_df": cfg.min_df,
            "max_df": cfg.max_df,
            "ngram_range": [cfg.ngram_min, cfg.ngram_max],
        },
        "performance": {
            "full_matrix_limit": cfg.full_matrix_limit,
            "nn_topk_base": cfg.nn_topk,
            "nn_topk_adaptive": True,
        },
    }

    data_range: Dict[str, Any] = {
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "modul": cfg.modul,
        "site": cfg.site,
        "n_rows": int(len(df)),
        "note": "sub_modul diabaikan untuk grouping; disimpan sebagai metadata pada cluster_members",
    }

    summary_rows: List[Dict[str, Any]] = []
    member_rows: List[Dict[str, Any]] = []

    grouped = df.groupby(["modul", "window_start"], sort=True)
    cluster_counter = 0

    for (modul, wstart), g in grouped:
        if len(g) < cfg.min_cluster_size:
            continue

        g = g.reset_index(drop=True)
        idx_global = g["_row_id"].to_numpy()
        X_seg = X_all[idx_global]
        n = int(X_seg.shape[0])

        labels = cluster_segment_threshold(
            X_seg,
            threshold=cfg.cosine_threshold,
            full_matrix_limit=cfg.full_matrix_limit,
            nn_topk=cfg.nn_topk,
        )
        labels = np.asarray(labels)

        for cl in np.unique(labels):
            local_positions = np.where(labels == cl)[0].tolist()
            if len(local_positions) < cfg.min_cluster_size:
                continue

            cluster_id = f"{modul}__{wstart}__{cluster_counter}"
            cluster_counter += 1

            rep_pos = pick_representative_pos(X_seg, local_positions)
            if rep_pos is None:
                rep_pos = local_positions[0]

            rep_row = g.iloc[int(rep_pos)]
            top_terms = compute_top_terms(vectorizer, X_seg, local_positions, topk=12)

            gc = g.iloc[local_positions]
            first_seen = gc["tgl_submit"].min()
            last_seen = gc["tgl_submit"].max()

            summary_rows.append({
                "run_id": run_id,
                "cluster_id": cluster_id,
                "modul": str(modul),
                "sub_modul": None,
                "window_start": wstart,
                "window_end": (wstart + timedelta(days=cfg.window_days - 1)),
                "n_tickets": int(len(gc)),
                "first_seen": first_seen,
                "last_seen": last_seen,
                "representative_incident": str(rep_row["incident_number"]),
                "representative_text": str(rep_row["text_sintaksis"]),
                "top_terms": top_terms,
                "metrics_json": json.dumps({
                "segment_size": int(n),
                "cluster_size": int(len(gc)),
                "cosine_threshold": float(cfg.cosine_threshold),
		}, ensure_ascii=False),

            })

            for _, row in gc.iterrows():
                member_rows.append({
                    "run_id": run_id,
                    "cluster_id": cluster_id,
                    "incident_number": str(row["incident_number"]),
                    "tgl_submit": row["tgl_submit"],
                    "site": str(row["site"]),
                    "assignee": str(row["assignee"]),
                    "modul": str(row["modul"]),
                    "sub_modul": str(row["sub_modul"]),
                    "detailed_decription": str(row["detailed_decription"]),
                    "text_sintaksis": str(row["text_sintaksis"]),
                    "tgl_preprocessed": row["tgl_preprocessed"],
                })

    df_runs = pd.DataFrame([{
    "run_id": run_id,
    "run_time": run_time,
    "approach": "sintaksis_tfidf_cosine_threshold",
    "threshold": float(cfg.cosine_threshold),
    "params_json": json.dumps(params_json, ensure_ascii=False),
    "data_range": json.dumps(data_range, ensure_ascii=False),
    "notes": None,
}])


    df_summary = pd.DataFrame(summary_rows)
    df_members = pd.DataFrame(member_rows)

    elapsed = time.time() - t0
    print(f"[OK] run_id={run_id}")
    print(f"[INFO] threshold={cfg.cosine_threshold}")
    print(f"[INFO] clusters={len(df_summary):,} members={len(df_members):,} source_rows={len(df):,} elapsed={elapsed:.1f}s")

    if dry_run:
        print("[DRY RUN] Tidak menulis ke database.")
        return run_id

    ensure_output_tables(engine, cfg)

    with engine.begin() as conn:
        df_runs.to_sql(cfg.runs_table, conn, schema=cfg.output_schema, if_exists="append", index=False)
        if not df_summary.empty:
            df_summary.to_sql(cfg.summary_table, conn, schema=cfg.output_schema, if_exists="append", index=False)
        if not df_members.empty:
            df_members.to_sql(cfg.members_table, conn, schema=cfg.output_schema, if_exists="append", index=False)

    return run_id


def main() -> int:
    ns = parse_args()
    cfg = build_config(ns)
    engine = get_engine(ns.db_url)

    print("[CONFIG]", json.dumps(asdict(cfg), indent=2, default=str))

    try:
        run_id = run_modeling(engine, cfg, dry_run=ns.dry_run)
        print(f"[DONE] run_id={run_id}")
        return 0
    except Exception as e:
        print("[ERROR]", str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
