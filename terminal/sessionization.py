"""
Sessionization (Temporal Split) untuk hasil clustering sintaksis
==============================================================
Sumber tabel (existing):
- lasis_djp.modeling_sintaksis_clusters
- lasis_djp.modeling_sintaksis_members

Parameter:
- job_id (uuid)
- modeling_id (uuid)
- windows (list hari): default [7,14,30]

Logic:
1) Ambil cluster eligible per window:
   (COALESCE(span_days,0) - window_days) > 1
2) Ambil members untuk (job_id, modeling_id, cluster_id) tersebut.
3) Urutkan per cluster_id dan tgl_submit.
4) Split episode: jika gap_days antar tiket berturut > window_days -> episode baru.
5) Simpan hasil ke tabel output:
   - lasis_djp.modeling_sintaksis_temporal_members
   - lasis_djp.modeling_sintaksis_temporal_summary (opsional ringkas per window)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# =========================
# CONFIG
# =========================

@dataclass
class DBConfig:
    host: str = os.getenv("PGHOST", "localhost")
    port: int = int(os.getenv("PGPORT", "5432"))
    database: str = os.getenv("PGDATABASE", "incident_djp")
    user: str = os.getenv("PGUSER", "postgres")
    password: str = os.getenv("PGPASSWORD", "admin*123")


@dataclass
class Config:
    schema: str = "lasis_djp"
    t_clusters: str = "modeling_sintaksis_clusters"
    t_members: str = "modeling_sintaksis_members"
    out_members: str = "modeling_sintaksis_temporal_members"
    out_summary: str = "modeling_sintaksis_temporal_summary"  # opsional
    windows: Tuple[int, ...] = (7, 14, 30)


def get_engine(db: DBConfig) -> Engine:
    url = f"postgresql+psycopg2://{db.user}:{db.password}@{db.host}:{db.port}/{db.database}"
    return create_engine(url, pool_pre_ping=True)


# =========================
# DDL OUTPUT TABLES
# =========================

def ensure_output_tables(engine: Engine, cfg: Config) -> None:
    ddl = f"""
    CREATE SCHEMA IF NOT EXISTS {cfg.schema};

    CREATE TABLE IF NOT EXISTS {cfg.schema}.{cfg.out_members} (
        job_id uuid NOT NULL,
        modeling_id uuid NOT NULL,
        window_days integer NOT NULL,
        cluster_id bigint NOT NULL,
        incident_number text NOT NULL,
        tgl_submit timestamp without time zone,
        site text,
        assignee text,
        modul text,
        sub_modul text,
        gap_days integer,
        temporal_cluster_no integer NOT NULL,
        temporal_cluster_id text NOT NULL,
        PRIMARY KEY (modeling_id, window_days, incident_number)
    );

    CREATE INDEX IF NOT EXISTS idx_{cfg.out_members}_job_model_win
      ON {cfg.schema}.{cfg.out_members} (job_id, modeling_id, window_days);

    CREATE INDEX IF NOT EXISTS idx_{cfg.out_members}_model_win_cluster
      ON {cfg.schema}.{cfg.out_members} (modeling_id, window_days, cluster_id);

    -- optional summary per window
    CREATE TABLE IF NOT EXISTS {cfg.schema}.{cfg.out_summary} (
        job_id uuid NOT NULL,
        modeling_id uuid NOT NULL,
        window_days integer NOT NULL,
        n_clusters_eligible bigint NOT NULL,
        n_clusters_split bigint NOT NULL,
        prop_clusters_split double precision NOT NULL,
        n_clusters_stable bigint NOT NULL,
        prop_clusters_stable double precision NOT NULL,
        total_episodes bigint NOT NULL,
        avg_episode_per_cluster double precision NOT NULL,
        median_episode_per_cluster double precision NOT NULL,
        run_time timestamp with time zone NOT NULL DEFAULT now(),
        PRIMARY KEY (job_id, modeling_id, window_days)
    );
    """
    # execute per statement (robust)
    statements = [s.strip() for s in ddl.split(";") if s.strip()]
    with engine.begin() as conn:
        for stmt in statements:
            conn.exec_driver_sql(stmt)


# =========================
# LOAD ELIGIBLE MEMBERS (PER WINDOW)
# =========================

def load_members_for_window(engine: Engine, cfg: Config, job_id: str, modeling_id: str, window_days: int) -> pd.DataFrame:
    """
    Load only members in clusters eligible for this window:
    (span_days - window_days) > 1
    """
    sql = text(f"""
        WITH eligible_clusters AS (
          SELECT c.cluster_id
          FROM {cfg.schema}.{cfg.t_clusters} c
          WHERE c.job_id = CAST(:job_id AS uuid)
            AND c.modeling_id = CAST(:modeling_id AS uuid)
            AND (COALESCE(c.span_days, 0) - :window_days) > 1
        )
        SELECT
          m.job_id,
          m.modeling_id,
          m.cluster_id,
          m.incident_number,
          m.tgl_submit,
          m.site,
          m.assignee,
          m.modul,
          m.sub_modul
        FROM {cfg.schema}.{cfg.t_members} m
        JOIN eligible_clusters ec
          ON ec.cluster_id = m.cluster_id
        WHERE m.job_id = CAST(:job_id AS uuid)
          AND m.modeling_id = CAST(:modeling_id AS uuid)
          AND m.tgl_submit IS NOT NULL
        ORDER BY m.cluster_id, m.tgl_submit, m.incident_number
    """)
    df = pd.read_sql(
        sql,
        engine,
        params={"job_id": job_id, "modeling_id": modeling_id, "window_days": int(window_days)},
    )
    if df.empty:
        return df
    df["tgl_submit"] = pd.to_datetime(df["tgl_submit"], errors="coerce")
    df = df.dropna(subset=["tgl_submit"]).copy()
    return df


# =========================
# SESSIONIZATION (TEMPORAL SPLIT)
# =========================

def sessionize(df: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """
    df harus sudah berisi baris anggota cluster, sorted by cluster_id, tgl_submit, incident_number.
    Output menambahkan:
      - window_days
      - gap_days (int, hari)
      - temporal_cluster_no (episode number within cluster, start from 1)
      - temporal_cluster_id (text unique: modeling_id-cluster_id-episode)
    """
    if df.empty:
        return df

    out = df.copy()
    out["window_days"] = int(window_days)

    # gunakan basis hari (date) sesuai narasi tesis
    out["tgl"] = out["tgl_submit"].dt.date

    # gap antar tiket berturut dalam cluster
    out["prev_tgl"] = out.groupby(["modeling_id", "cluster_id"])["tgl"].shift(1)
    # gap_days: tgl - prev_tgl (dalam hari)
    out["gap_days"] = (pd.to_datetime(out["tgl"]) - pd.to_datetime(out["prev_tgl"])).dt.days
    # is_new_episode: gap > window => episode baru
    out["is_new_episode"] = np.where(out["prev_tgl"].isna(), 0, (out["gap_days"] > int(window_days)).astype(int))

    # cumulative sum -> episode no (mulai dari 1)
    out["temporal_cluster_no"] = (
        out.groupby(["modeling_id", "cluster_id"])["is_new_episode"].cumsum() + 1
    ).astype(int)

    # temporal_cluster_id unik (text)
    out["temporal_cluster_id"] = (
        out["modeling_id"].astype(str)
        + "-"
        + out["cluster_id"].astype(str)
        + "-"
        + out["temporal_cluster_no"].astype(str)
    )

    # rapikan kolom
    out["gap_days"] = out["gap_days"].astype("Int64")  # nullable int
    keep_cols = [
        "job_id", "modeling_id", "window_days", "cluster_id",
        "incident_number", "tgl_submit", "site", "assignee", "modul", "sub_modul",
        "gap_days", "temporal_cluster_no", "temporal_cluster_id"
    ]
    out = out[keep_cols].copy()
    return out


# =========================
# SUMMARY PER WINDOW (OPSIONAL)
# =========================

def compute_summary(df_sessionized: pd.DataFrame, job_id: str, modeling_id: str, window_days: int) -> pd.DataFrame:
    """
    Menghasilkan 1 baris ringkasan metrik per window:
    n_clusters_eligible, split, stable, total episodes, avg/median episodes per cluster
    """
    if df_sessionized.empty:
        return pd.DataFrame([{
            "job_id": job_id,
            "modeling_id": modeling_id,
            "window_days": int(window_days),
            "n_clusters_eligible": 0,
            "n_clusters_split": 0,
            "prop_clusters_split": 0.0,
            "n_clusters_stable": 0,
            "prop_clusters_stable": 0.0,
            "total_episodes": 0,
            "avg_episode_per_cluster": 0.0,
            "median_episode_per_cluster": 0.0,
        }])

    g = df_sessionized.groupby(["cluster_id"], as_index=False).agg(
        n_episode=("temporal_cluster_no", "nunique")
    )
    n_clusters = int(len(g))
    n_split = int((g["n_episode"] > 1).sum())
    n_stable = int((g["n_episode"] == 1).sum())
    total_episodes = int(g["n_episode"].sum())
    avg_ep = float(g["n_episode"].mean())
    med_ep = float(g["n_episode"].median())
    prop_split = float(n_split / n_clusters) if n_clusters else 0.0
    prop_stable = float(n_stable / n_clusters) if n_clusters else 0.0

    return pd.DataFrame([{
        "job_id": job_id,
        "modeling_id": modeling_id,
        "window_days": int(window_days),
        "n_clusters_eligible": n_clusters,
        "n_clusters_split": n_split,
        "prop_clusters_split": prop_split,
        "n_clusters_stable": n_stable,
        "prop_clusters_stable": prop_stable,
        "total_episodes": total_episodes,
        "avg_episode_per_cluster": avg_ep,
        "median_episode_per_cluster": med_ep,
    }])


# =========================
# SAVE
# =========================

def upsert_summary(engine: Engine, cfg: Config, summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        return
    # upsert per (job_id, modeling_id, window_days)
    sql = text(f"""
        INSERT INTO {cfg.schema}.{cfg.out_summary}
        (job_id, modeling_id, window_days,
         n_clusters_eligible, n_clusters_split, prop_clusters_split,
         n_clusters_stable, prop_clusters_stable,
         total_episodes, avg_episode_per_cluster, median_episode_per_cluster)
        VALUES
        (CAST(:job_id AS uuid), CAST(:modeling_id AS uuid), :window_days,
         :n_clusters_eligible, :n_clusters_split, :prop_clusters_split,
         :n_clusters_stable, :prop_clusters_stable,
         :total_episodes, :avg_episode_per_cluster, :median_episode_per_cluster)
        ON CONFLICT (job_id, modeling_id, window_days) DO UPDATE SET
          n_clusters_eligible = EXCLUDED.n_clusters_eligible,
          n_clusters_split = EXCLUDED.n_clusters_split,
          prop_clusters_split = EXCLUDED.prop_clusters_split,
          n_clusters_stable = EXCLUDED.n_clusters_stable,
          prop_clusters_stable = EXCLUDED.prop_clusters_stable,
          total_episodes = EXCLUDED.total_episodes,
          avg_episode_per_cluster = EXCLUDED.avg_episode_per_cluster,
          median_episode_per_cluster = EXCLUDED.median_episode_per_cluster,
          run_time = now()
    """)
    row = summary_df.iloc[0].to_dict()
    with engine.begin() as conn:
        conn.execute(sql, row)


def save_sessionized(engine: Engine, cfg: Config, df_out: pd.DataFrame) -> None:
    if df_out.empty:
        return

    # Hapus dulu untuk (job_id, modeling_id, window_days) agar idempotent
    job_id = str(df_out["job_id"].iloc[0])
    modeling_id = str(df_out["modeling_id"].iloc[0])
    window_days = int(df_out["window_days"].iloc[0])

    with engine.begin() as conn:
        conn.execute(
            text(f"""
                DELETE FROM {cfg.schema}.{cfg.out_members}
                WHERE job_id = CAST(:job_id AS uuid)
                  AND modeling_id = CAST(:modeling_id AS uuid)
                  AND window_days = :window_days
            """),
            {"job_id": job_id, "modeling_id": modeling_id, "window_days": window_days},
        )

        df_out.to_sql(
            name=cfg.out_members,
            con=conn,
            schema=cfg.schema,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=5000,
        )


# =========================
# MAIN
# =========================

def run(job_id: str, modeling_id: str, windows: Iterable[int]) -> None:
    cfg = Config(windows=tuple(int(w) for w in windows))
    engine = get_engine(DBConfig())
    ensure_output_tables(engine, cfg)

    for w in cfg.windows:
        df = load_members_for_window(engine, cfg, job_id=job_id, modeling_id=modeling_id, window_days=int(w))
        print(f"[LOAD] window={w} | rows={len(df):,}")

        df_sess = sessionize(df, window_days=int(w))
        print(f"[SESSIONIZE] window={w} | rows_out={len(df_sess):,} | clusters={df_sess['cluster_id'].nunique() if not df_sess.empty else 0}")

        save_sessionized(engine, cfg, df_sess)
        print(f"[SAVE] {cfg.schema}.{cfg.out_members} | window={w}")

        summary = compute_summary(df_sess, job_id=job_id, modeling_id=modeling_id, window_days=int(w))
        upsert_summary(engine, cfg, summary)
        print(f"[SUMMARY] window={w} | {summary.to_dict(orient='records')[0]}")

    print("[DONE] sessionization completed for windows:", list(cfg.windows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", required=True, help="job_id UUID")
    ap.add_argument("--modeling-id", required=True, help="modeling_id UUID")
    ap.add_argument("--windows", default="7,14,30", help="comma-separated windows (days), e.g. 7,14,30")
    args = ap.parse_args()

    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
    run(job_id=args.job_id, modeling_id=args.modeling_id, windows=windows)
