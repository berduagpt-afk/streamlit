"""
Evaluasi Temporal (Sessionization / Temporal Split) — Pendekatan Semantik (HDBSCAN)
=================================================================================

PATCH FINAL:
1) Eligible cluster:
   - Jika tabel clusters punya kolom span_days => pakai rule span_days (strict/gt/all)
   - Jika span_days TIDAK ada => hitung span dari members (MIN/MAX event_time per cluster)
     lalu terapkan rule yang setara (strict/gt/all).

2) Kolom metadata opsional (site/assignee/modul/sub_modul):
   - Jika kolom tidak ada di members, akan di-select sebagai NULL::text.

3) Idempotent save:
   - DELETE subset (modeling_id, window_days, time_col) sebelum insert ulang.

4) PATCH penting (WINDOW SANITY):
   - Sorting eksplisit sebelum hitung gap
   - Print distribusi gap_days untuk melihat apakah window 7/14/30 benar-benar berpengaruh

Cara pakai:
python run_temporal_semantik.py --modeling-id <UUID> --windows 7,14,30 --time-col tgl_submit --eligible-rule span_days_gt
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable, Tuple

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
    t_clusters: str = "modeling_semantik_hdbscan_clusters"
    t_members: str = "modeling_semantik_hdbscan_members"
    out_members: str = "modeling_semantik_temporal_members"
    out_summary: str = "modeling_semantik_temporal_summary"
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
        modeling_id uuid NOT NULL,
        window_days integer NOT NULL,
        cluster_id bigint NOT NULL,
        incident_number text NOT NULL,

        time_col text NOT NULL,
        event_time timestamp without time zone,

        site text,
        assignee text,
        modul text,
        sub_modul text,

        gap_days integer,
        temporal_cluster_no integer NOT NULL,
        temporal_cluster_id text NOT NULL,

        PRIMARY KEY (modeling_id, window_days, incident_number, time_col)
    );

    CREATE INDEX IF NOT EXISTS idx_{cfg.out_members}_model_win
      ON {cfg.schema}.{cfg.out_members} (modeling_id, window_days);

    CREATE INDEX IF NOT EXISTS idx_{cfg.out_members}_model_win_cluster
      ON {cfg.schema}.{cfg.out_members} (modeling_id, window_days, cluster_id);

    CREATE INDEX IF NOT EXISTS idx_{cfg.out_members}_episode
      ON {cfg.schema}.{cfg.out_members} (modeling_id, window_days, temporal_cluster_id);

    CREATE TABLE IF NOT EXISTS {cfg.schema}.{cfg.out_summary} (
        modeling_id uuid NOT NULL,
        window_days integer NOT NULL,

        time_col text NOT NULL,
        include_noise boolean NOT NULL,
        eligible_rule text NOT NULL,

        n_clusters_eligible bigint NOT NULL,
        n_clusters_split bigint NOT NULL,
        prop_clusters_split double precision NOT NULL,
        n_clusters_stable bigint NOT NULL,
        prop_clusters_stable double precision NOT NULL,
        total_episodes bigint NOT NULL,
        avg_episode_per_cluster double precision NOT NULL,
        median_episode_per_cluster double precision NOT NULL,

        run_time timestamp with time zone NOT NULL DEFAULT now(),
        PRIMARY KEY (modeling_id, window_days, time_col, include_noise)
    );
    """
    statements = [s.strip() for s in ddl.split(";") if s.strip()]
    with engine.begin() as conn:
        for stmt in statements:
            conn.exec_driver_sql(stmt)


# =========================
# Introspection helper
# =========================

def table_has_column(engine: Engine, schema: str, table: str, column: str) -> bool:
    q = text("""
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = :schema
          AND table_name = :table
          AND column_name = :col
        LIMIT 1
    """)
    with engine.begin() as conn:
        r = conn.execute(q, {"schema": schema, "table": table, "col": column}).fetchone()
    return r is not None


def select_optional_text_cols(engine: Engine, schema: str, table: str, cols: list[str], alias_prefix: str = "m") -> str:
    parts: list[str] = []
    for c in cols:
        if table_has_column(engine, schema, table, c):
            parts.append(f"{alias_prefix}.{c} AS {c}")
        else:
            parts.append(f"NULL::text AS {c}")
    return ",\n          ".join(parts)


def pick_time_column(engine: Engine, cfg: Config, preferred: str) -> str:
    candidates = [preferred, "tgl_submit", "tgl_semantik"]
    for c in candidates:
        if c and table_has_column(engine, cfg.schema, cfg.t_members, c):
            return c
    raise RuntimeError(
        f"Tidak menemukan kolom waktu pada {cfg.schema}.{cfg.t_members}. "
        f"Pastikan ada salah satu kolom: {candidates}"
    )


# =========================
# Eligible filter builders
# =========================

def build_eligible_filter_for_clusters(rule: str, include_noise: bool) -> tuple[str, str]:
    noise_clause = "" if include_noise else "AND c.cluster_id <> -1"

    if rule == "all":
        return f"TRUE {noise_clause}", "all"
    if rule == "span_days_gt":
        return f"(COALESCE(c.span_days,0) > :window_days) {noise_clause}", "span_days_gt"
    return f"((COALESCE(c.span_days,0) - :window_days) > 1) {noise_clause}", "span_days_strict"


def build_eligible_filter_for_members_span(rule: str, include_noise: bool) -> tuple[str, str]:
    noise_clause = "" if include_noise else "AND s.cluster_id <> -1"

    if rule == "all":
        return f"TRUE {noise_clause}", "all (computed span from members)"
    if rule == "span_days_gt":
        return f"(s.span_days > :window_days) {noise_clause}", "span_days_gt (computed span from members)"
    return f"((s.span_days - :window_days) > 1) {noise_clause}", "span_days_strict (computed span from members)"


# =========================
# LOAD MEMBERS (PER WINDOW)
# =========================

def load_members_for_window(
    engine: Engine,
    cfg: Config,
    modeling_id: str,
    window_days: int,
    time_col: str,
    include_noise: bool,
    eligible_rule: str,
) -> pd.DataFrame:
    has_span = table_has_column(engine, cfg.schema, cfg.t_clusters, "span_days")

    optional_cols_sql = select_optional_text_cols(
        engine,
        cfg.schema,
        cfg.t_members,
        cols=["site", "assignee", "modul", "sub_modul"],
        alias_prefix="m",
    )

    if has_span:
        eligible_filter, eligible_rule_effective = build_eligible_filter_for_clusters(eligible_rule, include_noise)

        sql = text(f"""
            WITH eligible_clusters AS (
              SELECT c.modeling_id, c.cluster_id
              FROM {cfg.schema}.{cfg.t_clusters} c
              WHERE c.modeling_id = CAST(:modeling_id AS uuid)
                AND {eligible_filter}
            )
            SELECT
              m.modeling_id,
              m.cluster_id,
              m.incident_number,
              m.{time_col} AS event_time,
              {optional_cols_sql}
            FROM {cfg.schema}.{cfg.t_members} m
            JOIN eligible_clusters ec
              ON ec.modeling_id = m.modeling_id
             AND ec.cluster_id  = m.cluster_id
            WHERE m.modeling_id = CAST(:modeling_id AS uuid)
              AND m.{time_col} IS NOT NULL
            ORDER BY m.cluster_id, m.{time_col}, m.incident_number
        """)
    else:
        eligible_filter, eligible_rule_effective = build_eligible_filter_for_members_span(eligible_rule, include_noise)

        sql = text(f"""
            WITH span_by_cluster AS (
              SELECT
                m.modeling_id,
                m.cluster_id,
                (MAX(m.{time_col}::date) - MIN(m.{time_col}::date))::int AS span_days
              FROM {cfg.schema}.{cfg.t_members} m
              WHERE m.modeling_id = CAST(:modeling_id AS uuid)
                AND m.{time_col} IS NOT NULL
              GROUP BY m.modeling_id, m.cluster_id
            ),
            eligible_clusters AS (
              SELECT s.modeling_id, s.cluster_id
              FROM span_by_cluster s
              WHERE s.modeling_id = CAST(:modeling_id AS uuid)
                AND {eligible_filter}
            )
            SELECT
              m.modeling_id,
              m.cluster_id,
              m.incident_number,
              m.{time_col} AS event_time,
              {optional_cols_sql}
            FROM {cfg.schema}.{cfg.t_members} m
            JOIN eligible_clusters ec
              ON ec.modeling_id = m.modeling_id
             AND ec.cluster_id  = m.cluster_id
            WHERE m.modeling_id = CAST(:modeling_id AS uuid)
              AND m.{time_col} IS NOT NULL
            ORDER BY m.cluster_id, m.{time_col}, m.incident_number
        """)

    # ✅ FIX KRITIS: params harus via keyword `params=...`
    df = pd.read_sql(
        sql,
        engine,
        params={"modeling_id": modeling_id, "window_days": int(window_days)},
    )

    df.attrs["eligible_rule_effective"] = eligible_rule_effective

    if df.empty:
        return df

    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df = df.dropna(subset=["event_time"]).copy()
    return df


# =========================
# SESSIONIZATION
# =========================

def sessionize(df: pd.DataFrame, window_days: int, time_col: str) -> pd.DataFrame:
    if df.empty:
        return df

    # ✅ PATCH: sort eksplisit sebelum shift/gap
    out = df.sort_values(["modeling_id", "cluster_id", "event_time", "incident_number"]).reset_index(drop=True).copy()

    out["window_days"] = int(window_days)
    out["time_col"] = str(time_col)

    out["tgl"] = out["event_time"].dt.date
    out["prev_tgl"] = out.groupby(["modeling_id", "cluster_id"])["tgl"].shift(1)
    out["gap_days"] = (pd.to_datetime(out["tgl"]) - pd.to_datetime(out["prev_tgl"])).dt.days

    out["is_new_episode"] = np.where(
        out["prev_tgl"].isna(),
        0,
        (out["gap_days"] > int(window_days)).astype(int),
    )

    out["temporal_cluster_no"] = (
        out.groupby(["modeling_id", "cluster_id"])["is_new_episode"].cumsum() + 1
    ).astype(int)

    out["temporal_cluster_id"] = (
        out["modeling_id"].astype(str)
        + "-"
        + out["cluster_id"].astype(str)
        + "-"
        + out["temporal_cluster_no"].astype(str)
    )

    out["gap_days"] = out["gap_days"].astype("Int64")

    keep_cols = [
        "modeling_id", "window_days", "cluster_id",
        "incident_number", "time_col", "event_time",
        "site", "assignee", "modul", "sub_modul",
        "gap_days", "temporal_cluster_no", "temporal_cluster_id",
    ]
    return out[keep_cols].copy()


# =========================
# SUMMARY
# =========================

def compute_summary(
    df_sessionized: pd.DataFrame,
    modeling_id: str,
    window_days: int,
    time_col: str,
    include_noise: bool,
    eligible_rule_effective: str,
) -> pd.DataFrame:
    if df_sessionized.empty:
        return pd.DataFrame([{
            "modeling_id": modeling_id,
            "window_days": int(window_days),
            "time_col": str(time_col),
            "include_noise": bool(include_noise),
            "eligible_rule": str(eligible_rule_effective),
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
        "modeling_id": modeling_id,
        "window_days": int(window_days),
        "time_col": str(time_col),
        "include_noise": bool(include_noise),
        "eligible_rule": str(eligible_rule_effective),
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

    sql = text(f"""
        INSERT INTO {cfg.schema}.{cfg.out_summary}
        (modeling_id, window_days, time_col, include_noise, eligible_rule,
         n_clusters_eligible, n_clusters_split, prop_clusters_split,
         n_clusters_stable, prop_clusters_stable,
         total_episodes, avg_episode_per_cluster, median_episode_per_cluster)
        VALUES
        (CAST(:modeling_id AS uuid), :window_days, :time_col, :include_noise, :eligible_rule,
         :n_clusters_eligible, :n_clusters_split, :prop_clusters_split,
         :n_clusters_stable, :prop_clusters_stable,
         :total_episodes, :avg_episode_per_cluster, :median_episode_per_cluster)
        ON CONFLICT (modeling_id, window_days, time_col, include_noise) DO UPDATE SET
          eligible_rule = EXCLUDED.eligible_rule,
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

    modeling_id = str(df_out["modeling_id"].iloc[0])
    window_days = int(df_out["window_days"].iloc[0])
    time_col = str(df_out["time_col"].iloc[0])

    with engine.begin() as conn:
        conn.execute(
            text(f"""
                DELETE FROM {cfg.schema}.{cfg.out_members}
                WHERE modeling_id = CAST(:modeling_id AS uuid)
                  AND window_days = :window_days
                  AND time_col = :time_col
            """),
            {"modeling_id": modeling_id, "window_days": window_days, "time_col": time_col},
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

def run(
    modeling_id: str,
    windows: Iterable[int],
    time_col_preferred: str,
    include_noise: bool,
    eligible_rule: str,
) -> None:
    cfg = Config(windows=tuple(int(w) for w in windows))
    engine = get_engine(DBConfig())
    ensure_output_tables(engine, cfg)

    time_col = pick_time_column(engine, cfg, time_col_preferred)

    for w in cfg.windows:
        df = load_members_for_window(
            engine=engine,
            cfg=cfg,
            modeling_id=modeling_id,
            window_days=int(w),
            time_col=time_col,
            include_noise=include_noise,
            eligible_rule=eligible_rule,
        )
        eff_rule = df.attrs.get("eligible_rule_effective", eligible_rule)

        print(
            f"\n=== WINDOW {w} DAYS ===\n"
            f"[LOAD] rows={len(df):,} | time_col={time_col} | eligible_rule={eff_rule} | include_noise={include_noise}"
        )

        df_sess = sessionize(df, window_days=int(w), time_col=time_col)
        n_clusters = int(df_sess["cluster_id"].nunique()) if not df_sess.empty else 0
        print(f"[SESSIONIZE] rows_out={len(df_sess):,} | clusters={n_clusters:,}")

        # ✅ SANITY CHECK: apakah window berpengaruh?
        if not df_sess.empty:
            gd = df_sess["gap_days"].dropna()
            gd = gd[gd >= 0]  # aman
            if not gd.empty:
                print(
                    f"[GAP DIST] n_gap={len(gd):,} | "
                    f"<=7d={(gd<=7).mean()*100:.2f}% | "
                    f"<=14d={(gd<=14).mean()*100:.2f}% | "
                    f"<=30d={(gd<=30).mean()*100:.2f}% | "
                    f">30d={(gd>30).mean()*100:.2f}%"
                )

        save_sessionized(engine, cfg, df_sess)
        print(f"[SAVE] {cfg.schema}.{cfg.out_members} | window={w} | time_col={time_col}")

        summary = compute_summary(
            df_sessionized=df_sess,
            modeling_id=modeling_id,
            window_days=int(w),
            time_col=time_col,
            include_noise=include_noise,
            eligible_rule_effective=eff_rule,
        )
        upsert_summary(engine, cfg, summary)
        print(f"[SUMMARY] window={w} | {summary.to_dict(orient='records')[0]}")

    print("\n[DONE] semantic temporal evaluation completed for windows:", list(cfg.windows), "| time_col:", time_col)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modeling-id", required=True, help="modeling_id UUID (semantik HDBSCAN)")
    ap.add_argument("--windows", default="7,14,30", help="comma-separated windows (days), e.g. 7,14,30")
    ap.add_argument("--time-col", default="tgl_submit", help="kolom waktu pada members, default tgl_submit")
    ap.add_argument("--include-noise", action="store_true", help="ikutkan noise cluster (cluster_id=-1)")
    ap.add_argument(
        "--eligible-rule",
        default="span_days_strict",
        choices=["span_days_strict", "span_days_gt", "all"],
        help="aturan cluster eligible; jika span_days tidak ada, otomatis pakai computed span dari members",
    )
    args = ap.parse_args()

    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
    run(
        modeling_id=args.modeling_id,
        windows=windows,
        time_col_preferred=args.time_col,
        include_noise=bool(args.include_noise),
        eligible_rule=str(args.eligible_rule),
    )
