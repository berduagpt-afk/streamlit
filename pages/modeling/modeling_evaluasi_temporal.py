from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ======================================================
# 🔐 Guard Login
# ======================================================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login terlebih dahulu.")
    st.stop()

# Jangan set_page_config di sini kalau kamu sudah set di app.py
# st.set_page_config(layout="wide", page_title="Evaluasi Temporal — Sintaksis & Semantik")

SCHEMA = "lasis_djp"

T_SELECTED = "modeling_selected_representatives"

# SOURCES
T_SYN_CLUSTERS = "modeling_sintaksis_clusters"
T_SYN_MEMBERS = "modeling_sintaksis_members"

T_SEM_CLUSTERS = "modeling_semantik_hdbscan_clusters"
T_SEM_MEMBERS = "modeling_semantik_hdbscan_members"

# OUTPUT (existing, keep separate)
T_SYN_OUT_MEM = "modeling_sintaksis_temporal_members"
T_SYN_OUT_SUM = "modeling_sintaksis_temporal_summary"

T_SEM_OUT_MEM = "modeling_semantik_temporal_members"
T_SEM_OUT_SUM = "modeling_semantik_temporal_summary"


# ======================================================
# 🔌 DB Connection
# ======================================================
@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    cfg = st.secrets["connections"]["postgres"]
    return create_engine(
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}",
        pool_pre_ping=True,
    )


# ======================================================
# Shared helpers
# ======================================================
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


def read_selected_reps(engine: Engine) -> pd.DataFrame:
    q = text(f"""
        SELECT
            selected_id, selected_at,
            jenis_pendekatan, eval_id,
            job_id, modeling_id, embedding_run_id,
            temporal_id, threshold,
            source_page, notes
        FROM {SCHEMA}.{T_SELECTED}
        ORDER BY selected_at DESC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(q, conn)
    for c in ["selected_id", "jenis_pendekatan", "eval_id", "job_id", "modeling_id", "embedding_run_id", "temporal_id"]:
        if c in df.columns:
            df[c] = df[c].astype("string")
    if "selected_at" in df.columns:
        df["selected_at"] = pd.to_datetime(df["selected_at"], errors="coerce")
    return df


def parse_windows(s: str) -> list[int]:
    out = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            pass
    return sorted(list(set(out))) if out else [7, 14, 30]


# ======================================================
# ------------------ SINTAKSIS (adapted) ----------------
# Berdasar run_temporal_sintaksis_v2.py
# ======================================================
def ensure_output_tables_syn(engine: Engine) -> None:
    ddl = f"""
    CREATE SCHEMA IF NOT EXISTS {SCHEMA};

    CREATE TABLE IF NOT EXISTS {SCHEMA}.{T_SYN_OUT_MEM} (
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
        PRIMARY KEY (job_id, modeling_id, window_days, incident_number)
    );

    CREATE INDEX IF NOT EXISTS idx_{T_SYN_OUT_MEM}_job_model_win
      ON {SCHEMA}.{T_SYN_OUT_MEM} (job_id, modeling_id, window_days);

    CREATE INDEX IF NOT EXISTS idx_{T_SYN_OUT_MEM}_model_win_cluster
      ON {SCHEMA}.{T_SYN_OUT_MEM} (modeling_id, window_days, cluster_id);

    CREATE INDEX IF NOT EXISTS idx_{T_SYN_OUT_MEM}_episode
      ON {SCHEMA}.{T_SYN_OUT_MEM} (job_id, modeling_id, window_days, temporal_cluster_id);

    CREATE TABLE IF NOT EXISTS {SCHEMA}.{T_SYN_OUT_SUM} (
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
    statements = [s.strip() for s in ddl.split(";") if s.strip()]
    with engine.begin() as conn:
        for stmt in statements:
            conn.exec_driver_sql(stmt)


def load_members_syn(engine: Engine, job_id: str, modeling_id: str, window_days: int) -> pd.DataFrame:
    sql = text(f"""
        WITH eligible_clusters AS (
          SELECT c.job_id, c.modeling_id, c.cluster_id
          FROM {SCHEMA}.{T_SYN_CLUSTERS} c
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
        FROM {SCHEMA}.{T_SYN_MEMBERS} m
        JOIN eligible_clusters ec
          ON ec.job_id = m.job_id
         AND ec.modeling_id = m.modeling_id
         AND ec.cluster_id = m.cluster_id
        WHERE m.job_id = CAST(:job_id AS uuid)
          AND m.modeling_id = CAST(:modeling_id AS uuid)
          AND m.tgl_submit IS NOT NULL
        ORDER BY m.cluster_id, m.tgl_submit, m.incident_number
    """)
    df = pd.read_sql(sql, engine, params={"job_id": job_id, "modeling_id": modeling_id, "window_days": int(window_days)})
    if df.empty:
        return df
    df["tgl_submit"] = pd.to_datetime(df["tgl_submit"], errors="coerce")
    df = df.dropna(subset=["tgl_submit"]).copy()
    return df


def sessionize_syn(df: pd.DataFrame, window_days: int) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["window_days"] = int(window_days)
    out["tgl"] = out["tgl_submit"].dt.date
    out["prev_tgl"] = out.groupby(["job_id", "modeling_id", "cluster_id"])["tgl"].shift(1)
    out["gap_days"] = (pd.to_datetime(out["tgl"]) - pd.to_datetime(out["prev_tgl"])).dt.days
    out["is_new_episode"] = np.where(out["prev_tgl"].isna(), 0, (out["gap_days"] > int(window_days)).astype(int))
    out["temporal_cluster_no"] = (out.groupby(["job_id", "modeling_id", "cluster_id"])["is_new_episode"].cumsum() + 1).astype(int)
    out["temporal_cluster_id"] = (
        out["modeling_id"].astype(str) + "-" + out["cluster_id"].astype(str) + "-" + out["temporal_cluster_no"].astype(str)
    )
    out["gap_days"] = out["gap_days"].astype("Int64")
    keep_cols = [
        "job_id", "modeling_id", "window_days", "cluster_id",
        "incident_number", "tgl_submit", "site", "assignee", "modul", "sub_modul",
        "gap_days", "temporal_cluster_no", "temporal_cluster_id"
    ]
    return out[keep_cols].copy()


def summary_syn(df_sessionized: pd.DataFrame, job_id: str, modeling_id: str, window_days: int) -> pd.DataFrame:
    if df_sessionized.empty:
        return pd.DataFrame([{
            "job_id": job_id, "modeling_id": modeling_id, "window_days": int(window_days),
            "n_clusters_eligible": 0, "n_clusters_split": 0, "prop_clusters_split": 0.0,
            "n_clusters_stable": 0, "prop_clusters_stable": 0.0,
            "total_episodes": 0, "avg_episode_per_cluster": 0.0, "median_episode_per_cluster": 0.0
        }])
    g = df_sessionized.groupby(["cluster_id"], as_index=False).agg(n_episode=("temporal_cluster_no", "nunique"))
    n_clusters = int(len(g))
    n_split = int((g["n_episode"] > 1).sum())
    n_stable = int((g["n_episode"] == 1).sum())
    total_episodes = int(g["n_episode"].sum())
    avg_ep = float(g["n_episode"].mean())
    med_ep = float(g["n_episode"].median())
    return pd.DataFrame([{
        "job_id": job_id, "modeling_id": modeling_id, "window_days": int(window_days),
        "n_clusters_eligible": n_clusters,
        "n_clusters_split": n_split,
        "prop_clusters_split": float(n_split / n_clusters) if n_clusters else 0.0,
        "n_clusters_stable": n_stable,
        "prop_clusters_stable": float(n_stable / n_clusters) if n_clusters else 0.0,
        "total_episodes": total_episodes,
        "avg_episode_per_cluster": avg_ep,
        "median_episode_per_cluster": med_ep,
    }])


def save_syn(engine: Engine, df_out: pd.DataFrame, sum_df: pd.DataFrame) -> None:
    if df_out.empty and sum_df.empty:
        return
    with engine.begin() as conn:
        if not df_out.empty:
            job_id = str(df_out["job_id"].iloc[0])
            modeling_id = str(df_out["modeling_id"].iloc[0])
            window_days = int(df_out["window_days"].iloc[0])

            conn.execute(
                text(f"""
                    DELETE FROM {SCHEMA}.{T_SYN_OUT_MEM}
                    WHERE job_id = CAST(:job_id AS uuid)
                      AND modeling_id = CAST(:modeling_id AS uuid)
                      AND window_days = :window_days
                """),
                {"job_id": job_id, "modeling_id": modeling_id, "window_days": window_days},
            )
            df_out.to_sql(T_SYN_OUT_MEM, con=conn, schema=SCHEMA, if_exists="append", index=False, method="multi", chunksize=5000)

        if not sum_df.empty:
            row = sum_df.iloc[0].to_dict()
            conn.execute(
                text(f"""
                    INSERT INTO {SCHEMA}.{T_SYN_OUT_SUM}
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
                """),
                row,
            )


def run_temporal_syn(engine: Engine, job_id: str, modeling_id: str, windows: list[int]) -> pd.DataFrame:
    ensure_output_tables_syn(engine)
    rows = []
    for w in windows:
        df = load_members_syn(engine, job_id, modeling_id, int(w))
        df_sess = sessionize_syn(df, int(w))
        sum_df = summary_syn(df_sess, job_id, modeling_id, int(w))
        save_syn(engine, df_sess, sum_df)
        rows.append(sum_df.iloc[0].to_dict())
    return pd.DataFrame(rows)


# ======================================================
# ------------------ SEMANTIK (adapted) -----------------
# Berdasar run_temporal_semantik.py
# ======================================================
def ensure_output_tables_sem(engine: Engine) -> None:
    ddl = f"""
    CREATE SCHEMA IF NOT EXISTS {SCHEMA};

    CREATE TABLE IF NOT EXISTS {SCHEMA}.{T_SEM_OUT_MEM} (
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

    CREATE INDEX IF NOT EXISTS idx_{T_SEM_OUT_MEM}_model_win
      ON {SCHEMA}.{T_SEM_OUT_MEM} (modeling_id, window_days);

    CREATE INDEX IF NOT EXISTS idx_{T_SEM_OUT_MEM}_model_win_cluster
      ON {SCHEMA}.{T_SEM_OUT_MEM} (modeling_id, window_days, cluster_id);

    CREATE INDEX IF NOT EXISTS idx_{T_SEM_OUT_MEM}_episode
      ON {SCHEMA}.{T_SEM_OUT_MEM} (modeling_id, window_days, temporal_cluster_id);

    CREATE TABLE IF NOT EXISTS {SCHEMA}.{T_SEM_OUT_SUM} (
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


def pick_time_column_sem(engine: Engine, preferred: str) -> str:
    candidates = [preferred, "tgl_submit", "tgl_semantik"]
    for c in candidates:
        if c and table_has_column(engine, SCHEMA, T_SEM_MEMBERS, c):
            return c
    raise RuntimeError(
        f"Tidak menemukan kolom waktu pada {SCHEMA}.{T_SEM_MEMBERS}. "
        f"Coba pastikan ada salah satu kolom: {candidates}"
    )


def select_optional_text_cols_sem(engine: Engine, cols: list[str], alias_prefix: str = "m") -> str:
    parts: list[str] = []
    for c in cols:
        if table_has_column(engine, SCHEMA, T_SEM_MEMBERS, c):
            parts.append(f"{alias_prefix}.{c} AS {c}")
        else:
            parts.append(f"NULL::text AS {c}")
    return ",\n          ".join(parts)


def build_eligible_filter_clusters(rule: str, include_noise: bool) -> tuple[str, str]:
    noise_clause = "" if include_noise else "AND c.cluster_id <> -1"
    if rule == "all":
        return f"TRUE {noise_clause}", "all"
    if rule == "span_days_gt":
        return f"(COALESCE(c.span_days,0) > :window_days) {noise_clause}", "span_days_gt"
    return f"((COALESCE(c.span_days,0) - :window_days) > 1) {noise_clause}", "span_days_strict"


def build_eligible_filter_members_span(rule: str, include_noise: bool) -> tuple[str, str]:
    noise_clause = "" if include_noise else "AND s.cluster_id <> -1"
    if rule == "all":
        return f"TRUE {noise_clause}", "all (computed span from members)"
    if rule == "span_days_gt":
        return f"(s.span_days > :window_days) {noise_clause}", "span_days_gt (computed span from members)"
    return f"((s.span_days - :window_days) > 1) {noise_clause}", "span_days_strict (computed span from members)"


def load_members_sem(
    engine: Engine,
    modeling_id: str,
    window_days: int,
    time_col: str,
    include_noise: bool,
    eligible_rule: str,
) -> pd.DataFrame:
    has_span = table_has_column(engine, SCHEMA, T_SEM_CLUSTERS, "span_days")
    optional_cols_sql = select_optional_text_cols_sem(engine, ["site", "assignee", "modul", "sub_modul"], "m")

    if has_span:
        eligible_filter, eff_rule = build_eligible_filter_clusters(eligible_rule, include_noise)
        sql = text(f"""
            WITH eligible_clusters AS (
              SELECT c.modeling_id, c.cluster_id
              FROM {SCHEMA}.{T_SEM_CLUSTERS} c
              WHERE c.modeling_id = CAST(:modeling_id AS uuid)
                AND {eligible_filter}
            )
            SELECT
              m.modeling_id,
              m.cluster_id,
              m.incident_number,
              m.{time_col} AS event_time,
              {optional_cols_sql}
            FROM {SCHEMA}.{T_SEM_MEMBERS} m
            JOIN eligible_clusters ec
              ON ec.modeling_id = m.modeling_id
             AND ec.cluster_id  = m.cluster_id
            WHERE m.modeling_id = CAST(:modeling_id AS uuid)
              AND m.{time_col} IS NOT NULL
            ORDER BY m.cluster_id, m.{time_col}, m.incident_number
        """)
    else:
        eligible_filter, eff_rule = build_eligible_filter_members_span(eligible_rule, include_noise)
        sql = text(f"""
            WITH span_by_cluster AS (
              SELECT
                m.modeling_id,
                m.cluster_id,
                (MAX(m.{time_col}::date) - MIN(m.{time_col}::date))::int AS span_days
              FROM {SCHEMA}.{T_SEM_MEMBERS} m
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
            FROM {SCHEMA}.{T_SEM_MEMBERS} m
            JOIN eligible_clusters ec
              ON ec.modeling_id = m.modeling_id
             AND ec.cluster_id  = m.cluster_id
            WHERE m.modeling_id = CAST(:modeling_id AS uuid)
              AND m.{time_col} IS NOT NULL
            ORDER BY m.cluster_id, m.{time_col}, m.incident_number
        """)

    df = pd.read_sql(sql, engine, params={"modeling_id": modeling_id, "window_days": int(window_days)})
    df.attrs["eligible_rule_effective"] = eff_rule
    if df.empty:
        return df
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df = df.dropna(subset=["event_time"]).copy()
    return df


def sessionize_sem(df: pd.DataFrame, window_days: int, time_col: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["window_days"] = int(window_days)
    out["time_col"] = str(time_col)
    out["tgl"] = out["event_time"].dt.date
    out["prev_tgl"] = out.groupby(["modeling_id", "cluster_id"])["tgl"].shift(1)
    out["gap_days"] = (pd.to_datetime(out["tgl"]) - pd.to_datetime(out["prev_tgl"])).dt.days
    out["is_new_episode"] = np.where(out["prev_tgl"].isna(), 0, (out["gap_days"] > int(window_days)).astype(int))
    out["temporal_cluster_no"] = (out.groupby(["modeling_id", "cluster_id"])["is_new_episode"].cumsum() + 1).astype(int)
    out["temporal_cluster_id"] = (
        out["modeling_id"].astype(str) + "-" + out["cluster_id"].astype(str) + "-" + out["temporal_cluster_no"].astype(str)
    )
    out["gap_days"] = out["gap_days"].astype("Int64")

    keep_cols = [
        "modeling_id", "window_days", "cluster_id",
        "incident_number", "time_col", "event_time",
        "site", "assignee", "modul", "sub_modul",
        "gap_days", "temporal_cluster_no", "temporal_cluster_id",
    ]
    return out[keep_cols].copy()


def summary_sem(
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

    g = df_sessionized.groupby(["cluster_id"], as_index=False).agg(n_episode=("temporal_cluster_no", "nunique"))
    n_clusters = int(len(g))
    n_split = int((g["n_episode"] > 1).sum())
    n_stable = int((g["n_episode"] == 1).sum())
    total_episodes = int(g["n_episode"].sum())
    return pd.DataFrame([{
        "modeling_id": modeling_id,
        "window_days": int(window_days),
        "time_col": str(time_col),
        "include_noise": bool(include_noise),
        "eligible_rule": str(eligible_rule_effective),
        "n_clusters_eligible": n_clusters,
        "n_clusters_split": n_split,
        "prop_clusters_split": float(n_split / n_clusters) if n_clusters else 0.0,
        "n_clusters_stable": n_stable,
        "prop_clusters_stable": float(n_stable / n_clusters) if n_clusters else 0.0,
        "total_episodes": total_episodes,
        "avg_episode_per_cluster": float(g["n_episode"].mean()) if n_clusters else 0.0,
        "median_episode_per_cluster": float(g["n_episode"].median()) if n_clusters else 0.0,
    }])


def save_sem(engine: Engine, df_out: pd.DataFrame, sum_df: pd.DataFrame) -> None:
    if df_out.empty and sum_df.empty:
        return
    with engine.begin() as conn:
        if not df_out.empty:
            modeling_id = str(df_out["modeling_id"].iloc[0])
            window_days = int(df_out["window_days"].iloc[0])
            time_col = str(df_out["time_col"].iloc[0])

            conn.execute(
                text(f"""
                    DELETE FROM {SCHEMA}.{T_SEM_OUT_MEM}
                    WHERE modeling_id = CAST(:modeling_id AS uuid)
                      AND window_days = :window_days
                      AND time_col = :time_col
                """),
                {"modeling_id": modeling_id, "window_days": window_days, "time_col": time_col},
            )
            df_out.to_sql(T_SEM_OUT_MEM, con=conn, schema=SCHEMA, if_exists="append", index=False, method="multi", chunksize=5000)

        if not sum_df.empty:
            row = sum_df.iloc[0].to_dict()
            conn.execute(
                text(f"""
                    INSERT INTO {SCHEMA}.{T_SEM_OUT_SUM}
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
                """),
                row,
            )


def run_temporal_sem(
    engine: Engine,
    modeling_id: str,
    windows: list[int],
    time_col_preferred: str,
    include_noise: bool,
    eligible_rule: str,
) -> pd.DataFrame:
    ensure_output_tables_sem(engine)
    time_col = pick_time_column_sem(engine, time_col_preferred)

    rows = []
    for w in windows:
        df = load_members_sem(engine, modeling_id, int(w), time_col, include_noise, eligible_rule)
        eff_rule = df.attrs.get("eligible_rule_effective", eligible_rule)

        df_sess = sessionize_sem(df, int(w), time_col)
        sum_df = summary_sem(df_sess, modeling_id, int(w), time_col, include_noise, eff_rule)

        save_sem(engine, df_sess, sum_df)
        rows.append(sum_df.iloc[0].to_dict())

    return pd.DataFrame(rows)


# ======================================================
# UI
# ======================================================
st.title("🕒 Evaluasi Temporal (Sintaksis + Semantik) — Runner dari Selected Representatives")
st.caption(
    f"Sumber pilihan: {SCHEMA}.{T_SELECTED}. "
    f"Output disimpan terpisah untuk sintaksis dan semantik (tetap kompatibel dengan pipeline yang sudah ada), "
    f"namun hasil summary ditampilkan gabungan di dashboard ini."
)

engine = get_engine()
df_sel = read_selected_reps(engine)

if df_sel.empty:
    st.warning(f"Tabel {SCHEMA}.{T_SELECTED} masih kosong. Simpan perwakilan dulu dari halaman compare.")
    st.stop()

# split
df_sel_syn = df_sel[df_sel["jenis_pendekatan"] == "sintaksis"].copy()
df_sel_sem = df_sel[df_sel["jenis_pendekatan"] == "semantik"].copy()

# --- selectors
st.subheader("1) Pilih Representative Run")
c1, c2 = st.columns(2)

def _label_row(r: pd.Series) -> str:
    ts = r.get("selected_at")
    ts_s = ts.strftime("%Y-%m-%d %H:%M") if pd.notna(ts) else "?"
    return f"{ts_s} | eval_id={r.get('eval_id')} | modeling_id={r.get('modeling_id')}"

with c1:
    st.markdown("**Sintaksis** (butuh job_id + modeling_id)")
    if df_sel_syn.empty:
        st.info("Belum ada selected representative untuk sintaksis.")
        syn_idx = None
    else:
        syn_options = df_sel_syn.reset_index(drop=True)
        syn_choice = st.selectbox(
            "Pilih run sintaksis",
            options=list(range(len(syn_options))),
            format_func=lambda i: _label_row(syn_options.loc[i]),
        )
        syn_idx = int(syn_choice)

with c2:
    st.markdown("**Semantik** (filter utama embedding_run_id, temporal pakai modeling_id HDBSCAN)")
    if df_sel_sem.empty:
        st.info("Belum ada selected representative untuk semantik.")
        sem_idx = None
    else:
        sem_options = df_sel_sem.reset_index(drop=True)
        sem_choice = st.selectbox(
            "Pilih run semantik",
            options=list(range(len(sem_options))),
            format_func=lambda i: _label_row(sem_options.loc[i]) + f" | embedding_run_id={sem_options.loc[i].get('embedding_run_id')}",
        )
        sem_idx = int(sem_choice)

st.divider()

# --- parameters
st.subheader("2) Parameter Evaluasi Temporal")
c3, c4, c5 = st.columns([1.1, 1, 1])

with c3:
    windows_str = st.text_input("Windows (hari) comma-separated", value="7,14,30")
    windows = parse_windows(windows_str)

with c4:
    # semantik only
    time_col_pref = st.text_input("Semantik: time_col preferred", value="tgl_submit")
    eligible_rule = st.selectbox("Semantik: eligible_rule", ["span_days_strict", "span_days_gt", "all"], index=0)

with c5:
    include_noise = st.checkbox("Semantik: include_noise (cluster_id=-1)", value=False)

st.caption(f"Windows terpakai: {windows}")

st.divider()

# --- run buttons
st.subheader("3) Jalankan Evaluasi Temporal")

run_syn = st.button("▶️ Run Sintaksis", use_container_width=True, disabled=(syn_idx is None))
run_sem = st.button("▶️ Run Semantik", use_container_width=True, disabled=(sem_idx is None))

syn_result = None
sem_result = None

# Execute (idempotent on each window scope)
if run_syn and syn_idx is not None:
    row = df_sel_syn.reset_index(drop=True).loc[syn_idx]
    job_id = str(row.get("job_id") or "")
    modeling_id = str(row.get("modeling_id") or "")
    if not job_id or job_id.lower() == "na" or not modeling_id or modeling_id.lower() == "na":
        st.error("Representative sintaksis tidak valid: job_id/modeling_id kosong.")
    else:
        with st.spinner("Running sintaksis temporal..."):
            syn_result = run_temporal_syn(engine, job_id, modeling_id, windows)
        st.success("Selesai menjalankan sintaksis temporal.")

if run_sem and sem_idx is not None:
    row = df_sel_sem.reset_index(drop=True).loc[sem_idx]
    modeling_id = str(row.get("modeling_id") or "")
    if not modeling_id or modeling_id.lower() == "na":
        st.error("Representative semantik tidak valid: modeling_id kosong.")
    else:
        with st.spinner("Running semantik temporal..."):
            sem_result = run_temporal_sem(engine, modeling_id, windows, time_col_pref, include_noise, eligible_rule)
        st.success("Selesai menjalankan semantik temporal.")

st.divider()

# ======================================================
# 4) Tampilkan hasil summary (gabungan)
# ======================================================
st.subheader("4) Hasil Summary (gabungan)")

def fetch_syn_summary(engine: Engine, job_id: str, modeling_id: str, windows: list[int]) -> pd.DataFrame:
    q = text(f"""
        SELECT
            'sintaksis'::text AS jenis_pendekatan,
            job_id::text, modeling_id::text,
            NULL::text AS embedding_run_id,
            NULL::text AS time_col,
            NULL::boolean AS include_noise,
            NULL::text AS eligible_rule,
            window_days,
            n_clusters_eligible, n_clusters_split, prop_clusters_split,
            n_clusters_stable, prop_clusters_stable,
            total_episodes, avg_episode_per_cluster, median_episode_per_cluster,
            run_time
        FROM {SCHEMA}.{T_SYN_OUT_SUM}
        WHERE job_id = CAST(:job_id AS uuid)
          AND modeling_id = CAST(:modeling_id AS uuid)
          AND window_days = ANY(:windows)
        ORDER BY window_days
    """)
    with engine.connect() as conn:
        return pd.read_sql(q, conn, params={"job_id": job_id, "modeling_id": modeling_id, "windows": windows})


def fetch_sem_summary(engine: Engine, modeling_id: str, windows: list[int], time_col: str, include_noise: bool) -> pd.DataFrame:
    q = text(f"""
        SELECT
            'semantik'::text AS jenis_pendekatan,
            NULL::text AS job_id,
            modeling_id::text,
            NULL::text AS embedding_run_id,
            time_col,
            include_noise,
            eligible_rule,
            window_days,
            n_clusters_eligible, n_clusters_split, prop_clusters_split,
            n_clusters_stable, prop_clusters_stable,
            total_episodes, avg_episode_per_cluster, median_episode_per_cluster,
            run_time
        FROM {SCHEMA}.{T_SEM_OUT_SUM}
        WHERE modeling_id = CAST(:modeling_id AS uuid)
          AND window_days = ANY(:windows)
          AND time_col = :time_col
          AND include_noise = :include_noise
        ORDER BY window_days
    """)
    with engine.connect() as conn:
        return pd.read_sql(q, conn, params={
            "modeling_id": modeling_id,
            "windows": windows,
            "time_col": time_col,
            "include_noise": include_noise,
        })


summary_parts = []

# auto fetch based on selected reps (even if user didn't press run, still show if already exists)
if syn_idx is not None:
    r = df_sel_syn.reset_index(drop=True).loc[syn_idx]
    job_id = str(r.get("job_id") or "")
    modeling_id = str(r.get("modeling_id") or "")
    if job_id and modeling_id and job_id.lower() != "na" and modeling_id.lower() != "na":
        try:
            summary_parts.append(fetch_syn_summary(engine, job_id, modeling_id, windows))
        except Exception:
            pass

if sem_idx is not None:
    r = df_sel_sem.reset_index(drop=True).loc[sem_idx]
    modeling_id = str(r.get("modeling_id") or "")
    if modeling_id and modeling_id.lower() != "na":
        try:
            time_col_effective = pick_time_column_sem(engine, time_col_pref)
            summary_parts.append(fetch_sem_summary(engine, modeling_id, windows, time_col_effective, include_noise))
        except Exception:
            pass

summary_all = pd.concat([x for x in summary_parts if x is not None and not x.empty], ignore_index=True) if summary_parts else pd.DataFrame()

if summary_all.empty:
    st.info("Belum ada summary untuk kombinasi input/parameter ini. Jalankan tombol Run dulu.")
else:
    st.dataframe(summary_all, use_container_width=True, height=240)

    # chart: prop_clusters_split per window
    p = summary_all.copy()
    p["window_days"] = pd.to_numeric(p["window_days"], errors="coerce")
    p["prop_clusters_split"] = pd.to_numeric(p["prop_clusters_split"], errors="coerce")
    p = p.dropna(subset=["window_days", "prop_clusters_split"])

    if not p.empty:
        chart = alt.Chart(p).mark_line(point=True).encode(
            x=alt.X("window_days:Q", title="Window (days)"),
            y=alt.Y("prop_clusters_split:Q", title="Proporsi cluster split (↑ lebih banyak episode)"),
            color=alt.Color("jenis_pendekatan:N", title="Pendekatan"),
            tooltip=["jenis_pendekatan", "window_days", "prop_clusters_split", "n_clusters_eligible", "total_episodes"],
        )
        st.altair_chart(chart, use_container_width=True)

    st.download_button(
        "⬇️ Download summary gabungan (CSV)",
        data=summary_all.to_csv(index=False).encode("utf-8"),
        file_name="temporal_summary_gabungan.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.divider()

# ======================================================
# 5) Detail members (pilih pendekatan)
# ======================================================
st.subheader("5) Detail Members (hasil sessionization)")

mode = st.radio("Tampilkan detail:", ["Sintaksis", "Semantik"], horizontal=True)

def fetch_syn_members(engine: Engine, job_id: str, modeling_id: str, window_days: int) -> pd.DataFrame:
    q = text(f"""
        SELECT *
        FROM {SCHEMA}.{T_SYN_OUT_MEM}
        WHERE job_id = CAST(:job_id AS uuid)
          AND modeling_id = CAST(:modeling_id AS uuid)
          AND window_days = :w
        ORDER BY cluster_id, tgl_submit, incident_number
        LIMIT 5000
    """)
    with engine.connect() as conn:
        return pd.read_sql(q, conn, params={"job_id": job_id, "modeling_id": modeling_id, "w": int(window_days)})


def fetch_sem_members(engine: Engine, modeling_id: str, window_days: int, time_col: str) -> pd.DataFrame:
    q = text(f"""
        SELECT *
        FROM {SCHEMA}.{T_SEM_OUT_MEM}
        WHERE modeling_id = CAST(:modeling_id AS uuid)
          AND window_days = :w
          AND time_col = :time_col
        ORDER BY cluster_id, event_time, incident_number
        LIMIT 5000
    """)
    with engine.connect() as conn:
        return pd.read_sql(q, conn, params={"modeling_id": modeling_id, "w": int(window_days), "time_col": time_col})


win_choice = st.selectbox("Window untuk detail", options=windows, index=0)

if mode == "Sintaksis":
    if syn_idx is None:
        st.info("Pilih representative sintaksis dulu.")
    else:
        r = df_sel_syn.reset_index(drop=True).loc[syn_idx]
        job_id = str(r.get("job_id") or "")
        modeling_id = str(r.get("modeling_id") or "")
        if not job_id or not modeling_id or job_id.lower() == "na" or modeling_id.lower() == "na":
            st.warning("job_id/modeling_id sintaksis tidak valid.")
        else:
            dfm = fetch_syn_members(engine, job_id, modeling_id, win_choice)
            st.dataframe(dfm, use_container_width=True, height=420)
            st.download_button(
                "⬇️ Download detail sintaksis (CSV, limit 5000)",
                data=dfm.to_csv(index=False).encode("utf-8"),
                file_name=f"temporal_members_sintaksis_w{win_choice}.csv",
                mime="text/csv",
                use_container_width=True,
            )

else:
    if sem_idx is None:
        st.info("Pilih representative semantik dulu.")
    else:
        r = df_sel_sem.reset_index(drop=True).loc[sem_idx]
        modeling_id = str(r.get("modeling_id") or "")
        if not modeling_id or modeling_id.lower() == "na":
            st.warning("modeling_id semantik tidak valid.")
        else:
            time_col_effective = pick_time_column_sem(engine, time_col_pref)
            dfm = fetch_sem_members(engine, modeling_id, win_choice, time_col_effective)
            st.dataframe(dfm, use_container_width=True, height=420)
            st.download_button(
                "⬇️ Download detail semantik (CSV, limit 5000)",
                data=dfm.to_csv(index=False).encode("utf-8"),
                file_name=f"temporal_members_semantik_w{win_choice}_{time_col_effective}.csv",
                mime="text/csv",
                use_container_width=True,
            )
