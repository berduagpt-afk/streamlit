# pages/labeling/labeling_temporal_rule.py
# ============================================================
# 🏷️ Pelabelan Insiden Berulang (Rule-based Temporal)
# - Mendukung Sintaksis & Semantik
# - Basis: temporal_members (hasil evaluasi temporal / sessionization)
# - Output unified: lasis_djp.incident_labeling_results
#
# FINAL (v2) - +Ringkasan Episode Berulang (episode berisi >1 tiket)
# 1) Label tidak hanya berbasis cluster, tapi juga episode:
#    label_berulang = 1 jika:
#      - n_member_cluster >= N_min_cluster
#      - n_episode_cluster >= E_min_cluster
#      - n_member_episode >= N_min_episode  (mencegah episode singleton dianggap berulang)
#
# 2) Noise semantik dikendalikan sesuai include_noise:
#    - include_noise=False → cluster_id=-1 dikecualikan pada stats & rows
#
# 3) Self-healing schema:
#    - ADD COLUMN IF NOT EXISTS n_member_episode (untuk tabel lama)
#
# 4) Tambahan analisis:
#    - jumlah_episode_berulang = jumlah episode (temporal_cluster_no) dengan >1 tiket pada cluster_id
#    - jumlah_tiket_episode_berulang = total tiket dari episode berulang saja
#
# Catatan teknis:
# - time_col NOT NULL DEFAULT '' untuk PK/ON CONFLICT
# - Hindari :param::uuid → pakai CAST(:param AS uuid)
# ============================================================

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

# ======================================================
# 🔐 Guard Login
# ======================================================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login terlebih dahulu.")
    st.stop()

SCHEMA = "lasis_djp"

# --- Temporal tables ---
T_SYN_SUM = f"{SCHEMA}.modeling_sintaksis_temporal_summary"
T_SYN_MEM = f"{SCHEMA}.modeling_sintaksis_temporal_members"

T_SEM_SUM = f"{SCHEMA}.modeling_semantik_temporal_summary"
T_SEM_MEM = f"{SCHEMA}.modeling_semantik_temporal_members"

# --- Output labeling table (unified) ---
T_LABEL = f"{SCHEMA}.incident_labeling_results"


# ======================================================
# 🔌 DB Connection
# ======================================================
@st.cache_resource(show_spinner=False)
def get_engine():
    cfg = st.secrets["connections"]["postgres"]
    return create_engine(
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}",
        pool_pre_ping=True,
    )


def read_df(engine, sql: str, params: Optional[dict] = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def exec_sql(engine, sql: str, params: Optional[dict] = None) -> None:
    with engine.begin() as conn:
        conn.execute(text(sql), params or {})


def ensure_output_table(engine) -> None:
    """
    Create table if not exists + self-healing migration for older table versions.
    """
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {T_LABEL} (
        jenis_pendekatan text NOT NULL,
        run_time timestamptz NOT NULL DEFAULT now(),

        -- identity of the run
        job_id uuid NULL,
        modeling_id uuid NOT NULL,
        window_days integer NOT NULL,

        -- IMPORTANT: non-null for PK/ON CONFLICT
        time_col text NOT NULL DEFAULT '',
        include_noise boolean NULL,
        eligible_rule text NULL,

        -- incident identity
        incident_number text NOT NULL,

        -- time fields
        event_time timestamp NULL,
        tgl_submit timestamp NULL,

        -- membership (from temporal_members)
        cluster_id bigint NOT NULL,
        temporal_cluster_no integer NOT NULL,
        temporal_cluster_id text NOT NULL,
        gap_days integer NULL,

        site text NULL,
        assignee text NULL,
        modul text NULL,
        sub_modul text NULL,

        -- aggregated stats
        n_member_cluster bigint NOT NULL,
        n_episode_cluster bigint NOT NULL,
        min_time timestamp NULL,
        max_time timestamp NULL,

        -- label
        label_berulang integer NOT NULL,
        rule_json jsonb NOT NULL,

        CONSTRAINT incident_labeling_results_pkey
            PRIMARY KEY (jenis_pendekatan, modeling_id, window_days, incident_number, time_col)
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))

        # ✅ self-healing migration: add episode-size column (for old tables)
        conn.execute(text(f"ALTER TABLE {T_LABEL} ADD COLUMN IF NOT EXISTS n_member_episode bigint;"))
        # NOTE: update massal bisa berat kalau tabel besar; tapi tetap dipertahankan sesuai versi kamu.
        conn.execute(text(f"UPDATE {T_LABEL} SET n_member_episode = 1 WHERE n_member_episode IS NULL;"))
        conn.execute(text(f"ALTER TABLE {T_LABEL} ALTER COLUMN n_member_episode SET DEFAULT 1;"))
        conn.execute(text(f"ALTER TABLE {T_LABEL} ALTER COLUMN n_member_episode SET NOT NULL;"))

        # indexes
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_labeling_modeling ON {T_LABEL} (modeling_id, window_days);"))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_labeling_label ON {T_LABEL} (label_berulang);"))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_labeling_incident ON {T_LABEL} (incident_number);"))
        conn.execute(
            text(
                f"CREATE INDEX IF NOT EXISTS idx_labeling_run "
                f"ON {T_LABEL} (jenis_pendekatan, modeling_id, window_days, time_col);"
            )
        )


# ======================================================
# 🧭 UI
# ======================================================
st.title("🏷️ Pelabelan Insiden Berulang (Rule-based Temporal)")
st.caption(
    "Memberi label berulang/tidak berulang menggunakan episode temporal dari temporal_members "
    "untuk pendekatan sintaksis maupun semantik."
)

engine = get_engine()
ensure_output_table(engine)

# --- Run selection state ---
job_id: Optional[str] = None
modeling_id: Optional[str] = None
window_days: Optional[int] = None
time_col: str = ""
include_noise: Optional[bool] = None
eligible_rule: Optional[str] = None

colA, colB, colC, colD = st.columns([1.2, 1, 1, 1])
with colA:
    pendekatan = st.selectbox("Jenis pendekatan", ["sintaksis", "semantik"], index=0)
with colB:
    min_member_cluster = st.number_input("Min tiket per cluster (N_min_cluster)", min_value=2, value=100, step=1)
with colC:
    min_episode_cluster = st.number_input("Min episode per cluster (E_min_cluster)", min_value=1, value=2, step=1)
with colD:
    min_member_episode = st.number_input("Min tiket per episode (N_min_episode)", min_value=1, value=2, step=1)

st.markdown("---")

# ======================================================
# 🔎 Load available runs (from temporal_summary)
# ======================================================
if pendekatan == "sintaksis":
    runs = read_df(
        engine,
        f"""
        SELECT job_id, modeling_id, window_days, run_time
        FROM {T_SYN_SUM}
        ORDER BY run_time DESC
        LIMIT 300
        """,
    )
    if runs.empty:
        st.warning("Tidak ada data di modeling_sintaksis_temporal_summary.")
        st.stop()

    run_label = runs.apply(
        lambda r: f"job={r['job_id']} | modeling={r['modeling_id']} | w={r['window_days']} | {r['run_time']}",
        axis=1,
    )
    idx = st.selectbox(
        "Pilih run (sintaksis)",
        options=list(range(len(runs))),
        format_func=lambda i: run_label.iloc[i],
    )

    job_id = str(runs.loc[idx, "job_id"])
    modeling_id = str(runs.loc[idx, "modeling_id"])
    window_days = int(runs.loc[idx, "window_days"])

    time_col = ""  # sintaksis default
    include_noise = None
    eligible_rule = None

    st.info(f"Run terpilih: job_id={job_id} | modeling_id={modeling_id} | window_days={window_days}")

else:
    runs = read_df(
        engine,
        f"""
        SELECT modeling_id, window_days, time_col, include_noise, eligible_rule, run_time
        FROM {T_SEM_SUM}
        ORDER BY run_time DESC
        LIMIT 300
        """,
    )
    if runs.empty:
        st.warning("Tidak ada data di modeling_semantik_temporal_summary.")
        st.stop()

    run_label = runs.apply(
        lambda r: (
            f"modeling={r['modeling_id']} | w={r['window_days']} | time_col={r['time_col']} | "
            f"include_noise={r['include_noise']} | {r['run_time']}"
        ),
        axis=1,
    )
    idx = st.selectbox(
        "Pilih run (semantik)",
        options=list(range(len(runs))),
        format_func=lambda i: run_label.iloc[i],
    )

    modeling_id = str(runs.loc[idx, "modeling_id"])
    window_days = int(runs.loc[idx, "window_days"])
    time_col = str(runs.loc[idx, "time_col"])
    include_noise = bool(runs.loc[idx, "include_noise"])
    eligible_rule = str(runs.loc[idx, "eligible_rule"])

    job_id = None

    st.info(
        f"Run terpilih: modeling_id={modeling_id} | window_days={window_days} | time_col={time_col} | include_noise={include_noise}"
    )
    st.caption(f"eligible_rule: {eligible_rule}")

st.markdown("---")

# ======================================================
# 🧠 Rule JSON (audit trail)
# ======================================================
rule_json: Dict[str, Any] = {
    "min_member_cluster": int(min_member_cluster),
    "min_episode_cluster": int(min_episode_cluster),
    "min_member_episode": int(min_member_episode),
    "basis": "temporal_members",
    "pendekatan": pendekatan,
    "window_days": int(window_days or 0),
    "time_col": time_col,
    "include_noise": include_noise if pendekatan == "semantik" else None,
    "eligible_rule": eligible_rule if pendekatan == "semantik" else None,
    "created_at": datetime.utcnow().isoformat() + "Z",
}

st.subheader("Aturan Pelabelan")
st.write(
    {
        "label_berulang = 1 jika": (
            f"n_member_cluster ≥ {min_member_cluster} "
            f"DAN n_episode_cluster ≥ {min_episode_cluster} "
            f"DAN n_member_episode ≥ {min_member_episode}"
        ),
        "label_berulang = 0 jika": "tidak memenuhi salah satu syarat di atas",
    }
)

col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
with col1:
    preview_limit = st.number_input("Preview rows", min_value=50, value=200, step=50)
with col2:
    do_replace = st.checkbox("Replace (hapus dulu run ini, lalu tulis ulang)", value=False)
with col3:
    do_write = st.checkbox("Tulis ke database", value=True)
with col4:
    topk_cluster = st.number_input("Top-K cluster (ringkasan)", min_value=10, value=30, step=10)

st.markdown("---")


# ======================================================
# 🩺 Diagnostics
# ======================================================
def diagnostics_sintaksis() -> pd.DataFrame:
    sql = f"""
    SELECT
      COUNT(*)::bigint AS n_rows,
      COUNT(DISTINCT incident_number)::bigint AS n_ticket,
      COUNT(DISTINCT cluster_id)::bigint AS n_cluster,
      SUM(CASE WHEN tgl_submit IS NULL THEN 1 ELSE 0 END)::bigint AS n_null_time,
      SUM(CASE WHEN temporal_cluster_no IS NULL THEN 1 ELSE 0 END)::bigint AS n_null_episode
    FROM {T_SYN_MEM}
    WHERE job_id = CAST(:job_id AS uuid)
      AND modeling_id = CAST(:modeling_id AS uuid)
      AND window_days = :window_days
    """
    return read_df(engine, sql, {"job_id": job_id, "modeling_id": modeling_id, "window_days": window_days})


def diagnostics_semantik() -> pd.DataFrame:
    sql = f"""
    SELECT
      COUNT(*)::bigint AS n_rows,
      COUNT(DISTINCT incident_number)::bigint AS n_ticket,
      COUNT(DISTINCT cluster_id)::bigint AS n_cluster,
      SUM(CASE WHEN event_time IS NULL THEN 1 ELSE 0 END)::bigint AS n_null_time,
      SUM(CASE WHEN cluster_id = -1 THEN 1 ELSE 0 END)::bigint AS n_rows_noise,
      COUNT(DISTINCT CASE WHEN cluster_id = -1 THEN incident_number END)::bigint AS n_ticket_noise,
      SUM(CASE WHEN temporal_cluster_no IS NULL THEN 1 ELSE 0 END)::bigint AS n_null_episode
    FROM {T_SEM_MEM}
    WHERE modeling_id = CAST(:modeling_id AS uuid)
      AND window_days = :window_days
      AND time_col = :time_col
    """
    return read_df(engine, sql, {"modeling_id": modeling_id, "window_days": window_days, "time_col": time_col})


with st.expander("🩺 Diagnostics (coverage / noise / null time)", expanded=True):
    if pendekatan == "sintaksis":
        d = diagnostics_sintaksis()
        st.dataframe(d, use_container_width=True, hide_index=True)
        st.caption("n_ticket adalah jumlah tiket yang masuk temporal_members (basis pelabelan).")
    else:
        d = diagnostics_semantik()
        st.dataframe(d, use_container_width=True, hide_index=True)
        st.caption(
            "Jika include_noise=False tapi n_ticket_noise besar, noise memang banyak di hasil HDBSCAN. "
            "Halaman ini memfilter noise saat stats & saat pelabelan."
        )

st.markdown("---")


# ======================================================
# 📌 Ringkasan Episode Berulang (episode berisi >1 tiket)
# ======================================================
def summary_repeat_episode_sintaksis(limit_top: int) -> pd.DataFrame:
    sql = f"""
    WITH base AS (
      SELECT
        cluster_id,
        temporal_cluster_no,
        COUNT(DISTINCT incident_number)::bigint AS n_tiket_episode
      FROM {T_SYN_MEM}
      WHERE job_id = CAST(:job_id AS uuid)
        AND modeling_id = CAST(:modeling_id AS uuid)
        AND window_days = :window_days
      GROUP BY cluster_id, temporal_cluster_no
    )
    SELECT
      cluster_id,
      SUM(n_tiket_episode)::bigint AS jumlah_tiket,
      COUNT(*)::bigint AS jumlah_episode,
      COUNT(*) FILTER (WHERE n_tiket_episode > 1)::bigint AS jumlah_episode_berulang,
      COALESCE(SUM(n_tiket_episode) FILTER (WHERE n_tiket_episode > 1), 0)::bigint AS jumlah_tiket_episode_berulang
    FROM base
    GROUP BY cluster_id
    HAVING COUNT(*) FILTER (WHERE n_tiket_episode > 1) > 0
    ORDER BY jumlah_tiket_episode_berulang DESC, jumlah_episode_berulang DESC, jumlah_tiket DESC, cluster_id ASC
    LIMIT :lim;
    """
    return read_df(
        engine,
        sql,
        {
            "job_id": job_id,
            "modeling_id": modeling_id,
            "window_days": window_days,
            "lim": int(limit_top),
        },
    )


def summary_repeat_episode_semantik(limit_top: int) -> pd.DataFrame:
    sql = f"""
    WITH base AS (
      SELECT
        cluster_id,
        temporal_cluster_no,
        COUNT(DISTINCT incident_number)::bigint AS n_tiket_episode
      FROM {T_SEM_MEM}
      WHERE modeling_id = CAST(:modeling_id AS uuid)
        AND window_days = :window_days
        AND time_col = :time_col
        AND ( :include_noise OR cluster_id <> -1 )
      GROUP BY cluster_id, temporal_cluster_no
    )
    SELECT
      cluster_id,
      SUM(n_tiket_episode)::bigint AS jumlah_tiket,
      COUNT(*)::bigint AS jumlah_episode,
      COUNT(*) FILTER (WHERE n_tiket_episode > 1)::bigint AS jumlah_episode_berulang,
      COALESCE(SUM(n_tiket_episode) FILTER (WHERE n_tiket_episode > 1), 0)::bigint AS jumlah_tiket_episode_berulang
    FROM base
    GROUP BY cluster_id
    HAVING COUNT(*) FILTER (WHERE n_tiket_episode > 1) > 0
    ORDER BY jumlah_tiket_episode_berulang DESC, jumlah_episode_berulang DESC, jumlah_tiket DESC, cluster_id ASC
    LIMIT :lim;
    """
    return read_df(
        engine,
        sql,
        {
            "modeling_id": modeling_id,
            "window_days": window_days,
            "time_col": time_col,
            "include_noise": bool(include_noise),
            "lim": int(limit_top),
        },
    )


with st.expander("📌 Ringkasan Cluster dengan Episode Berulang (episode >1 tiket)", expanded=False):
    st.caption(
        "Definisi di sini: episode berulang = episode (temporal_cluster_no) yang berisi >1 tiket. "
        "Tabel ini otomatis mengabaikan cluster yang tidak punya episode berulang."
    )
    if pendekatan == "sintaksis":
        df_sum = summary_repeat_episode_sintaksis(int(topk_cluster))
    else:
        df_sum = summary_repeat_episode_semantik(int(topk_cluster))

    if df_sum.empty:
        st.info("Tidak ada cluster dengan episode berulang (episode berisi >1 tiket) untuk run ini.")
    else:
        st.dataframe(df_sum, use_container_width=True, hide_index=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Cluster berulang (episode>1 tiket)", f"{df_sum['cluster_id'].nunique():,}")
        c2.metric("Total tiket (di cluster berulang)", f"{int(df_sum['jumlah_tiket'].sum()):,}")
        c3.metric("Total tiket episode berulang", f"{int(df_sum['jumlah_tiket_episode_berulang'].sum()):,}")

st.markdown("---")


# ======================================================
# ▶️ Query Builders (PREVIEW)
# ======================================================
def preview_labeling_sintaksis(limit: int) -> pd.DataFrame:
    sql = f"""
    WITH cluster_stats AS (
        SELECT
            job_id,
            modeling_id,
            window_days,
            cluster_id,
            COUNT(*)::bigint AS n_member_cluster,
            COUNT(DISTINCT temporal_cluster_no)::bigint AS n_episode_cluster,
            MIN(tgl_submit) AS min_time,
            MAX(tgl_submit) AS max_time
        FROM {T_SYN_MEM}
        WHERE job_id = CAST(:job_id AS uuid)
          AND modeling_id = CAST(:modeling_id AS uuid)
          AND window_days = :window_days
        GROUP BY job_id, modeling_id, window_days, cluster_id
    ),
    episode_stats AS (
        SELECT
            job_id,
            modeling_id,
            window_days,
            cluster_id,
            temporal_cluster_no,
            COUNT(*)::bigint AS n_member_episode
        FROM {T_SYN_MEM}
        WHERE job_id = CAST(:job_id AS uuid)
          AND modeling_id = CAST(:modeling_id AS uuid)
          AND window_days = :window_days
        GROUP BY job_id, modeling_id, window_days, cluster_id, temporal_cluster_no
    )
    SELECT
        'sintaksis'::text AS jenis_pendekatan,
        CAST(:job_id AS uuid) AS job_id,
        CAST(:modeling_id AS uuid) AS modeling_id,
        CAST(:window_days AS int) AS window_days,
        ''::text AS time_col,
        NULL::boolean AS include_noise,
        NULL::text AS eligible_rule,

        m.incident_number,
        NULL::timestamp AS event_time,
        m.tgl_submit AS tgl_submit,

        m.cluster_id,
        m.temporal_cluster_no,
        m.temporal_cluster_id,
        m.gap_days,

        m.site, m.assignee, m.modul, m.sub_modul,

        s.n_member_cluster,
        s.n_episode_cluster,
        e.n_member_episode,
        s.min_time,
        s.max_time,

        CASE
          WHEN s.n_member_cluster >= :min_member_cluster
           AND s.n_episode_cluster >= :min_episode_cluster
           AND e.n_member_episode >= :min_member_episode
          THEN 1 ELSE 0
        END AS label_berulang,
        CAST(:rule_json AS jsonb) AS rule_json
    FROM {T_SYN_MEM} m
    JOIN cluster_stats s
      ON s.job_id = m.job_id
     AND s.modeling_id = m.modeling_id
     AND s.window_days = m.window_days
     AND s.cluster_id = m.cluster_id
    JOIN episode_stats e
      ON e.job_id = m.job_id
     AND e.modeling_id = m.modeling_id
     AND e.window_days = m.window_days
     AND e.cluster_id = m.cluster_id
     AND e.temporal_cluster_no = m.temporal_cluster_no
    WHERE m.job_id = CAST(:job_id AS uuid)
      AND m.modeling_id = CAST(:modeling_id AS uuid)
      AND m.window_days = :window_days
    ORDER BY m.cluster_id, m.temporal_cluster_no, m.incident_number
    LIMIT :lim
    """
    return read_df(
        engine,
        sql,
        {
            "job_id": job_id,
            "modeling_id": modeling_id,
            "window_days": window_days,
            "min_member_cluster": int(min_member_cluster),
            "min_episode_cluster": int(min_episode_cluster),
            "min_member_episode": int(min_member_episode),
            "rule_json": json.dumps(rule_json),
            "lim": int(limit),
        },
    )


def preview_labeling_semantik(limit: int) -> pd.DataFrame:
    noise_rows = "TRUE" if bool(include_noise) else "m.cluster_id <> -1"
    noise_stats = "TRUE" if bool(include_noise) else "cluster_id <> -1"

    sql = f"""
    WITH cluster_stats AS (
        SELECT
            modeling_id,
            window_days,
            time_col,
            cluster_id,
            COUNT(*)::bigint AS n_member_cluster,
            COUNT(DISTINCT temporal_cluster_no)::bigint AS n_episode_cluster,
            MIN(event_time) AS min_time,
            MAX(event_time) AS max_time
        FROM {T_SEM_MEM}
        WHERE modeling_id = CAST(:modeling_id AS uuid)
          AND window_days = :window_days
          AND time_col = :time_col
          AND ({noise_stats})
        GROUP BY modeling_id, window_days, time_col, cluster_id
    ),
    episode_stats AS (
        SELECT
            modeling_id,
            window_days,
            time_col,
            cluster_id,
            temporal_cluster_no,
            COUNT(*)::bigint AS n_member_episode
        FROM {T_SEM_MEM}
        WHERE modeling_id = CAST(:modeling_id AS uuid)
          AND window_days = :window_days
          AND time_col = :time_col
          AND ({noise_stats})
        GROUP BY modeling_id, window_days, time_col, cluster_id, temporal_cluster_no
    )
    SELECT
        'semantik'::text AS jenis_pendekatan,
        NULL::uuid AS job_id,
        CAST(:modeling_id AS uuid) AS modeling_id,
        CAST(:window_days AS int) AS window_days,
        CAST(:time_col AS text) AS time_col,
        CAST(:include_noise AS boolean) AS include_noise,
        CAST(:eligible_rule AS text) AS eligible_rule,

        m.incident_number,
        m.event_time AS event_time,
        NULL::timestamp AS tgl_submit,

        m.cluster_id,
        m.temporal_cluster_no,
        m.temporal_cluster_id,
        m.gap_days,

        m.site, m.assignee, m.modul, m.sub_modul,

        s.n_member_cluster,
        s.n_episode_cluster,
        e.n_member_episode,
        s.min_time,
        s.max_time,

        CASE
          WHEN s.n_member_cluster >= :min_member_cluster
           AND s.n_episode_cluster >= :min_episode_cluster
           AND e.n_member_episode >= :min_member_episode
          THEN 1 ELSE 0
        END AS label_berulang,
        CAST(:rule_json AS jsonb) AS rule_json
    FROM {T_SEM_MEM} m
    JOIN cluster_stats s
      ON s.modeling_id = m.modeling_id
     AND s.window_days = m.window_days
     AND s.time_col = m.time_col
     AND s.cluster_id = m.cluster_id
    JOIN episode_stats e
      ON e.modeling_id = m.modeling_id
     AND e.window_days = m.window_days
     AND e.time_col = m.time_col
     AND e.cluster_id = m.cluster_id
     AND e.temporal_cluster_no = m.temporal_cluster_no
    WHERE m.modeling_id = CAST(:modeling_id AS uuid)
      AND m.window_days = :window_days
      AND m.time_col = :time_col
      AND ({noise_rows})
    ORDER BY m.cluster_id, m.temporal_cluster_no, m.incident_number
    LIMIT :lim
    """
    return read_df(
        engine,
        sql,
        {
            "modeling_id": modeling_id,
            "window_days": window_days,
            "time_col": time_col,
            "include_noise": bool(include_noise),
            "eligible_rule": str(eligible_rule),
            "min_member_cluster": int(min_member_cluster),
            "min_episode_cluster": int(min_episode_cluster),
            "min_member_episode": int(min_member_episode),
            "rule_json": json.dumps(rule_json),
            "lim": int(limit),
        },
    )


# ======================================================
# ▶️ Query Builders (WRITE)
# ======================================================
def write_labeling_sintaksis(replace: bool) -> None:
    if replace:
        exec_sql(
            engine,
            f"""
            DELETE FROM {T_LABEL}
            WHERE jenis_pendekatan='sintaksis'
              AND job_id = CAST(:job_id AS uuid)
              AND modeling_id = CAST(:modeling_id AS uuid)
              AND window_days = :window_days
              AND time_col = ''
            """,
            {"job_id": job_id, "modeling_id": modeling_id, "window_days": window_days},
        )

    sql = f"""
    WITH cluster_stats AS (
        SELECT
            job_id,
            modeling_id,
            window_days,
            cluster_id,
            COUNT(*)::bigint AS n_member_cluster,
            COUNT(DISTINCT temporal_cluster_no)::bigint AS n_episode_cluster,
            MIN(tgl_submit) AS min_time,
            MAX(tgl_submit) AS max_time
        FROM {T_SYN_MEM}
        WHERE job_id = CAST(:job_id AS uuid)
          AND modeling_id = CAST(:modeling_id AS uuid)
          AND window_days = :window_days
        GROUP BY job_id, modeling_id, window_days, cluster_id
    ),
    episode_stats AS (
        SELECT
            job_id,
            modeling_id,
            window_days,
            cluster_id,
            temporal_cluster_no,
            COUNT(*)::bigint AS n_member_episode
        FROM {T_SYN_MEM}
        WHERE job_id = CAST(:job_id AS uuid)
          AND modeling_id = CAST(:modeling_id AS uuid)
          AND window_days = :window_days
        GROUP BY job_id, modeling_id, window_days, cluster_id, temporal_cluster_no
    )
    INSERT INTO {T_LABEL} (
        jenis_pendekatan, job_id, modeling_id, window_days, time_col, include_noise, eligible_rule,
        incident_number, event_time, tgl_submit,
        cluster_id, temporal_cluster_no, temporal_cluster_id, gap_days,
        site, assignee, modul, sub_modul,
        n_member_cluster, n_episode_cluster, n_member_episode, min_time, max_time,
        label_berulang, rule_json
    )
    SELECT
        'sintaksis'::text AS jenis_pendekatan,
        CAST(:job_id AS uuid) AS job_id,
        CAST(:modeling_id AS uuid) AS modeling_id,
        CAST(:window_days AS int) AS window_days,
        ''::text AS time_col,
        NULL::boolean AS include_noise,
        NULL::text AS eligible_rule,

        m.incident_number,
        NULL::timestamp AS event_time,
        m.tgl_submit AS tgl_submit,

        m.cluster_id,
        m.temporal_cluster_no,
        m.temporal_cluster_id,
        m.gap_days,

        m.site, m.assignee, m.modul, m.sub_modul,

        s.n_member_cluster,
        s.n_episode_cluster,
        e.n_member_episode,
        s.min_time,
        s.max_time,

        CASE
          WHEN s.n_member_cluster >= :min_member_cluster
           AND s.n_episode_cluster >= :min_episode_cluster
           AND e.n_member_episode >= :min_member_episode
          THEN 1 ELSE 0
        END AS label_berulang,
        CAST(:rule_json AS jsonb) AS rule_json
    FROM {T_SYN_MEM} m
    JOIN cluster_stats s
      ON s.job_id = m.job_id
     AND s.modeling_id = m.modeling_id
     AND s.window_days = m.window_days
     AND s.cluster_id = m.cluster_id
    JOIN episode_stats e
      ON e.job_id = m.job_id
     AND e.modeling_id = m.modeling_id
     AND e.window_days = m.window_days
     AND e.cluster_id = m.cluster_id
     AND e.temporal_cluster_no = m.temporal_cluster_no
    WHERE m.job_id = CAST(:job_id AS uuid)
      AND m.modeling_id = CAST(:modeling_id AS uuid)
      AND m.window_days = :window_days
    ON CONFLICT (jenis_pendekatan, modeling_id, window_days, incident_number, time_col)
    DO UPDATE SET
        run_time = now(),
        label_berulang = EXCLUDED.label_berulang,
        rule_json = EXCLUDED.rule_json,
        n_member_cluster = EXCLUDED.n_member_cluster,
        n_episode_cluster = EXCLUDED.n_episode_cluster,
        n_member_episode = EXCLUDED.n_member_episode,
        min_time = EXCLUDED.min_time,
        max_time = EXCLUDED.max_time,
        cluster_id = EXCLUDED.cluster_id,
        temporal_cluster_no = EXCLUDED.temporal_cluster_no,
        temporal_cluster_id = EXCLUDED.temporal_cluster_id,
        gap_days = EXCLUDED.gap_days,
        site = EXCLUDED.site,
        assignee = EXCLUDED.assignee,
        modul = EXCLUDED.modul,
        sub_modul = EXCLUDED.sub_modul,
        tgl_submit = EXCLUDED.tgl_submit;
    """
    exec_sql(
        engine,
        sql,
        {
            "job_id": job_id,
            "modeling_id": modeling_id,
            "window_days": window_days,
            "min_member_cluster": int(min_member_cluster),
            "min_episode_cluster": int(min_episode_cluster),
            "min_member_episode": int(min_member_episode),
            "rule_json": json.dumps(rule_json),
        },
    )


def write_labeling_semantik(replace: bool) -> None:
    if replace:
        exec_sql(
            engine,
            f"""
            DELETE FROM {T_LABEL}
            WHERE jenis_pendekatan='semantik'
              AND modeling_id = CAST(:modeling_id AS uuid)
              AND window_days = :window_days
              AND time_col = :time_col
            """,
            {"modeling_id": modeling_id, "window_days": window_days, "time_col": time_col},
        )

    noise_rows = "TRUE" if bool(include_noise) else "m.cluster_id <> -1"
    noise_stats = "TRUE" if bool(include_noise) else "cluster_id <> -1"

    sql = f"""
    WITH cluster_stats AS (
        SELECT
            modeling_id,
            window_days,
            time_col,
            cluster_id,
            COUNT(*)::bigint AS n_member_cluster,
            COUNT(DISTINCT temporal_cluster_no)::bigint AS n_episode_cluster,
            MIN(event_time) AS min_time,
            MAX(event_time) AS max_time
        FROM {T_SEM_MEM}
        WHERE modeling_id = CAST(:modeling_id AS uuid)
          AND window_days = :window_days
          AND time_col = :time_col
          AND ({noise_stats})
        GROUP BY modeling_id, window_days, time_col, cluster_id
    ),
    episode_stats AS (
        SELECT
            modeling_id,
            window_days,
            time_col,
            cluster_id,
            temporal_cluster_no,
            COUNT(*)::bigint AS n_member_episode
        FROM {T_SEM_MEM}
        WHERE modeling_id = CAST(:modeling_id AS uuid)
          AND window_days = :window_days
          AND time_col = :time_col
          AND ({noise_stats})
        GROUP BY modeling_id, window_days, time_col, cluster_id, temporal_cluster_no
    )
    INSERT INTO {T_LABEL} (
        jenis_pendekatan, job_id, modeling_id, window_days, time_col, include_noise, eligible_rule,
        incident_number, event_time, tgl_submit,
        cluster_id, temporal_cluster_no, temporal_cluster_id, gap_days,
        site, assignee, modul, sub_modul,
        n_member_cluster, n_episode_cluster, n_member_episode, min_time, max_time,
        label_berulang, rule_json
    )
    SELECT
        'semantik'::text AS jenis_pendekatan,
        NULL::uuid AS job_id,
        CAST(:modeling_id AS uuid) AS modeling_id,
        CAST(:window_days AS int) AS window_days,
        CAST(:time_col AS text) AS time_col,
        CAST(:include_noise AS boolean) AS include_noise,
        CAST(:eligible_rule AS text) AS eligible_rule,

        m.incident_number,
        m.event_time AS event_time,
        NULL::timestamp AS tgl_submit,

        m.cluster_id,
        m.temporal_cluster_no,
        m.temporal_cluster_id,
        m.gap_days,

        m.site, m.assignee, m.modul, m.sub_modul,

        s.n_member_cluster,
        s.n_episode_cluster,
        e.n_member_episode,
        s.min_time,
        s.max_time,

        CASE
          WHEN s.n_member_cluster >= :min_member_cluster
           AND s.n_episode_cluster >= :min_episode_cluster
           AND e.n_member_episode >= :min_member_episode
          THEN 1 ELSE 0
        END AS label_berulang,
        CAST(:rule_json AS jsonb) AS rule_json
    FROM {T_SEM_MEM} m
    JOIN cluster_stats s
      ON s.modeling_id = m.modeling_id
     AND s.window_days = m.window_days
     AND s.time_col = m.time_col
     AND s.cluster_id = m.cluster_id
    JOIN episode_stats e
      ON e.modeling_id = m.modeling_id
     AND e.window_days = m.window_days
     AND e.time_col = m.time_col
     AND e.cluster_id = m.cluster_id
     AND e.temporal_cluster_no = m.temporal_cluster_no
    WHERE m.modeling_id = CAST(:modeling_id AS uuid)
      AND m.window_days = :window_days
      AND m.time_col = :time_col
      AND ({noise_rows})
    ON CONFLICT (jenis_pendekatan, modeling_id, window_days, incident_number, time_col)
    DO UPDATE SET
        run_time = now(),
        label_berulang = EXCLUDED.label_berulang,
        rule_json = EXCLUDED.rule_json,
        n_member_cluster = EXCLUDED.n_member_cluster,
        n_episode_cluster = EXCLUDED.n_episode_cluster,
        n_member_episode = EXCLUDED.n_member_episode,
        min_time = EXCLUDED.min_time,
        max_time = EXCLUDED.max_time,
        cluster_id = EXCLUDED.cluster_id,
        temporal_cluster_no = EXCLUDED.temporal_cluster_no,
        temporal_cluster_id = EXCLUDED.temporal_cluster_id,
        gap_days = EXCLUDED.gap_days,
        site = EXCLUDED.site,
        assignee = EXCLUDED.assignee,
        modul = EXCLUDED.modul,
        sub_modul = EXCLUDED.sub_modul,
        event_time = EXCLUDED.event_time,
        include_noise = EXCLUDED.include_noise,
        eligible_rule = EXCLUDED.eligible_rule;
    """
    exec_sql(
        engine,
        sql,
        {
            "modeling_id": modeling_id,
            "window_days": window_days,
            "time_col": time_col,
            "include_noise": bool(include_noise),
            "eligible_rule": str(eligible_rule),
            "min_member_cluster": int(min_member_cluster),
            "min_episode_cluster": int(min_episode_cluster),
            "min_member_episode": int(min_member_episode),
            "rule_json": json.dumps(rule_json),
        },
    )


def recap_saved() -> pd.DataFrame:
    if pendekatan == "sintaksis":
        sql = f"""
        SELECT
          COUNT(*)::bigint AS n_rows,
          SUM(CASE WHEN label_berulang=1 THEN 1 ELSE 0 END)::bigint AS n_label1,
          SUM(CASE WHEN label_berulang=0 THEN 1 ELSE 0 END)::bigint AS n_label0,
          COUNT(DISTINCT cluster_id)::bigint AS n_clusters,
          COUNT(DISTINCT temporal_cluster_id)::bigint AS n_episodes
        FROM {T_LABEL}
        WHERE jenis_pendekatan='sintaksis'
          AND job_id = CAST(:job_id AS uuid)
          AND modeling_id = CAST(:modeling_id AS uuid)
          AND window_days = :window_days
          AND time_col = ''
        """
        return read_df(engine, sql, {"job_id": job_id, "modeling_id": modeling_id, "window_days": window_days})

    sql = f"""
    SELECT
      COUNT(*)::bigint AS n_rows,
      SUM(CASE WHEN label_berulang=1 THEN 1 ELSE 0 END)::bigint AS n_label1,
      SUM(CASE WHEN label_berulang=0 THEN 1 ELSE 0 END)::bigint AS n_label0,
      COUNT(DISTINCT cluster_id)::bigint AS n_clusters,
      COUNT(DISTINCT temporal_cluster_id)::bigint AS n_episodes
    FROM {T_LABEL}
    WHERE jenis_pendekatan='semantik'
      AND modeling_id = CAST(:modeling_id AS uuid)
      AND window_days = :window_days
      AND time_col = :time_col
    """
    return read_df(engine, sql, {"modeling_id": modeling_id, "window_days": window_days, "time_col": time_col})


# ======================================================
# ▶️ RUN
# ======================================================
if st.button("🚀 Proses Pelabelan", type="primary"):
    # 1) Preview
    with st.spinner("Menghitung agregasi cluster+episode & membentuk label (preview)..."):
        if pendekatan == "sintaksis":
            df_preview = preview_labeling_sintaksis(int(preview_limit))
        else:
            df_preview = preview_labeling_semantik(int(preview_limit))

    if df_preview.empty:
        st.warning("Hasil preview kosong. Periksa apakah temporal_members untuk run terpilih berisi data.")
        st.stop()

    st.success(f"Preview berhasil: {len(df_preview):,} baris")
    st.dataframe(df_preview, use_container_width=True, height=420)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows (preview)", f"{len(df_preview):,}")
    c2.metric("Label=1", f"{int((df_preview['label_berulang'] == 1).sum()):,}")
    c3.metric("Label=0", f"{int((df_preview['label_berulang'] == 0).sum()):,}")
    c4.metric("Clusters", f"{df_preview['cluster_id'].nunique():,}")
    c5.metric("Episodes", f"{df_preview['temporal_cluster_id'].nunique():,}")

    # 2) Write full
    if do_write:
        with st.spinner("Menulis hasil pelabelan ke database (full set)..."):
            if pendekatan == "sintaksis":
                write_labeling_sintaksis(replace=do_replace)
            else:
                write_labeling_semantik(replace=do_replace)

        st.success(f"✅ Hasil pelabelan tersimpan ke {T_LABEL}")

        # 3) Recap saved
        st.subheader("Rekap hasil tersimpan")
        df_cnt = recap_saved()
        st.dataframe(df_cnt, use_container_width=True, hide_index=True)
    else:
        st.info("Mode tulis dimatikan (do_write=False). Preview saja ditampilkan.")
