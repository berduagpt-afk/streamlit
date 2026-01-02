# pages/modeling_semantic_hdbscan_temporal.py
# ============================================================
# Analisis Temporal HDBSCAN — Semantik
#
# Input:
# - lasis_djp.modeling_semantic_hdbscan_runs
# - lasis_djp.modeling_semantic_hdbscan_members
# - lasis_djp.incident_semantic (tgl_submit, modul, sub_modul, text_semantic)
#
# Output: Read-only viewer (tidak menulis DB)
#
# Fitur:
# - Pilih modeling_id (HDBSCAN run)
# - Pilih bucket waktu: day / week / month
# - KPI temporal per cluster: first_seen, last_seen, active buckets, span_days
# - Heuristik recurring: n_active_buckets >= threshold
# - Visual:
#   1) Trend total tickets per time bucket
#   2) Heatmap cluster_id x time bucket (top-N clusters)
#   3) Drilldown cluster -> trend + contoh tiket
#
# PATCH FINAL:
# ✅ FIX Streamlit: tidak ada nested expander (expander dalam expander)
# ✅ Semua agregasi dilakukan di PostgreSQL (hemat RAM)
# ✅ Hindari UnhashableParamError: arg Engine di cache pakai _engine
# ============================================================

from __future__ import annotations

import pandas as pd
import streamlit as st
import altair as alt
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# =========================
# 🔐 Guard login
# =========================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login terlebih dahulu untuk mengakses halaman ini.")
    st.stop()


SCHEMA = "lasis_djp"
T_RUNS = "modeling_semantic_hdbscan_runs"
T_MEM = "modeling_semantic_hdbscan_members"
T_SEM = "incident_semantic"


# =========================
# 🔌 DB
# =========================
@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    cfg = st.secrets["connections"]["postgres"]
    url = (
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg.get('port', 5432)}/{cfg['database']}"
    )
    return create_engine(url, pool_pre_ping=True)


@st.cache_data(show_spinner=False, ttl=120)
def load_runs(_engine: Engine, limit: int = 200) -> pd.DataFrame:
    q = f"""
    SELECT
      modeling_id::text AS modeling_id,
      embedding_run_id::text AS embedding_run_id,
      model_name,
      run_time,
      n_rows,
      n_clusters,
      n_noise,
      silhouette,
      dbi,
      notes
    FROM {SCHEMA}.{T_RUNS}
    ORDER BY run_time DESC
    LIMIT :lim
    """
    return pd.read_sql_query(text(q), _engine, params={"lim": int(limit)})


def _bucket_expr(bucket: str) -> str:
    if bucket == "day":
        return "date_trunc('day', s.tgl_submit)"
    if bucket == "week":
        return "date_trunc('week', s.tgl_submit)"
    return "date_trunc('month', s.tgl_submit)"


@st.cache_data(show_spinner=False, ttl=120)
def load_total_trend(_engine: Engine, modeling_id: str, bucket: str, include_noise: bool) -> pd.DataFrame:
    bucket_expr = _bucket_expr(bucket)
    noise_cond = "" if include_noise else "AND m.cluster_id <> -1"
    q = f"""
    SELECT
      {bucket_expr} AS t_bucket,
      COUNT(*) AS n_tickets
    FROM {SCHEMA}.{T_MEM} m
    JOIN {SCHEMA}.{T_SEM} s
      ON s.incident_number::text = m.incident_number::text
    WHERE m.modeling_id = :mid
      AND s.tgl_submit IS NOT NULL
      {noise_cond}
    GROUP BY 1
    ORDER BY 1
    """
    return pd.read_sql_query(text(q), _engine, params={"mid": modeling_id})


@st.cache_data(show_spinner=False, ttl=120)
def load_cluster_summary_temporal(
    _engine: Engine,
    modeling_id: str,
    bucket: str,
    include_noise: bool,
    min_cluster_size: int,
) -> pd.DataFrame:
    bucket_expr = _bucket_expr(bucket)
    noise_cond = "" if include_noise else "AND m.cluster_id <> -1"

    q = f"""
    WITH base AS (
      SELECT
        m.cluster_id,
        s.tgl_submit,
        {bucket_expr} AS t_bucket
      FROM {SCHEMA}.{T_MEM} m
      JOIN {SCHEMA}.{T_SEM} s
        ON s.incident_number::text = m.incident_number::text
      WHERE m.modeling_id = :mid
        AND s.tgl_submit IS NOT NULL
        {noise_cond}
    ),
    agg AS (
      SELECT
        cluster_id,
        MIN(tgl_submit) AS first_seen,
        MAX(tgl_submit) AS last_seen,
        COUNT(*) AS total_tickets,
        COUNT(DISTINCT t_bucket) AS n_active_buckets
      FROM base
      GROUP BY cluster_id
    )
    SELECT
      a.*,
      EXTRACT(EPOCH FROM (a.last_seen - a.first_seen)) / 86400.0 AS span_days
    FROM agg a
    WHERE a.total_tickets >= :min_size
    ORDER BY a.total_tickets DESC
    """
    return pd.read_sql_query(text(q), _engine, params={"mid": modeling_id, "min_size": int(min_cluster_size)})


@st.cache_data(show_spinner=False, ttl=120)
def load_heatmap(
    _engine: Engine,
    modeling_id: str,
    bucket: str,
    include_noise: bool,
    top_n: int,
    min_cluster_size: int,
) -> pd.DataFrame:
    bucket_expr = _bucket_expr(bucket)
    noise_cond = "" if include_noise else "AND m.cluster_id <> -1"

    q = f"""
    WITH base AS (
      SELECT
        m.cluster_id,
        {bucket_expr} AS t_bucket
      FROM {SCHEMA}.{T_MEM} m
      JOIN {SCHEMA}.{T_SEM} s
        ON s.incident_number::text = m.incident_number::text
      WHERE m.modeling_id = :mid
        AND s.tgl_submit IS NOT NULL
        {noise_cond}
    ),
    totals AS (
      SELECT cluster_id, COUNT(*) AS total_tickets
      FROM base
      GROUP BY cluster_id
      HAVING COUNT(*) >= :min_size
      ORDER BY total_tickets DESC
      LIMIT :topn
    )
    SELECT
      b.cluster_id,
      b.t_bucket,
      COUNT(*) AS n_tickets
    FROM base b
    JOIN totals t USING (cluster_id)
    GROUP BY b.cluster_id, b.t_bucket
    ORDER BY b.cluster_id, b.t_bucket
    """
    return pd.read_sql_query(
        text(q),
        _engine,
        params={"mid": modeling_id, "topn": int(top_n), "min_size": int(min_cluster_size)},
    )


@st.cache_data(show_spinner=False, ttl=120)
def load_cluster_trend(_engine: Engine, modeling_id: str, cluster_id: int, bucket: str) -> pd.DataFrame:
    bucket_expr = _bucket_expr(bucket)
    q = f"""
    SELECT
      {bucket_expr} AS t_bucket,
      COUNT(*) AS n_tickets
    FROM {SCHEMA}.{T_MEM} m
    JOIN {SCHEMA}.{T_SEM} s
      ON s.incident_number::text = m.incident_number::text
    WHERE m.modeling_id = :mid
      AND m.cluster_id = :cid
      AND s.tgl_submit IS NOT NULL
    GROUP BY 1
    ORDER BY 1
    """
    return pd.read_sql_query(text(q), _engine, params={"mid": modeling_id, "cid": int(cluster_id)})


@st.cache_data(show_spinner=False, ttl=120)
def load_cluster_examples(_engine: Engine, modeling_id: str, cluster_id: int, limit: int = 30) -> pd.DataFrame:
    q = f"""
    SELECT
      m.incident_number,
      m.score,
      s.tgl_submit,
      s.modul,
      s.sub_modul,
      s.text_semantic
    FROM {SCHEMA}.{T_MEM} m
    LEFT JOIN {SCHEMA}.{T_SEM} s
      ON s.incident_number::text = m.incident_number::text
    WHERE m.modeling_id = :mid
      AND m.cluster_id = :cid
    ORDER BY m.score DESC NULLS LAST, s.tgl_submit DESC NULLS LAST
    LIMIT :lim
    """
    return pd.read_sql_query(text(q), _engine, params={"mid": modeling_id, "cid": int(cluster_id), "lim": int(limit)})


# =========================
# 🧭 UI
# =========================
st.title("🗓️ Analisis Temporal — HDBSCAN Semantik")
st.caption(
    "Memetakan kemunculan cluster dari waktu ke waktu untuk melihat pola berulang (recurring), musiman, atau sekali muncul (burst)."
)

engine = get_engine()
df_runs = load_runs(engine, limit=300)
if df_runs.empty:
    st.warning("Belum ada run HDBSCAN. Jalankan modeling_semantic_hdbscan.py dulu.")
    st.stop()

with st.sidebar:
    st.header("📌 Pilih Run")

    idx = st.selectbox(
        "Run (modeling_id | model | rows)",
        options=list(range(len(df_runs))),
        format_func=lambda i: (
            f"{df_runs.loc[i,'modeling_id']} | {df_runs.loc[i,'model_name']} | "
            f"rows={int(df_runs.loc[i,'n_rows'] or 0):,} | clusters={int(df_runs.loc[i,'n_clusters'] or 0):,}"
        ),
    )
    modeling_id = str(df_runs.loc[idx, "modeling_id"])

    st.divider()
    st.header("⚙️ Pengaturan Temporal")
    bucket = st.selectbox("Time bucket", options=["day", "week", "month"], index=1)
    include_noise = st.checkbox("Include noise (-1) dalam analisis", value=False)

    min_cluster_size = st.number_input(
        "Min cluster size (filter)",
        min_value=1,
        max_value=10000,
        value=10,
        step=1,
        help="Fokus ke cluster bermakna dan mengurangi cluster kecil/noise.",
    )

    top_n = st.slider("Top-N clusters untuk heatmap", min_value=5, max_value=60, value=20, step=1)

    st.divider()
    st.subheader("Heuristik Recurring")
    recurring_min_active = st.number_input(
        "Minimal bucket aktif agar dianggap recurring",
        min_value=2, max_value=500, value=6, step=1,
        help="Contoh: bucket=week, nilai 6 berarti cluster aktif minimal 6 minggu berbeda."
    )

# KPI run
r = df_runs.loc[idx].to_dict()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{int(r.get('n_rows') or 0):,}")
c2.metric("Clusters", f"{int(r.get('n_clusters') or 0):,}")
c3.metric("Noise", f"{int(r.get('n_noise') or 0):,}")
c4.metric("Silhouette", "-" if r.get("silhouette") is None else f"{float(r['silhouette']):.4f}")

# Trend total
st.subheader("Trend Total Tiket per Periode")
df_total = load_total_trend(engine, modeling_id, bucket=bucket, include_noise=include_noise)
if df_total.empty:
    st.info("Tidak ada data trend (cek tgl_submit di incident_semantic dan join incident_number).")
else:
    ch_total = (
        alt.Chart(df_total)
        .mark_line(point=True)
        .encode(
            x=alt.X("t_bucket:T", title=f"time bucket ({bucket})"),
            y=alt.Y("n_tickets:Q", title="jumlah tiket"),
            tooltip=[alt.Tooltip("t_bucket:T"), alt.Tooltip("n_tickets:Q")]
        )
        .properties(height=300)
    )
    st.altair_chart(ch_total, use_container_width=True)

# Ringkasan cluster temporal
st.subheader("Ringkasan Temporal per Cluster")
df_sum = load_cluster_summary_temporal(
    engine, modeling_id, bucket=bucket, include_noise=include_noise, min_cluster_size=int(min_cluster_size)
)
if df_sum.empty:
    st.info("Tidak ada cluster yang lolos filter min_cluster_size.")
    st.stop()

df_sum = df_sum.copy()
df_sum["is_recurring"] = df_sum["n_active_buckets"] >= int(recurring_min_active)
df_sum["first_seen"] = pd.to_datetime(df_sum["first_seen"])
df_sum["last_seen"] = pd.to_datetime(df_sum["last_seen"])

cA, cB, cC = st.columns(3)
cA.metric("Clusters dianalisis", f"{len(df_sum):,}")
cB.metric("Recurring clusters", f"{int(df_sum['is_recurring'].sum()):,}")
cC.metric("Non-recurring clusters", f"{int((~df_sum['is_recurring']).sum()):,}")

st.dataframe(
    df_sum[["cluster_id", "total_tickets", "n_active_buckets", "span_days", "first_seen", "last_seen", "is_recurring"]],
    use_container_width=True,
    height=360
)

# Heatmap
st.subheader("Heatmap Cluster × Waktu (Top-N)")
df_hm = load_heatmap(engine, modeling_id, bucket=bucket, include_noise=include_noise, top_n=int(top_n), min_cluster_size=int(min_cluster_size))
if df_hm.empty:
    st.info("Heatmap kosong (cek filter).")
else:
    df_hm2 = df_hm.copy()
    df_hm2["cluster_id"] = df_hm2["cluster_id"].astype(int).astype(str)
    ch_hm = (
        alt.Chart(df_hm2)
        .mark_rect()
        .encode(
            x=alt.X("t_bucket:T", title="waktu"),
            y=alt.Y("cluster_id:O", title="cluster_id"),
            color=alt.Color("n_tickets:Q", title="tickets"),
            tooltip=["cluster_id", alt.Tooltip("t_bucket:T"), "n_tickets"]
        )
        .properties(height=380)
    )
    st.altair_chart(ch_hm, use_container_width=True)

# Drilldown cluster
st.subheader("Drilldown Cluster (Temporal + Contoh Tiket)")
cluster_choices = df_sum["cluster_id"].astype(int).tolist()
selected_cluster = st.selectbox("Pilih cluster_id", options=cluster_choices, index=0)

df_ct = load_cluster_trend(engine, modeling_id, int(selected_cluster), bucket=bucket)
if not df_ct.empty:
    ch_ct = (
        alt.Chart(df_ct)
        .mark_line(point=True)
        .encode(
            x=alt.X("t_bucket:T", title=f"time bucket ({bucket})"),
            y=alt.Y("n_tickets:Q", title="jumlah tiket"),
            tooltip=[alt.Tooltip("t_bucket:T"), alt.Tooltip("n_tickets:Q")]
        )
        .properties(height=280)
    )
    st.altair_chart(ch_ct, use_container_width=True)
else:
    st.info("Trend cluster kosong.")

# Contoh tiket + preview text (NO nested expander)
with st.expander("Contoh tiket (top by score)", expanded=False):
    df_ex = load_cluster_examples(engine, modeling_id, int(selected_cluster), limit=30)
    if df_ex.empty:
        st.info("Tidak ada contoh.")
    else:
        tab1, tab2 = st.tabs(["Tabel", "Preview Top 5"])
        with tab1:
            st.dataframe(df_ex, use_container_width=True, height=420)
        with tab2:
            top5 = df_ex.head(5)
            for _, rr in top5.iterrows():
                st.markdown(f"**{rr['incident_number']}** | score={rr.get('score')}")
                st.write(rr.get("text_semantic", ""))

st.caption(
    "Catatan: Heuristik recurring di sini berbasis jumlah bucket aktif. "
    "Untuk analisis lanjutan, bisa ditambah metrik gap antar kemunculan dan burstiness."
)
