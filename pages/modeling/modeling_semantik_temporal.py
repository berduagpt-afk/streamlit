# pages/modeling_semantic_hdbscan_temporal.py
# ============================================================
# Analisis Temporal HDBSCAN — Semantik (DDL Baru)
#
# Sumber (DDL baru):
# - lasis_djp.modeling_semantik_hdbscan_runs
# - lasis_djp.modeling_semantik_hdbscan_clusters
# - lasis_djp.modeling_semantik_hdbscan_members  (sudah ada tgl_submit/site/modul/sub_modul)
#
# Opsional:
# - lasis_djp.incident_semantik (untuk text_semantic) -> auto-detect jika ada
#
# Output: Read-only viewer
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
T_RUNS = "modeling_semantik_hdbscan_runs"
T_CLUST = "modeling_semantik_hdbscan_clusters"
T_MEM = "modeling_semantik_hdbscan_members"

# opsional (hanya untuk preview teks)
T_TEXT = "incident_semantik"  # jika ada kolom text_semantic + incident_number

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


@st.cache_data(show_spinner=False, ttl=600)
def has_table(_engine: Engine, schema: str, table: str) -> bool:
    q = "SELECT to_regclass(:full_name) IS NOT NULL AS ok"
    full_name = f"{schema}.{table}"
    df = pd.read_sql_query(text(q), _engine, params={"full_name": full_name})
    return bool(df.iloc[0]["ok"])


def _bucket_expr(bucket: str) -> str:
    if bucket == "day":
        return "date_trunc('day', m.tgl_submit)"
    if bucket == "week":
        return "date_trunc('week', m.tgl_submit)"
    return "date_trunc('month', m.tgl_submit)"


def _noise_cond(include_noise: bool) -> str:
    # DDL baru: is_noise boolean
    return "" if include_noise else "AND m.is_noise = FALSE"


# =========================
# 📥 Loaders (PostgreSQL heavy lifting)
# =========================
@st.cache_data(show_spinner=False, ttl=300)
def load_runs(_engine: Engine, limit: int = 200) -> pd.DataFrame:
    q = f"""
    SELECT
      modeling_id::text AS modeling_id,
      embedding_run_id::text AS embedding_run_id,
      run_time,
      n_rows,
      n_clusters,
      n_noise,
      silhouette,
      dbi,
      params_json,
      notes
    FROM {SCHEMA}.{T_RUNS}
    ORDER BY run_time DESC
    LIMIT :lim
    """
    return pd.read_sql_query(text(q), _engine, params={"lim": int(limit)})


@st.cache_data(show_spinner=False, ttl=300)
def load_total_trend(_engine: Engine, modeling_id: str, bucket: str, include_noise: bool) -> pd.DataFrame:
    bucket_expr = _bucket_expr(bucket)
    noise_cond = _noise_cond(include_noise)

    q = f"""
    SELECT
      {bucket_expr} AS t_bucket,
      COUNT(*) AS n_tickets
    FROM {SCHEMA}.{T_MEM} m
    WHERE m.modeling_id = CAST(:mid AS uuid)
      AND m.tgl_submit IS NOT NULL
      {noise_cond}
    GROUP BY 1
    ORDER BY 1
    """
    return pd.read_sql_query(text(q), _engine, params={"mid": modeling_id})


@st.cache_data(show_spinner=False, ttl=300)
def load_cluster_summary_temporal(
    _engine: Engine,
    modeling_id: str,
    bucket: str,
    include_noise: bool,
    min_cluster_size: int,
) -> pd.DataFrame:
    bucket_expr = _bucket_expr(bucket)
    noise_cond = _noise_cond(include_noise)

    q = f"""
    WITH base AS (
      SELECT
        m.cluster_id,
        m.tgl_submit,
        {bucket_expr} AS t_bucket
      FROM {SCHEMA}.{T_MEM} m
      WHERE m.modeling_id = CAST(:mid AS uuid)
        AND m.tgl_submit IS NOT NULL
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
    ORDER BY a.total_tickets DESC, a.cluster_id ASC
    """
    return pd.read_sql_query(
        text(q),
        _engine,
        params={"mid": modeling_id, "min_size": int(min_cluster_size)},
    )


@st.cache_data(show_spinner=False, ttl=300)
def load_heatmap(
    _engine: Engine,
    modeling_id: str,
    bucket: str,
    include_noise: bool,
    top_n: int,
    min_cluster_size: int,
) -> pd.DataFrame:
    bucket_expr = _bucket_expr(bucket)
    noise_cond = _noise_cond(include_noise)

    q = f"""
    WITH base AS (
      SELECT
        m.cluster_id,
        {bucket_expr} AS t_bucket
      FROM {SCHEMA}.{T_MEM} m
      WHERE m.modeling_id = CAST(:mid AS uuid)
        AND m.tgl_submit IS NOT NULL
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


@st.cache_data(show_spinner=False, ttl=300)
def load_cluster_trend(
    _engine: Engine,
    modeling_id: str,
    cluster_id: int,
    bucket: str,
    include_noise: bool,
) -> pd.DataFrame:
    bucket_expr = _bucket_expr(bucket)
    noise_cond = _noise_cond(include_noise)

    q = f"""
    SELECT
      {bucket_expr} AS t_bucket,
      COUNT(*) AS n_tickets
    FROM {SCHEMA}.{T_MEM} m
    WHERE m.modeling_id = CAST(:mid AS uuid)
      AND m.cluster_id = :cid
      AND m.tgl_submit IS NOT NULL
      {noise_cond}
    GROUP BY 1
    ORDER BY 1
    """
    return pd.read_sql_query(text(q), _engine, params={"mid": modeling_id, "cid": int(cluster_id)})


@st.cache_data(show_spinner=False, ttl=300)
def load_cluster_examples(
    _engine: Engine,
    modeling_id: str,
    cluster_id: int,
    limit: int = 30,
    join_text: bool = False,
) -> pd.DataFrame:
    if join_text:
        q = f"""
        SELECT
          m.incident_number,
          m.tgl_submit,
          m.site,
          m.modul,
          m.sub_modul,
          m.prob,
          m.outlier_score,
          s.text_semantic
        FROM {SCHEMA}.{T_MEM} m
        LEFT JOIN {SCHEMA}.{T_TEXT} s
          ON s.incident_number::text = m.incident_number::text
        WHERE m.modeling_id = CAST(:mid AS uuid)
          AND m.cluster_id = :cid
        ORDER BY m.prob DESC NULLS LAST, m.outlier_score ASC NULLS LAST, m.tgl_submit DESC NULLS LAST
        LIMIT :lim
        """
    else:
        q = f"""
        SELECT
          m.incident_number,
          m.tgl_submit,
          m.site,
          m.modul,
          m.sub_modul,
          m.prob,
          m.outlier_score
        FROM {SCHEMA}.{T_MEM} m
        WHERE m.modeling_id = CAST(:mid AS uuid)
          AND m.cluster_id = :cid
        ORDER BY m.prob DESC NULLS LAST, m.outlier_score ASC NULLS LAST, m.tgl_submit DESC NULLS LAST
        LIMIT :lim
        """
    return pd.read_sql_query(
        text(q),
        _engine,
        params={"mid": modeling_id, "cid": int(cluster_id), "lim": int(limit)},
    )


# =========================
# 🧭 UI
# =========================
st.title("🗓️ Analisis Temporal — HDBSCAN Semantik")
st.caption("Memetakan kemunculan cluster dari waktu ke waktu untuk melihat pola berulang (recurring), musiman, atau burst.")

engine = get_engine()
df_runs = load_runs(engine, limit=300)
if df_runs.empty:
    st.warning("Belum ada run HDBSCAN. Jalankan proses modeling terlebih dahulu.")
    st.stop()

text_available = has_table(engine, SCHEMA, T_TEXT)


def _fmt_run(i: int) -> str:
    row = df_runs.loc[i]
    mid = str(row["modeling_id"])
    model_hint = None
    try:
        pj = row.get("params_json")
        if isinstance(pj, dict):
            model_hint = pj.get("model_name") or pj.get("embedding_model") or pj.get("model")
    except Exception:
        model_hint = None

    hint = f" | {model_hint}" if model_hint else ""
    rows = int(row.get("n_rows") or 0)
    clus = int(row.get("n_clusters") or 0)
    noise = int(row.get("n_noise") or 0)
    return f"{mid}{hint} | rows={rows:,} | clusters={clus:,} | noise={noise:,}"


with st.sidebar:
    st.header("📌 Pilih Run")

    idx = st.selectbox(
        "Run (modeling_id | params | rows)",
        options=list(range(len(df_runs))),
        format_func=_fmt_run,
    )
    modeling_id = str(df_runs.loc[idx, "modeling_id"])

    st.divider()
    st.header("⚙️ Pengaturan Temporal")
    bucket = st.selectbox("Time bucket", options=["day", "week", "month"], index=1)
    include_noise = st.checkbox("Include noise (is_noise=true) dalam analisis", value=False)

    min_cluster_size = st.number_input(
        "Min cluster size (filter)",
        min_value=1,
        max_value=100000,
        value=10,
        step=1,
        help="Fokus ke cluster bermakna dan mengurangi cluster kecil/noise.",
    )

    top_n = st.slider("Top-N clusters untuk heatmap", min_value=5, max_value=80, value=20, step=1)

    st.divider()
    st.subheader("Heuristik Recurring")
    recurring_min_active = st.number_input(
        "Minimal bucket aktif agar dianggap recurring",
        min_value=2,
        max_value=500,
        value=6,
        step=1,
        help="Contoh: bucket=week, nilai 6 berarti cluster aktif minimal 6 minggu berbeda.",
    )

    st.divider()
    st.subheader("Contoh Tiket")
    example_limit = st.slider("Limit contoh tiket", 10, 200, 30, 10)
    join_text = st.checkbox(
        "Tampilkan text_semantic (jika tabel tersedia)", value=False, disabled=not text_available
    )
    if not text_available:
        st.caption("Tabel text tidak terdeteksi, preview hanya metadata/prob/outlier_score.")

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
    st.info("Tidak ada data trend (cek tgl_submit pada tabel members).")
else:
    ch_total = (
        alt.Chart(df_total)
        .mark_line(point=True)
        .encode(
            x=alt.X("t_bucket:T", title=f"time bucket ({bucket})"),
            y=alt.Y("n_tickets:Q", title="jumlah tiket"),
            tooltip=[alt.Tooltip("t_bucket:T"), alt.Tooltip("n_tickets:Q")],
        )
        .properties(height=300)
    )
    st.altair_chart(ch_total, use_container_width=True)

# Ringkasan cluster temporal
st.subheader("Ringkasan Temporal per Cluster")
df_sum = load_cluster_summary_temporal(
    engine,
    modeling_id,
    bucket=bucket,
    include_noise=include_noise,
    min_cluster_size=int(min_cluster_size),
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
    df_sum[
        ["cluster_id", "total_tickets", "n_active_buckets", "span_days", "first_seen", "last_seen", "is_recurring"]
    ],
    use_container_width=True,
    height=360,
)

# Heatmap
st.subheader("Heatmap Cluster × Waktu (Top-N)")
df_hm = load_heatmap(
    engine,
    modeling_id,
    bucket=bucket,
    include_noise=include_noise,
    top_n=int(top_n),
    min_cluster_size=int(min_cluster_size),
)
if df_hm.empty:
    st.info("Heatmap kosong (cek filter).")
else:
    ranked = df_sum.sort_values("total_tickets", ascending=False)["cluster_id"].astype(int).tolist()
    df_hm2 = df_hm.copy()
    df_hm2["cluster_id"] = df_hm2["cluster_id"].astype(int).astype(str)

    ch_hm = (
        alt.Chart(df_hm2)
        .mark_rect()
        .encode(
            x=alt.X("t_bucket:T", title="waktu"),
            y=alt.Y("cluster_id:O", title="cluster_id", sort=[str(x) for x in ranked]),
            color=alt.Color("n_tickets:Q", title="tickets"),
            tooltip=["cluster_id", alt.Tooltip("t_bucket:T"), "n_tickets"],
        )
        .properties(height=380)
    )
    st.altair_chart(ch_hm, use_container_width=True)

# Drilldown cluster
st.subheader("Drilldown Cluster (Temporal + Contoh Tiket)")
cluster_choices = df_sum["cluster_id"].astype(int).tolist()
selected_cluster = st.selectbox("Pilih cluster_id", options=cluster_choices, index=0)

df_ct = load_cluster_trend(engine, modeling_id, int(selected_cluster), bucket=bucket, include_noise=include_noise)
if not df_ct.empty:
    ch_ct = (
        alt.Chart(df_ct)
        .mark_line(point=True)
        .encode(
            x=alt.X("t_bucket:T", title=f"time bucket ({bucket})"),
            y=alt.Y("n_tickets:Q", title="jumlah tiket"),
            tooltip=[alt.Tooltip("t_bucket:T"), alt.Tooltip("n_tickets:Q")],
        )
        .properties(height=280)
    )
    st.altair_chart(ch_ct, use_container_width=True)
else:
    st.info("Trend cluster kosong (atau ter-filter oleh noise toggle).")

with st.expander("Contoh tiket (top by prob / outlier)", expanded=False):
    df_ex = load_cluster_examples(
        engine,
        modeling_id,
        int(selected_cluster),
        limit=int(example_limit),
        join_text=bool(join_text and text_available),
    )
    if df_ex.empty:
        st.info("Tidak ada contoh.")
    else:
        tab1, tab2 = st.tabs(["Tabel", "Preview Top 5"])
        with tab1:
            st.dataframe(df_ex, use_container_width=True, height=420)
        with tab2:
            top5 = df_ex.head(5)
            for _, rr in top5.iterrows():
                st.markdown(
                    f"**{rr['incident_number']}** | prob={rr.get('prob')} | outlier={rr.get('outlier_score')} | "
                    f"{'' if pd.isna(rr.get('tgl_submit')) else str(rr.get('tgl_submit'))}"
                )
                if "text_semantic" in top5.columns:
                    st.write(rr.get("text_semantic", ""))

st.caption(
    "Catatan: Heuristik recurring saat ini berbasis jumlah bucket aktif. "
    "Jika diperlukan, bisa ditambah metrik gap antar kemunculan (median gap) dan burstiness."
)
