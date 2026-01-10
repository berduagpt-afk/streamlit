# pages/modeling_semantik_hdbscan_viewer.py
# ============================================================
# Viewer Hasil Clustering Semantik — HDBSCAN (Read-only)
#
# Sumber:
# - lasis_djp.modeling_semantik_hdbscan_runs
# - lasis_djp.modeling_semantik_hdbscan_clusters
# - lasis_djp.modeling_semantik_hdbscan_members
# - lasis_djp.incident_semantik (text_semantic)
#
# Fitur:
# - Pilih modeling_id (dropdown)
# - KPI ringkas: tickets, clusters, noise, silhouette, DBI
# - Distribusi ukuran cluster (bar chart)
# - Tabel cluster
# - Drilldown cluster → member + text_semantic
# - Aman untuk kondisi kosong / semua noise
# ============================================================

from __future__ import annotations

import pandas as pd
import streamlit as st
import altair as alt
from sqlalchemy import create_engine, text


# ======================================================
# 🔐 Guard Login
# ======================================================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login terlebih dahulu.")
    st.stop()


# ======================================================
# ⚙️ KONSTANTA
# ======================================================
SCHEMA = "lasis_djp"

T_RUNS = "modeling_semantik_hdbscan_runs"
T_CLUSTERS = "modeling_semantik_hdbscan_clusters"
T_MEMBERS = "modeling_semantik_hdbscan_members"
T_SEMANTIK = "incident_semantik"


# ======================================================
# 🔌 DB CONNECTION
# ======================================================
@st.cache_resource(show_spinner=False)
def get_engine():
    cfg = st.secrets["connections"]["postgres"]
    return create_engine(
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}",
        pool_pre_ping=True,
        future=True,
    )


# ======================================================
# 📥 LOADERS
# ======================================================
@st.cache_data(show_spinner=False)
def load_runs() -> pd.DataFrame:
    eng = get_engine()
    q = f"""
        SELECT
            modeling_id::text,
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
    """
    with eng.connect() as c:
        return pd.read_sql(text(q), c)


@st.cache_data(show_spinner=False)
def load_clusters(modeling_id: str) -> pd.DataFrame:
    eng = get_engine()
    q = f"""
        SELECT
            cluster_id,
            cluster_size,
            avg_prob,
            avg_outlier_score
        FROM {SCHEMA}.{T_CLUSTERS}
        WHERE modeling_id = :mid
        ORDER BY cluster_size DESC
    """
    with eng.connect() as c:
        return pd.read_sql(text(q), c, params={"mid": modeling_id})


@st.cache_data(show_spinner=False)
def load_members(modeling_id: str, cluster_id: int, limit: int) -> pd.DataFrame:
    eng = get_engine()
    q = f"""
        SELECT
            m.cluster_id,
        -- m.is_noise,
            m.incident_number,
            m.tgl_submit,
            m.site,
            m.modul,
        --    m.sub_modul,
        --    m.prob,
        --    m.outlier_score,
            s.text_semantic
        FROM {SCHEMA}.{T_MEMBERS} m
        LEFT JOIN {SCHEMA}.{T_SEMANTIK} s
          ON s.incident_number = m.incident_number
        WHERE m.modeling_id = :mid
          AND m.cluster_id = :cid
        ORDER BY m.prob DESC NULLS LAST
        LIMIT :limit
    """
    with eng.connect() as c:
        return pd.read_sql(
            text(q),
            c,
            params={
                "mid": modeling_id,
                "cid": int(cluster_id),
                "limit": int(limit),
            },
        )


# ======================================================
# 🧾 UI HEADER
# ======================================================
st.title("🧠 Hasil Clustering Semantik — HDBSCAN")
st.caption(
    "Viewer read-only untuk eksplorasi hasil clustering semantik "
    "(SBERT / IndoBERT embedding + HDBSCAN)."
)


# ======================================================
# 📦 LOAD RUNS
# ======================================================
df_runs = load_runs()

if df_runs.empty:
    st.warning("Belum ada hasil clustering semantik.")
    st.stop()


# ======================================================
# 🧭 SIDEBAR
# ======================================================
with st.sidebar:
    st.header("⚙️ Parameter Viewer")

    modeling_id = st.selectbox(
        "Pilih Modeling ID",
        options=df_runs["modeling_id"].tolist(),
        format_func=lambda x: x[:8],
    )

    limit_members = st.number_input(
        "Limit member (drilldown)",
        min_value=100,
        max_value=40_000,
        value=1_000,
        step=500,
    )

    hide_noise = st.checkbox("Sembunyikan noise (-1)", value=True)


# ======================================================
# 📌 RUN INFO + KPI
# ======================================================
run = df_runs[df_runs["modeling_id"] == modeling_id].iloc[0]

c1, c2, c3 = st.columns(3)
c1.metric("Total Tiket", f"{int(run.n_rows):,}")
c2.metric("Jumlah Cluster", f"{int(run.n_clusters):,}")
c3.metric("Noise", f"{int(run.n_noise):,}")
# c4.metric("Silhouette", f"{run.silhouette:.3f}" if run.silhouette is not None else "—")
# c5.metric("DBI", f"{run.dbi:.3f}" if run.dbi is not None else "—")

with st.expander("ℹ️ Parameter Modeling (params_json)"):
    st.json(run.params_json)


# ======================================================
# 📊 DISTRIBUSI CLUSTER
# ======================================================
df_clusters = load_clusters(modeling_id)

if hide_noise:
    df_clusters = df_clusters[df_clusters["cluster_id"] != -1]

st.subheader("📊 Distribusi Ukuran Cluster")

if df_clusters.empty:
    st.info(
        "Tidak ada cluster yang dapat ditampilkan "
        "(kemungkinan semua tiket noise atau filter noise aktif)."
    )
else:
    # ✅ Top 20 cluster terbesar untuk grafik (tetap tampilkan tabel full di bawah)
    df_top20 = df_clusters.sort_values("cluster_size", ascending=False).head(20).copy()

    chart = (
        alt.Chart(df_top20)
        .mark_bar()
        .encode(
            x=alt.X("cluster_size:Q", title="Ukuran Cluster"),
            y=alt.Y("cluster_id:N", sort="-x", title="Cluster ID"),
            tooltip=["cluster_id", "cluster_size"],
        )
        .properties(height=420)
    )
    st.altair_chart(chart, use_container_width=True)

    st.caption(f"Grafik menampilkan **Top 20** cluster terbesar dari total **{len(df_clusters):,}** cluster (setelah filter).")
    st.dataframe(df_clusters, use_container_width=True)



# ======================================================
# 🔍 DRILLDOWN MEMBERS (PATCHED SAFE)
# ======================================================
st.subheader("🔍 Drilldown Member Cluster")

if df_clusters.empty:
    st.info("Tidak ada cluster untuk didrilldown.")
    st.stop()

cluster_options = sorted(
    [int(x) for x in df_clusters["cluster_id"].dropna().unique().tolist()]
)

if not cluster_options:
    st.info("Tidak ada cluster_id valid.")
    st.stop()

# default: cluster terbesar (baris pertama sudah sort desc)
default_cluster = int(df_clusters.iloc[0]["cluster_id"])
default_index = (
    cluster_options.index(default_cluster)
    if default_cluster in cluster_options
    else 0
)

cluster_sel = st.selectbox(
    "Pilih cluster",
    options=cluster_options,
    index=default_index,
)

df_members = load_members(
    modeling_id=modeling_id,
    cluster_id=cluster_sel,
    limit=int(limit_members),
)

st.caption(
    f"Menampilkan hingga {len(df_members):,} tiket "
    f"dari cluster {cluster_sel} (limit={int(limit_members):,})."
)

st.dataframe(
    df_members,
    use_container_width=True,
    column_config={
        "text_semantic": st.column_config.TextColumn(
            "text_semantic",
            width="large",
        )
    },
)
