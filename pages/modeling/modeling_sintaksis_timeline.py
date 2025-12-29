# pages/timeline_single_run_clusters.py
# ======================================================
# Timeline Cluster untuk 1 Run (dropdown) + Cluster selector (kiri)
# - X axis: tgl_submit
# - Y axis: cluster_id (atau label)
# - Size  : cluster_size (atau count per hari)
#
# Data source:
# - lasis_djp.modeling_sintaksis_runs
# - lasis_djp.modeling_sintaksis_members
#
# Fix:
# - cache _engine
# - type-safe filter: modeling_id::text = :mid
# ======================================================

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ======================================================
# 🔐 Guard login
# ======================================================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login terlebih dahulu.")
    st.stop()


SCHEMA = "lasis_djp"
T_RUNS = "modeling_sintaksis_runs"
T_MEMBERS = "modeling_sintaksis_members"
NOISE_ID = -1


# ======================================================
# 🔌 DB Connection (secrets.toml)
# ======================================================
@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    cfg = st.secrets["connections"]["postgres"]
    url = (
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )
    return create_engine(url, pool_pre_ping=True)


# ======================================================
# 📦 Loaders
# ======================================================
@st.cache_data(show_spinner=False)
def load_runs(_engine: Engine, limit_runs: int = 200) -> pd.DataFrame:
    sql = f"""
    SELECT
      modeling_id::text AS modeling_id,
      run_time,
      approach,
      tfidf_run_id::text AS tfidf_run_id,
      threshold,
      window_days,
      knn_k,
      min_cluster_size,
      n_rows,
      n_clusters_recurring,
      n_noise_tickets
    FROM {SCHEMA}.{T_RUNS}
    ORDER BY run_time DESC
    LIMIT :lim
    """
    df = pd.read_sql(text(sql), _engine, params={"lim": int(limit_runs)})
    df["run_time"] = pd.to_datetime(df["run_time"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_members_for_run(_engine: Engine, modeling_id: str, include_noise: bool) -> pd.DataFrame:
    noise_clause = "" if include_noise else f"AND cluster_id <> {NOISE_ID}"
    sql = f"""
    SELECT
      modeling_id::text AS modeling_id,
      cluster_id,
      incident_number,
      tgl_submit,
      site,
      assignee,
      modul,
      sub_modul,
      is_recurring
    FROM {SCHEMA}.{T_MEMBERS}
    WHERE modeling_id::text = :mid
      {noise_clause}
    ORDER BY tgl_submit NULLS LAST
    """
    df = pd.read_sql(text(sql), _engine, params={"mid": str(modeling_id)})
    df["tgl_submit"] = pd.to_datetime(df["tgl_submit"], errors="coerce")
    return df


def make_chart_per_ticket(df: pd.DataFrame) -> alt.Chart:
    # df: incident-level
    # size: cluster_size (konstan per cluster)
    base = alt.Chart(df).mark_circle().encode(
        x=alt.X("tgl_submit:T", title="Tanggal Submit"),
        y=alt.Y("cluster_id:N", title="Cluster ID"),
        size=alt.Size("cluster_size:Q", legend=alt.Legend(title="Cluster size"),
                      scale=alt.Scale(range=[20, 700])),
        color=alt.Color("is_recurring:N", legend=alt.Legend(title="Recurring")),
        tooltip=[
            alt.Tooltip("cluster_id:N"),
            alt.Tooltip("incident_number:N"),
            alt.Tooltip("tgl_submit:T"),
            alt.Tooltip("cluster_size:Q"),
            alt.Tooltip("site:N"),
            alt.Tooltip("assignee:N"),
            alt.Tooltip("modul:N"),
            alt.Tooltip("sub_modul:N"),
        ],
    )
    return base.properties(height=560).interactive()


def make_chart_agg_per_day(df_day: pd.DataFrame) -> alt.Chart:
    # df_day: aggregated per (cluster_id, date)
    base = alt.Chart(df_day).mark_circle().encode(
        x=alt.X("day:T", title="Tanggal"),
        y=alt.Y("cluster_id:N", title="Cluster ID"),
        size=alt.Size("n_tickets:Q", legend=alt.Legend(title="Tiket per hari"),
                      scale=alt.Scale(range=[30, 1200])),
        color=alt.Color("is_recurring:N", legend=alt.Legend(title="Recurring")),
        tooltip=[
            alt.Tooltip("cluster_id:N"),
            alt.Tooltip("day:T", title="Tanggal"),
            alt.Tooltip("n_tickets:Q", title="Jumlah tiket"),
            alt.Tooltip("cluster_size:Q", title="Cluster size"),
        ],
    )
    return base.properties(height=560).interactive()


# ======================================================
# UI
# ======================================================
st.title("🫧 Timeline Cluster untuk 1 Run (Dropdown + Selector Cluster)")
st.caption(
    "Cluster yang sama bisa muncul di banyak waktu karena berisi banyak tiket dengan tanggal submit berbeda. "
    "Gunakan mode agregasi per hari jika titik terlalu padat."
)

engine = get_engine()
runs = load_runs(engine, limit_runs=200)

if runs.empty:
    st.warning("Tidak ada data pada modeling_sintaksis_runs.")
    st.stop()

runs = runs.copy()
runs["label"] = runs.apply(
    lambda r: (
        f"{str(r['run_time'])[:19]} | mid={r['modeling_id']} | "
        f"thr={r.get('threshold')} | win={r.get('window_days')} | k={r.get('knn_k')}"
    ),
    axis=1
)

pick = st.selectbox("Pilih 1 run", runs["label"].tolist(), index=0)
run_row = runs.loc[runs["label"] == pick].iloc[0]
modeling_id = str(run_row["modeling_id"])

# ===== Sidebar controls =====
st.sidebar.header("⚙️ Tampilan & Filter")
include_noise = st.sidebar.checkbox("Include noise (cluster_id = -1)", value=False)
only_recurring = st.sidebar.checkbox("Hanya recurring (is_recurring=1)", value=True)
mode = st.sidebar.radio("Mode plot", ["Aggregated per Hari", "Per Ticket"], index=0)
top_n_clusters = st.sidebar.slider("Ambil top-N cluster terbesar", 5, 300, 60, 5)
date_min = st.sidebar.date_input("Tanggal mulai (opsional)", value=None)
date_max = st.sidebar.date_input("Tanggal akhir (opsional)", value=None)

with st.spinner("Memuat members untuk run..."):
    mem = load_members_for_run(engine, modeling_id, include_noise=include_noise)

if mem.empty:
    st.warning("Tidak ada data members untuk run ini.")
    st.stop()

# date filter
if date_min is not None:
    mem = mem[mem["tgl_submit"] >= pd.Timestamp(date_min)]
if date_max is not None:
    mem = mem[mem["tgl_submit"] <= pd.Timestamp(date_max) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]

# recurring filter
mem["is_recurring"] = pd.to_numeric(mem["is_recurring"], errors="coerce").fillna(0).astype(int)
if only_recurring:
    mem = mem[mem["is_recurring"] == 1]

if mem.empty:
    st.warning("Data kosong setelah filter (tanggal/recurring).")
    st.stop()

# cluster size
cluster_sizes = (
    mem.groupby("cluster_id", as_index=False)
       .agg(cluster_size=("incident_number", "count"),
            min_time=("tgl_submit", "min"),
            max_time=("tgl_submit", "max"))
       .sort_values("cluster_size", ascending=False)
)

# ambil top-N (agar selector tidak kebanyakan)
cluster_sizes_top = cluster_sizes.head(int(top_n_clusters)).copy()

# ===== Layout 2 kolom: kiri selector cluster, kanan chart =====
left, right = st.columns([1, 3], vertical_alignment="top")

with left:
    st.subheader("📌 Pilih Cluster")
    options = cluster_sizes_top["cluster_id"].tolist()
    default_pick = options[: min(10, len(options))]  # default pilih 10 teratas
    picked_clusters = st.multiselect(
        "Cluster ID",
        options=options,
        default=default_pick,
    )

    st.markdown("---")
    st.write("Ringkas run:")
    st.write(f"- modeling_id: `{modeling_id}`")
    st.write(f"- approach: `{run_row.get('approach')}`")
    st.write(f"- threshold: `{run_row.get('threshold')}`")
    st.write(f"- window_days: `{run_row.get('window_days')}`")

    st.markdown("---")
    st.write("Top cluster (preview):")
    st.dataframe(cluster_sizes_top.head(12), use_container_width=True)

with right:
    st.subheader("📈 Timeline")
    if not picked_clusters:
        st.info("Pilih minimal 1 cluster di panel kiri.")
        st.stop()

    df = mem[mem["cluster_id"].isin(picked_clusters)].copy()
    df = df.dropna(subset=["tgl_submit"]).copy()

    # join cluster_size ke setiap baris
    df = df.merge(cluster_sizes[["cluster_id", "cluster_size", "min_time", "max_time"]],
                  on="cluster_id", how="left")

    # KPI
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clusters selected", df["cluster_id"].nunique())
    c2.metric("Tickets shown", len(df))
    c3.metric("Min time", str(df["tgl_submit"].min())[:19])
    c4.metric("Max time", str(df["tgl_submit"].max())[:19])

    if mode == "Per Ticket":
        chart = make_chart_per_ticket(df)
        st.altair_chart(chart, use_container_width=True)
    else:
        # Aggregated per day: (cluster, day) count -> size bubble
        df_day = df.copy()
        df_day["day"] = df_day["tgl_submit"].dt.floor("D")
        df_day = (
            df_day.groupby(["cluster_id", "day"], as_index=False)
                  .agg(n_tickets=("incident_number", "count"),
                       cluster_size=("cluster_size", "max"),
                       is_recurring=("is_recurring", "max"))
        )
        chart = make_chart_agg_per_day(df_day)
        st.altair_chart(chart, use_container_width=True)

    with st.expander("📋 Data detail (sample)"):
        st.dataframe(
            df[["cluster_id", "incident_number", "tgl_submit", "cluster_size", "site", "assignee", "modul", "sub_modul"]]
            .sort_values(["cluster_id", "tgl_submit"])
            .head(1000),
            use_container_width=True
        )
