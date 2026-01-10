# pages/modeling_semantik_temporal_viewer.py
from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from sqlalchemy import create_engine, text

# ======================================================
# 🔐 Guard Login (sesuaikan dengan sistemmu)
# ======================================================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login terlebih dahulu.")
    st.stop()

#st.set_page_config(page_title="Temporal Semantik Viewer", layout="wide")

SCHEMA = "lasis_djp"
T_MEMBERS = "modeling_semantik_temporal_members"
T_SUMMARY = "modeling_semantik_temporal_summary"

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

engine = get_engine()

# ======================================================
# ✅ Load option lists
# ======================================================
@st.cache_data(show_spinner=False, ttl=60)
def load_modeling_ids() -> pd.DataFrame:
    q = text(f"""
        SELECT DISTINCT modeling_id::text AS modeling_id
        FROM {SCHEMA}.{T_SUMMARY}
        ORDER BY modeling_id::text DESC
    """)
    return pd.read_sql(q, engine)

@st.cache_data(show_spinner=False, ttl=60)
def load_summary_for_modeling(modeling_id: str) -> pd.DataFrame:
    q = text(f"""
        SELECT
          modeling_id::text AS modeling_id,
          window_days,
          time_col,
          include_noise,
          eligible_rule,
          n_clusters_eligible,
          n_clusters_split,
          prop_clusters_split,
          n_clusters_stable,
          prop_clusters_stable,
          total_episodes,
          avg_episode_per_cluster,
          median_episode_per_cluster,
          run_time
        FROM {SCHEMA}.{T_SUMMARY}
        WHERE modeling_id = CAST(:modeling_id AS uuid)
        ORDER BY window_days ASC, time_col ASC, include_noise ASC
    """)
    return pd.read_sql(q, engine, params={"modeling_id": modeling_id})

@st.cache_data(show_spinner=False, ttl=60)
def load_members(modeling_id: str, window_days: int, time_col: str) -> pd.DataFrame:
    q = text(f"""
        SELECT
          modeling_id::text AS modeling_id,
          window_days,
          cluster_id,
          incident_number,
          time_col,
          event_time,
          site,
          assignee,
          modul,
          sub_modul,
          gap_days,
          temporal_cluster_no,
          temporal_cluster_id
        FROM {SCHEMA}.{T_MEMBERS}
        WHERE modeling_id = CAST(:modeling_id AS uuid)
          AND window_days = :window_days
          AND time_col = :time_col
        ORDER BY cluster_id ASC, temporal_cluster_no ASC, event_time ASC, incident_number ASC
    """)
    df = pd.read_sql(q, engine, params={"modeling_id": modeling_id, "window_days": int(window_days), "time_col": time_col})
    if not df.empty:
        df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    return df

# ======================================================
# 🎛️ Sidebar
# ======================================================
st.title("🧠📆 Viewer Evaluasi Temporal — Pendekatan Semantik (HDBSCAN)")

with st.sidebar:
    st.header("Parameter")

    mids = load_modeling_ids()
    if mids.empty:
        st.warning("Belum ada data summary temporal semantik. Jalankan script run_temporal_semantik.py dulu.")
        st.stop()

    modeling_id = st.selectbox("modeling_id", mids["modeling_id"].tolist(), index=0)

    summary_all = load_summary_for_modeling(modeling_id)
    if summary_all.empty:
        st.warning("Summary kosong untuk modeling_id ini.")
        st.stop()

    # combo selector: time_col + include_noise + window_days
    time_cols = sorted(summary_all["time_col"].unique().tolist())
    time_col = st.selectbox("time_col", time_cols, index=0)

    noises = sorted(summary_all[summary_all["time_col"] == time_col]["include_noise"].unique().tolist())
    include_noise = st.selectbox("include_noise", noises, index=0)

    wins = sorted(summary_all[(summary_all["time_col"] == time_col) & (summary_all["include_noise"] == include_noise)]["window_days"].unique().tolist())
    window_days = st.selectbox("window_days", wins, index=0)

    st.divider()
    st.subheader("Filter Drilldown")
    max_rows = st.slider("Limit rows (drilldown)", 1_000, 40_000, 10_000, step=1_000)
    only_split_clusters = st.checkbox("Hanya cluster split (episode > 1)", value=False)

    date_filter_on = st.checkbox("Filter rentang tanggal", value=False)
    filter_site_on = st.checkbox("Filter site", value=False)
    filter_modul_on = st.checkbox("Filter modul", value=False)

# ======================================================
# 📥 Data
# ======================================================
df = load_members(modeling_id, int(window_days), str(time_col))
if df.empty:
    st.info("Tidak ada data temporal members untuk parameter yang dipilih.")
    st.stop()

# filter candidates
sites = sorted([x for x in df["site"].dropna().unique().tolist() if str(x).strip() != ""])
moduls = sorted([x for x in df["modul"].dropna().unique().tolist() if str(x).strip() != ""])

with st.sidebar:
    date_range = None
    if date_filter_on:
        dmin = df["event_time"].min()
        dmax = df["event_time"].max()
        if pd.isna(dmin) or pd.isna(dmax):
            st.warning("event_time invalid/kosong, filter tanggal dinonaktifkan.")
            date_filter_on = False
        else:
            d1, d2 = st.date_input(
                "Rentang tanggal (event_time)",
                value=(dmin.date(), dmax.date()),
                min_value=dmin.date(),
                max_value=dmax.date(),
            )
            date_range = (pd.to_datetime(d1), pd.to_datetime(d2) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))

    sel_sites = None
    if filter_site_on and sites:
        sel_sites = st.multiselect("Pilih site", sites, default=sites[: min(5, len(sites))])

    sel_moduls = None
    if filter_modul_on and moduls:
        sel_moduls = st.multiselect("Pilih modul", moduls, default=moduls[: min(5, len(moduls))])

# apply filters (for analytics + drilldown)
df_f = df.copy()
if date_filter_on and date_range is not None:
    df_f = df_f[(df_f["event_time"] >= date_range[0]) & (df_f["event_time"] <= date_range[1])]
if filter_site_on and sel_sites:
    df_f = df_f[df_f["site"].isin(sel_sites)]
if filter_modul_on and sel_moduls:
    df_f = df_f[df_f["modul"].isin(sel_moduls)]

# ======================================================
# 🧾 Summary (selected row)
# ======================================================
row = summary_all[
    (summary_all["window_days"] == int(window_days))
    & (summary_all["time_col"] == str(time_col))
    & (summary_all["include_noise"] == bool(include_noise))
]
row_dict = row.iloc[0].to_dict() if not row.empty else None

st.subheader("Ringkasan Run (Summary)")

cA, cB = st.columns([1.2, 1])
with cA:
    st.dataframe(summary_all, use_container_width=True, hide_index=True)

with cB:
    if row_dict:
        k1, k2, k3 = st.columns(3)
        k1.metric("Cluster eligible", f"{int(row_dict['n_clusters_eligible']):,}")
        k2.metric("Cluster split", f"{int(row_dict['n_clusters_split']):,}", f"{row_dict['prop_clusters_split']*100:.1f}%")
        k3.metric("Cluster stabil", f"{int(row_dict['n_clusters_stable']):,}", f"{row_dict['prop_clusters_stable']*100:.1f}%")
        k4, k5, k6 = st.columns(3)
        k4.metric("Total episode", f"{int(row_dict['total_episodes']):,}")
        k5.metric("Avg ep/cluster", f"{row_dict['avg_episode_per_cluster']:.2f}")
        k6.metric("Median ep/cluster", f"{row_dict['median_episode_per_cluster']:.2f}")
        st.caption(f"eligible_rule: **{row_dict['eligible_rule']}** | run_time: {row_dict['run_time']}")
    else:
        st.info("Baris summary terpilih tidak ditemukan (cek parameter).")

# ======================================================
# 🧩 Cluster-level aggregation
# ======================================================
cluster_agg = (
    df_f.groupby(["cluster_id"], as_index=False)
      .agg(
          n_member=("incident_number", "count"),
          n_episode=("temporal_cluster_no", "nunique"),
          min_date=("event_time", "min"),
          max_date=("event_time", "max"),
          n_site=("site", "nunique"),
          n_modul=("modul", "nunique"),
      )
)
cluster_agg["span_days"] = (cluster_agg["max_date"] - cluster_agg["min_date"]).dt.days

if only_split_clusters:
    cluster_agg = cluster_agg[cluster_agg["n_episode"] > 1].copy()

# ======================================================
# 📊 Charts
# ======================================================
st.subheader("Analitik Temporal Split")

cc1, cc2 = st.columns(2)

with cc1:
    tmp = pd.DataFrame({
        "status": ["Split (>1 episode)", "Stabil (1 episode)"],
        "n_cluster": [
            int((cluster_agg["n_episode"] > 1).sum()),
            int((cluster_agg["n_episode"] == 1).sum()),
        ],
    })
    chart1 = (
        alt.Chart(tmp)
        .mark_bar()
        .encode(
            x=alt.X("status:N", title=None, sort=None),
            y=alt.Y("n_cluster:Q", title="Jumlah Cluster"),
            tooltip=["status:N", "n_cluster:Q"],
        )
        .properties(height=280)
    )
    st.altair_chart(chart1, use_container_width=True)

with cc2:
    dist = cluster_agg.copy()
    dist["n_episode"] = dist["n_episode"].astype(int)
    dist2 = dist.groupby("n_episode", as_index=False).agg(n_cluster=("cluster_id", "count"))
    chart2 = (
        alt.Chart(dist2)
        .mark_bar()
        .encode(
            x=alt.X("n_episode:O", title="Jumlah Episode per Cluster"),
            y=alt.Y("n_cluster:Q", title="Jumlah Cluster"),
            tooltip=["n_episode:O", "n_cluster:Q"],
        )
        .properties(height=280)
    )
    st.altair_chart(chart2, use_container_width=True)

st.markdown("**Top Cluster Paling Terbelah (episode terbanyak)**")
topn = (
    cluster_agg.sort_values(["n_episode", "n_member"], ascending=[False, False])
    .head(20)
    .reset_index(drop=True)
)
top_chart = (
    alt.Chart(topn)
    .mark_bar()
    .encode(
        y=alt.Y("cluster_id:O", sort="-x", title="cluster_id"),
        x=alt.X("n_episode:Q", title="Jumlah Episode"),
        tooltip=["cluster_id:O", "n_episode:Q", "n_member:Q", "span_days:Q", "min_date:T", "max_date:T"],
    )
    .properties(height=420)
)
st.altair_chart(top_chart, use_container_width=True)

# ======================================================
# 📋 Cluster table + Drilldown
# ======================================================
st.subheader("Tabel Cluster (ringkas)")
show_cols = ["cluster_id", "n_member", "n_episode", "span_days", "min_date", "max_date", "n_site", "n_modul"]
st.dataframe(
    cluster_agg[show_cols].sort_values(["n_episode", "n_member"], ascending=[False, False]),
    use_container_width=True,
    hide_index=True,
)

cluster_ids = cluster_agg.sort_values(["n_episode", "n_member"], ascending=[False, False])["cluster_id"].tolist()
if not cluster_ids:
    st.info("Tidak ada cluster sesuai filter.")
    st.stop()

sel_cluster = st.selectbox("Pilih cluster_id untuk drilldown", cluster_ids, index=0)

episodes = (
    df_f[df_f["cluster_id"] == sel_cluster][["temporal_cluster_id", "temporal_cluster_no"]]
    .drop_duplicates()
    .sort_values("temporal_cluster_no")
)
episode_ids = episodes["temporal_cluster_id"].tolist()

st.subheader("Drilldown Episode (temporal_cluster_id)")
colL, colR = st.columns([1, 2])

with colL:
    sel_episode = st.selectbox("Pilih episode", episode_ids, index=0)
    st.write("Jumlah episode:", len(episode_ids))
    ep_df = df_f[df_f["temporal_cluster_id"] == sel_episode]
    st.metric("Member (episode)", f"{len(ep_df):,}")
    if not ep_df.empty:
        st.write("Periode:", ep_df["event_time"].min(), "—", ep_df["event_time"].max())
        st.write("Modul unik:", ep_df["modul"].nunique())

with colR:
    drill = df_f[df_f["temporal_cluster_id"] == sel_episode].copy()
    drill = drill.sort_values(["event_time", "incident_number"]).head(int(max_rows))

    st.dataframe(
        drill[
            [
                "incident_number", "event_time", "gap_days",
                "site", "assignee", "modul", "sub_modul",
                "cluster_id", "temporal_cluster_no", "temporal_cluster_id",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    csv = drill.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download CSV (episode terpilih)",
        data=csv,
        file_name=f"semantik_temporal_members_model_{modeling_id}_w{window_days}_{time_col}_cluster{sel_cluster}.csv",
        mime="text/csv",
    )

with st.expander("Catatan interpretasi (untuk Bab Evaluasi)"):
    st.markdown(
        f"""
- **Cluster split** berarti cluster semantik yang serupa secara makna ternyata muncul pada beberapa episode terpisah
  ketika jeda antar tiket melebihi **window {window_days} hari**.
- **Cluster stabil** berarti tiket-tiket cluster muncul relatif kontinu dalam rentang waktu observasi (hanya 1 episode).
- Parameter **eligible_rule** yang digunakan: **{row_dict['eligible_rule'] if row_dict else '-'}**.
        """
    )
