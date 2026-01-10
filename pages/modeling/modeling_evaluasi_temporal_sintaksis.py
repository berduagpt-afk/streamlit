# pages/modeling_sintaksis_temporal_viewer.py
from __future__ import annotations

import json
import math
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
import altair as alt
from sqlalchemy import create_engine, text

# ======================================================
# 🔐 Guard Login (opsional, sesuaikan dengan sistemmu)
# ======================================================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login terlebih dahulu.")
    st.stop()

#st.set_page_config(page_title="Temporal Split Viewer", layout="wide")

# ======================================================
# ⚙️ Konstanta DB
# ======================================================
SCHEMA = "lasis_djp"
T_TEMP_MEMBERS = "modeling_sintaksis_temporal_members"
T_TEMP_SUMMARY = "modeling_sintaksis_temporal_summary"

# ======================================================
# 🔌 Database Connection
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
# 🧰 Helper: Load options
# ======================================================
@st.cache_data(show_spinner=False, ttl=60)
def load_job_ids() -> pd.DataFrame:
    q = text(f"""
        SELECT DISTINCT job_id::text AS job_id
        FROM {SCHEMA}.{T_TEMP_MEMBERS}
        ORDER BY job_id::text DESC
    """)
    return pd.read_sql(q, engine)

@st.cache_data(show_spinner=False, ttl=60)
def load_modeling_ids(job_id: str) -> pd.DataFrame:
    q = text(f"""
        SELECT DISTINCT modeling_id::text AS modeling_id
        FROM {SCHEMA}.{T_TEMP_MEMBERS}
        WHERE job_id = CAST(:job_id AS uuid)
        ORDER BY modeling_id::text DESC
    """)
    return pd.read_sql(q, engine, params={"job_id": job_id})

@st.cache_data(show_spinner=False, ttl=60)
def load_windows(job_id: str, modeling_id: str) -> pd.DataFrame:
    q = text(f"""
        SELECT DISTINCT window_days
        FROM {SCHEMA}.{T_TEMP_MEMBERS}
        WHERE job_id = CAST(:job_id AS uuid)
          AND modeling_id = CAST(:modeling_id AS uuid)
        ORDER BY window_days ASC
    """)
    return pd.read_sql(q, engine, params={"job_id": job_id, "modeling_id": modeling_id})

# ======================================================
# 🧾 Load summary & members
# ======================================================
@st.cache_data(show_spinner=False, ttl=60)
def load_summary(job_id: str, modeling_id: str) -> pd.DataFrame:
    q = text(f"""
        SELECT
          job_id::text AS job_id,
          modeling_id::text AS modeling_id,
          window_days,
          n_clusters_eligible,
          n_clusters_split,
          prop_clusters_split,
          n_clusters_stable,
          prop_clusters_stable,
          total_episodes,
          avg_episode_per_cluster,
          median_episode_per_cluster,
          run_time
        FROM {SCHEMA}.{T_TEMP_SUMMARY}
        WHERE job_id = CAST(:job_id AS uuid)
          AND modeling_id = CAST(:modeling_id AS uuid)
        ORDER BY window_days ASC
    """)
    return pd.read_sql(q, engine, params={"job_id": job_id, "modeling_id": modeling_id})

@st.cache_data(show_spinner=False, ttl=60)
def load_members_base(job_id: str, modeling_id: str, window_days: int) -> pd.DataFrame:
    q = text(f"""
        SELECT
          job_id::text AS job_id,
          modeling_id::text AS modeling_id,
          window_days,
          cluster_id,
          incident_number,
          tgl_submit,
          site,
          assignee,
          modul,
          sub_modul,
          gap_days,
          temporal_cluster_no,
          temporal_cluster_id
        FROM {SCHEMA}.{T_TEMP_MEMBERS}
        WHERE job_id = CAST(:job_id AS uuid)
          AND modeling_id = CAST(:modeling_id AS uuid)
          AND window_days = :window_days
        ORDER BY cluster_id ASC, temporal_cluster_no ASC, tgl_submit ASC, incident_number ASC
    """)
    df = pd.read_sql(
        q, engine,
        params={"job_id": job_id, "modeling_id": modeling_id, "window_days": int(window_days)}
    )
    if not df.empty:
        df["tgl_submit"] = pd.to_datetime(df["tgl_submit"], errors="coerce")
    return df

# ======================================================
# 🎛️ Sidebar Controls
# ======================================================
st.title("📆 Viewer Temporal Split (Sessionization) — Clustering Sintaksis")

with st.sidebar:
    st.header("Parameter Run")

    jobs = load_job_ids()
    if jobs.empty:
        st.warning("Belum ada data pada tabel temporal. Jalankan script sessionization dulu.")
        st.stop()

    job_id = st.selectbox("job_id", jobs["job_id"].tolist(), index=0)

    mids = load_modeling_ids(job_id)
    if mids.empty:
        st.warning("Tidak ditemukan modeling_id untuk job_id ini pada tabel temporal.")
        st.stop()

    modeling_id = st.selectbox("modeling_id", mids["modeling_id"].tolist(), index=0)

    wins = load_windows(job_id, modeling_id)
    if wins.empty:
        st.warning("Tidak ditemukan window_days untuk kombinasi job_id & modeling_id ini.")
        st.stop()

    window_days = st.selectbox("window_days", wins["window_days"].tolist(), index=0)

    st.divider()
    st.subheader("Filter & Drilldown")

    max_rows = st.slider("Limit drilldown rows", 1_000, 40_000, 10_000, step=1_000)
    only_split_clusters = st.checkbox("Hanya tampilkan cluster yang split (episode > 1)", value=False)

    # Filter tanggal
    date_filter_on = st.checkbox("Aktifkan filter rentang tanggal", value=False)
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None

    # Filter dimensi
    filter_site_on = st.checkbox("Filter site", value=False)
    filter_modul_on = st.checkbox("Filter modul", value=False)

# ======================================================
# 📥 Data Load
# ======================================================
summary_all = load_summary(job_id, modeling_id)
df = load_members_base(job_id, modeling_id, int(window_days))

if df.empty:
    st.info("Tidak ada data members untuk parameter yang dipilih.")
    st.stop()

# derive per-cluster episodes
cluster_agg = (
    df.groupby(["cluster_id"], as_index=False)
      .agg(
          n_member=("incident_number", "count"),
          n_episode=("temporal_cluster_no", "nunique"),
          min_date=("tgl_submit", "min"),
          max_date=("tgl_submit", "max"),
          n_site=("site", "nunique"),
          n_modul=("modul", "nunique"),
      )
)
cluster_agg["span_days"] = (cluster_agg["max_date"] - cluster_agg["min_date"]).dt.days

if only_split_clusters:
    cluster_agg = cluster_agg[cluster_agg["n_episode"] > 1].copy()

# sidebar dynamic filter options
sites = sorted([x for x in df["site"].dropna().unique().tolist() if str(x).strip() != ""])
moduls = sorted([x for x in df["modul"].dropna().unique().tolist() if str(x).strip() != ""])

with st.sidebar:
    if date_filter_on:
        dmin = df["tgl_submit"].min()
        dmax = df["tgl_submit"].max()
        if pd.isna(dmin) or pd.isna(dmax):
            st.warning("tgl_submit kosong/invalid, filter tanggal dinonaktifkan otomatis.")
            date_filter_on = False
        else:
            d1, d2 = st.date_input(
                "Rentang tanggal (tgl_submit)",
                value=(dmin.date(), dmax.date()),
                min_value=dmin.date(),
                max_value=dmax.date(),
            )
            date_range = (pd.to_datetime(d1), pd.to_datetime(d2) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))

    sel_sites = None
    if filter_site_on:
        sel_sites = st.multiselect("Pilih site", sites, default=sites[: min(5, len(sites))])

    sel_moduls = None
    if filter_modul_on:
        sel_moduls = st.multiselect("Pilih modul", moduls, default=moduls[: min(5, len(moduls))])

# apply filters to df (for drilldown and some charts)
df_f = df.copy()
if date_filter_on and date_range is not None:
    df_f = df_f[(df_f["tgl_submit"] >= date_range[0]) & (df_f["tgl_submit"] <= date_range[1])]

if filter_site_on and sel_sites:
    df_f = df_f[df_f["site"].isin(sel_sites)]

if filter_modul_on and sel_moduls:
    df_f = df_f[df_f["modul"].isin(sel_moduls)]

# ======================================================
# 🧾 KPI Summary
# ======================================================
st.subheader("Ringkasan Run (Summary)")

colA, colB = st.columns([1.2, 1])
with colA:
    if summary_all.empty:
        st.warning("Tabel summary kosong. (opsional) Pastikan script mengisi modeling_sintaksis_temporal_summary.")
    else:
        st.dataframe(
            summary_all,
            use_container_width=True,
            hide_index=True
        )

with colB:
    # KPI untuk window terpilih, fallback jika summary kosong
    row = None
    if not summary_all.empty:
        row = summary_all[summary_all["window_days"] == int(window_days)]
        row = row.iloc[0].to_dict() if not row.empty else None

    if row:
        k1, k2, k3 = st.columns(3)
        k1.metric("Cluster eligible", f"{int(row['n_clusters_eligible']):,}")
        k2.metric("Cluster split", f"{int(row['n_clusters_split']):,}", f"{row['prop_clusters_split']*100:.1f}%")
        k3.metric("Total episode", f"{int(row['total_episodes']):,}")
        k4, k5 = st.columns(2)
        k4.metric("Avg episode/cluster", f"{row['avg_episode_per_cluster']:.2f}")
        k5.metric("Median episode/cluster", f"{row['median_episode_per_cluster']:.2f}")
    else:
        k1, k2, k3 = st.columns(3)
        k1.metric("Rows (members)", f"{len(df_f):,}")
        k2.metric("Clusters", f"{df_f['cluster_id'].nunique():,}")
        k3.metric("Episodes", f"{df_f['temporal_cluster_id'].nunique():,}")

# ======================================================
# 📊 Charts
# ======================================================
st.subheader("Analitik Temporal Split")

c1, c2 = st.columns(2)

with c1:
    # Pie-ish bar: split vs stable (pakai summary window terpilih kalau ada)
    if row:
        split_val = int(row["n_clusters_split"])
        stable_val = int(row["n_clusters_stable"])
        tmp = pd.DataFrame({
            "status": ["Split (>1 episode)", "Stabil (1 episode)"],
            "n_cluster": [split_val, stable_val]
        })
    else:
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

with c2:
    # Distribusi jumlah episode per cluster (histogram / bar count)
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

# Top split clusters
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
# 🧩 Cluster-level table + selection
# ======================================================
st.subheader("Tabel Cluster (ringkas)")

st.caption("Klik/seleksi cluster lewat dropdown di bawah untuk melihat episode & anggota tiket per episode.")

# table view
show_cols = ["cluster_id", "n_member", "n_episode", "span_days", "min_date", "max_date", "n_site", "n_modul"]
st.dataframe(cluster_agg[show_cols].sort_values(["n_episode", "n_member"], ascending=[False, False]),
             use_container_width=True, hide_index=True)

# selector
cluster_ids = cluster_agg.sort_values(["n_episode", "n_member"], ascending=[False, False])["cluster_id"].tolist()
if not cluster_ids:
    st.info("Tidak ada cluster sesuai filter.")
    st.stop()

sel_cluster = st.selectbox("Pilih cluster_id untuk drilldown", cluster_ids, index=0)

# list episode IDs for selected cluster
episodes = (
    df_f[df_f["cluster_id"] == sel_cluster][["temporal_cluster_id", "temporal_cluster_no"]]
    .drop_duplicates()
    .sort_values("temporal_cluster_no")
)
episode_ids = episodes["temporal_cluster_id"].tolist()

# ======================================================
# 🔎 Episode drilldown
# ======================================================
st.subheader("Drilldown Episode (temporal_cluster_id)")

colL, colR = st.columns([1, 2])

with colL:
    sel_episode = st.selectbox("Pilih episode", episode_ids, index=0)
    st.write("Jumlah episode:", len(episode_ids))

    # quick stats episode size
    ep_df = df_f[df_f["temporal_cluster_id"] == sel_episode]
    st.metric("Member (episode)", f"{len(ep_df):,}")
    if not ep_df.empty:
        st.write("Periode:", ep_df["tgl_submit"].min(), "—", ep_df["tgl_submit"].max())
        st.write("Modul unik:", ep_df["modul"].nunique())

with colR:
    drill = df_f[df_f["temporal_cluster_id"] == sel_episode].copy()
    drill = drill.sort_values(["tgl_submit", "incident_number"]).head(int(max_rows))

    st.dataframe(
        drill[
            [
                "incident_number", "tgl_submit", "gap_days",
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
        file_name=f"temporal_members_job_{job_id}_model_{modeling_id}_w{window_days}_cluster{sel_cluster}.csv",
        mime="text/csv",
    )

# ======================================================
# 📌 Info tambahan
# ======================================================
with st.expander("Catatan interpretasi (untuk Bab Evaluasi)"):
    st.markdown(
        """
- **Cluster split** berarti anggota cluster yang secara sintaksis mirip ternyata **tidak selalu kontinu secara waktu**; ada jeda yang melebihi `window_days` sehingga dipisah menjadi beberapa episode.
- **Proporsi split yang tinggi** pada window kecil (mis. 7 hari) mengindikasikan pola kemunculan yang terfragmentasi (kejadian berulang tetapi berjeda).
- **Proporsi split yang menurun** pada window besar (mis. 30 hari) mengindikasikan bahwa pada skala bulanan, kejadian cenderung terlihat lebih “kontinu”.
        """
    )
