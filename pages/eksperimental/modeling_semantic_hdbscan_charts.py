# pages/eksperimental/modeling_semantic_hdbscan_charts.py
# ============================================================
# Viewer Chart Hasil Modeling Semantik — HDBSCAN (Read-only)
#
# Sumber:
#   - lasis_djp.modeling_semantik_hdbscan_runs
#   - lasis_djp.modeling_semantik_hdbscan_clusters
#   - lasis_djp.modeling_semantik_hdbscan_members
#
# Catatan penting:
# - Jangan gunakan ":mid::uuid" (akan error). Gunakan uuid.UUID di Python.
# ============================================================

from __future__ import annotations

import uuid
import pandas as pd
import streamlit as st
import altair as alt
from sqlalchemy import create_engine, text

# =========================
# 🔐 Guard Login (opsional)
# =========================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login terlebih dahulu.")
    st.stop()

SCHEMA = "lasis_djp"
T_RUNS = "modeling_semantik_hdbscan_runs"
T_CLUSTERS = "modeling_semantik_hdbscan_clusters"
T_MEMBERS = "modeling_semantik_hdbscan_members"


# =========================
# 🔌 DB Connection
# =========================
@st.cache_resource(show_spinner=False)
def get_engine():
    cfg = st.secrets["connections"]["postgres"]
    url = (
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )
    return create_engine(url, pool_pre_ping=True, future=True)


def _as_uuid(modeling_id: str) -> uuid.UUID:
    """Parse modeling_id string -> UUID (raise dengan pesan yang jelas)."""
    try:
        return uuid.UUID(str(modeling_id))
    except Exception as e:
        raise ValueError(f"modeling_id bukan UUID valid: {modeling_id}") from e


# =========================
# ⛏️ Loaders
# =========================
@st.cache_data(show_spinner=False)
def load_runs(_engine) -> pd.DataFrame:
    sql = f"""
    SELECT
      modeling_id::text AS modeling_id,
      run_time,
      embedding_run_id::text AS embedding_run_id,
      n_rows, n_clusters, n_noise,
      silhouette, dbi,
      params_json,
      notes
    FROM {SCHEMA}.{T_RUNS}
    ORDER BY run_time DESC
    """
    with _engine.connect() as conn:
        return pd.read_sql(text(sql), conn)


@st.cache_data(show_spinner=False)
def load_clusters(_engine, modeling_id: str) -> pd.DataFrame:
    # ✅ FIX FINAL: bind param tanpa ::uuid, UUID diparse di Python
    mid = _as_uuid(modeling_id)
    sql = f"""
    SELECT
      modeling_id::text AS modeling_id,
      cluster_id,
      cluster_size,
      avg_prob,
      avg_outlier_score
    FROM {SCHEMA}.{T_CLUSTERS}
    WHERE modeling_id = :mid
    ORDER BY cluster_size DESC, cluster_id ASC
    """
    with _engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params={"mid": mid})


@st.cache_data(show_spinner=False)
def load_members(_engine, modeling_id: str, limit: int) -> pd.DataFrame:
    # ✅ FIX FINAL: bind param tanpa ::uuid, UUID diparse di Python
    mid = _as_uuid(modeling_id)
    limit = int(limit) if int(limit) > 0 else 0
    limit_sql = f"LIMIT {limit}" if limit > 0 else ""

    sql = f"""
    SELECT
      modeling_id::text AS modeling_id,
      cluster_id,
      is_noise,
      incident_number,
      tgl_submit,
      site,
      modul,
      sub_modul,
      prob,
      outlier_score
    FROM {SCHEMA}.{T_MEMBERS}
    WHERE modeling_id = :mid
    ORDER BY is_noise ASC, cluster_id ASC, incident_number ASC
    {limit_sql}
    """
    with _engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params={"mid": mid})


def fmt_int(x):
    try:
        return f"{int(x):,}"
    except Exception:
        return "-"


def fmt_float(x, nd=3):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "-"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "-"


# =========================
# 🧱 UI
# =========================
# st.set_page_config(page_title="Modeling Semantik — HDBSCAN (Charts)", layout="wide")
st.title("Modeling Semantik — HDBSCAN")
st.caption("Visualisasi hasil klasterisasi semantik dari tabel output (runs, clusters, members).")

engine = get_engine()
df_runs = load_runs(engine)

if df_runs.empty:
    st.warning(f"Belum ada data pada {SCHEMA}.{T_RUNS}. Jalankan runner HDBSCAN terlebih dahulu.")
    st.stop()

# Sidebar controls
with st.sidebar:
    st.header("Pengaturan")
    run_options = df_runs["modeling_id"].tolist()
    modeling_id = st.selectbox("Pilih modeling_id", run_options, index=0)

    top_k = st.slider("Top-K klaster", 5, 50, 20, 5)
    hist_bins = st.slider("Bin histogram", 5, 60, 25, 5)

    members_limit = st.number_input(
        "Limit Members (drilldown)",
        min_value=1_000,
        max_value=200_000,
        value=40_000,   # ✅ sesuai permintaan kamu sebelumnya
        step=5_000,
    )
    show_noise_members = st.checkbox("Tampilkan juga tiket noise pada drilldown", value=False)

row_run = df_runs.loc[df_runs["modeling_id"] == modeling_id].iloc[0]

n_rows = int(row_run["n_rows"])
n_clusters = int(row_run["n_clusters"])
n_noise = int(row_run["n_noise"])
noise_pct = (n_noise / n_rows * 100.0) if n_rows > 0 else 0.0
sil = row_run["silhouette"]
dbi = row_run["dbi"]

# Load clusters + members
df_clusters = load_clusters(engine, modeling_id)
df_members = load_members(engine, modeling_id, int(members_limit))

# =========================
# KPI
# =========================
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Tickets (valid)", fmt_int(n_rows))
c2.metric("Jumlah Klaster", fmt_int(n_clusters))
c3.metric("Noise", fmt_int(n_noise))
c4.metric("Noise (%)", f"{noise_pct:.1f}%")
c5.metric("Embedding Run", str(row_run["embedding_run_id"]))

st.write("")
st.info(
    f"Run time: {row_run['run_time']} | Silhouette: {fmt_float(sil)} | DBI: {fmt_float(dbi)} | Notes: {row_run.get('notes','')}"
)

# =========================
# Row 1: Noise vs Non-noise + Eval metrics
# =========================
left, right = st.columns([1, 1])

with left:
    st.subheader("Proporsi Noise vs Non-noise")
    df_noise = pd.DataFrame(
        {
            "kategori": ["Non-noise", "Noise"],
            "jumlah": [max(n_rows - n_noise, 0), n_noise],
        }
    )
    pie = (
        alt.Chart(df_noise)
        .mark_arc()
        .encode(
            theta=alt.Theta("jumlah:Q"),
            color=alt.Color("kategori:N"),
            tooltip=["kategori:N", alt.Tooltip("jumlah:Q", format=",")],
        )
        .properties(height=320)
    )
    st.altair_chart(pie, use_container_width=True)

with right:
    st.subheader("Metrik Evaluasi Internal (Non-noise)")
    df_eval = pd.DataFrame(
        {
            "metrik": ["Silhouette Score", "Davies–Bouldin Index (DBI)"],
            "nilai": [
                None if pd.isna(sil) else float(sil),
                None if pd.isna(dbi) else float(dbi),
            ],
        }
    )
    bar_eval = (
        alt.Chart(df_eval)
        .mark_bar()
        .encode(
            x=alt.X("metrik:N", sort=None, title=None),
            y=alt.Y("nilai:Q", title="Nilai"),
            tooltip=["metrik:N", "nilai:Q"],
        )
        .properties(height=320)
    )
    st.altair_chart(bar_eval, use_container_width=True)

st.divider()

# =========================
# Row 2: Cluster size distribution
# =========================
st.subheader("Distribusi Ukuran Klaster")

if df_clusters.empty:
    st.warning("Tidak ada klaster (n_clusters=0). Periksa parameter HDBSCAN (min_cluster_size/min_samples) atau kualitas embedding.")
else:
    cA, cB = st.columns([1.2, 1])

    with cA:
        st.caption("Histogram ukuran klaster (cluster_size).")
        hist = (
            alt.Chart(df_clusters)
            .mark_bar()
            .encode(
                x=alt.X("cluster_size:Q", bin=alt.Bin(maxbins=int(hist_bins)), title="Ukuran klaster (jumlah tiket)"),
                y=alt.Y("count():Q", title="Jumlah klaster"),
                tooltip=[alt.Tooltip("count():Q", title="Jumlah klaster")],
            )
            .properties(height=340)
        )
        st.altair_chart(hist, use_container_width=True)

    with cB:
        st.caption(f"Top-{top_k} klaster terbesar.")
        df_top = df_clusters.head(int(top_k)).copy()
        df_top["cluster_id"] = df_top["cluster_id"].astype(int)

        bar_top = (
            alt.Chart(df_top)
            .mark_bar()
            .encode(
                x=alt.X("cluster_size:Q", title="Ukuran klaster"),
                y=alt.Y("cluster_id:N", sort="-x", title="cluster_id"),
                tooltip=[
                    alt.Tooltip("cluster_id:N"),
                    alt.Tooltip("cluster_size:Q", format=","),
                    alt.Tooltip("avg_prob:Q", format=".3f"),
                    alt.Tooltip("avg_outlier_score:Q", format=".3f"),
                ],
            )
            .properties(height=340)
        )
        st.altair_chart(bar_top, use_container_width=True)

st.divider()

# =========================
# Table: clusters + drilldown
# =========================
st.subheader("Ringkasan Klaster & Drilldown Members")

tab1, tab2 = st.tabs(["Tabel Klaster", "Drilldown Members"])

with tab1:
    st.caption("Ringkasan setiap klaster (cluster_size, avg_prob, avg_outlier_score).")
    st.dataframe(df_clusters, use_container_width=True, height=420)

    csv_clusters = df_clusters.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download CSV (clusters)",
        data=csv_clusters,
        file_name=f"semantik_hdbscan_clusters_{modeling_id}.csv",
        mime="text/csv",
    )

with tab2:
    if df_clusters.empty:
        st.info("Tidak ada klaster untuk drilldown.")
    else:
        cluster_ids = df_clusters["cluster_id"].astype(int).tolist()
        sel_cluster = st.selectbox("Pilih cluster_id", cluster_ids, index=0)

        dfm = df_members.copy()
        if not show_noise_members:
            dfm = dfm[dfm["is_noise"] == False]  # noqa: E712

        dfm = dfm[dfm["cluster_id"].astype(int) == int(sel_cluster)].copy()

        st.write(
            f"Jumlah baris pada cluster_id={sel_cluster}: **{len(dfm):,}** "
            f"(dari limit {int(members_limit):,} rows yang dimuat)."
        )
        st.dataframe(dfm, use_container_width=True, height=420)

        csv_members = dfm.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV (members - cluster terpilih)",
            data=csv_members,
            file_name=f"semantik_hdbscan_members_{modeling_id}_cluster_{sel_cluster}.csv",
            mime="text/csv",
        )

with st.expander("Catatan interpretasi (untuk Bab IV)"):
    st.markdown(
        f"""
- **Jumlah klaster = {n_clusters:,}** menunjukkan struktur kelompok berbasis kemiripan makna berhasil terbentuk.
- **Noise = {n_noise:,} ({noise_pct:.1f}%)** menunjukkan HDBSCAN selektif dan tidak memaksakan seluruh tiket ke dalam klaster.
- **Silhouette = {fmt_float(sil)}** dan **DBI = {fmt_float(dbi)}** digunakan sebagai indikator evaluasi internal pada tiket non-noise.
"""
    )
