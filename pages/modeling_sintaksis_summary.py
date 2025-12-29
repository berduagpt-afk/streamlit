# pages/modeling_sintaksis_summary.py
# ======================================================
# SUMMARY Hasil Modeling Sintaksis (ALL MODUL) — Read-only dari DB
# Membaca: lasis_djp.modeling_runs, lasis_djp.cluster_summary
# Output: ringkasan per modul (jumlah cluster, threshold run, statistik ukuran cluster, dll)
# ======================================================

import json
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sqlalchemy import create_engine, text

# ======================================================
# 🔐 Guard login
# ======================================================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login terlebih dahulu.")
    st.stop()

SCHEMA = "lasis_djp"
T_RUNS = "modeling_runs"
T_SUMMARY = "cluster_summary"

# ======================================================
# 🔌 DB Connection
# ======================================================
@st.cache_resource(show_spinner=False)
def get_engine():
    cfg = st.secrets["connections"]["postgres"]
    url = (
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )
    return create_engine(url, pool_pre_ping=True)

ENGINE = get_engine()

# ======================================================
# 🧠 Helpers
# ======================================================
def _safe_json_load(x):
    """params_json bisa dict / string JSON / None"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return {}
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return {}
        try:
            return json.loads(s)
        except Exception:
            return {"_raw": s}
    return {"_raw": str(x)}

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")

def pick_param(params: dict, keys: list[str], default=None):
    """Ambil nilai dari beberapa kandidat key (buat jaga-jaga beda nama key)."""
    for k in keys:
        if k in params:
            return params.get(k)
    return default

@st.cache_data(show_spinner=False)
def load_runs(limit: int = 500) -> pd.DataFrame:
    sql = f"""
    SELECT run_id, run_time, approach, params_json, data_range, notes
    FROM {SCHEMA}.{T_RUNS}
    ORDER BY run_time DESC
    LIMIT :limit
    """
    return pd.read_sql(text(sql), ENGINE, params={"limit": limit})

@st.cache_data(show_spinner=False)
def load_summary(run_id: str) -> pd.DataFrame:
    sql = f"""
    SELECT
        run_id, cluster_id, modul, window_start, window_end,
        n_tickets, first_seen, last_seen,
        representative_incident, representative_text, top_terms, metrics_json
    FROM {SCHEMA}.{T_SUMMARY}
    WHERE run_id = :run_id
    """
    df = pd.read_sql(text(sql), ENGINE, params={"run_id": run_id})
    if df.empty:
        return df

    # typing aman
    for c in ["window_start", "window_end", "first_seen", "last_seen"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["n_tickets"] = pd.to_numeric(df["n_tickets"], errors="coerce").fillna(0).astype(int)
    df["modul"] = df["modul"].fillna("").astype(str)
    df["cluster_id"] = df["cluster_id"].astype(str)
    df["top_terms"] = df["top_terms"].fillna("").astype(str)
    return df

def build_modul_summary(df_summary: pd.DataFrame) -> pd.DataFrame:
    """Ringkasan per modul dari cluster_summary"""
    if df_summary.empty:
        return pd.DataFrame()

    g = df_summary.groupby("modul", dropna=False)

    def p90(x):
        x = pd.Series(x).dropna()
        if len(x) == 0:
            return np.nan
        return float(np.quantile(x, 0.90))

    out = (
        g.agg(
            n_clusters=("cluster_id", "count"),
            total_tickets=("n_tickets", "sum"),
            avg_cluster_size=("n_tickets", "mean"),
            median_cluster_size=("n_tickets", "median"),
            max_cluster_size=("n_tickets", "max"),
            min_cluster_size=("n_tickets", "min"),
            p90_cluster_size=("n_tickets", p90),
            first_seen_min=("first_seen", "min"),
            last_seen_max=("last_seen", "max"),
            window_start_min=("window_start", "min"),
            window_end_max=("window_end", "max"),
        )
        .reset_index()
    )

    # indikator tambahan yang sering dipakai untuk narasi tesis
    out["large_clusters_ge_50"] = g["n_tickets"].apply(lambda s: int((s >= 50).sum())).values
    out["large_clusters_ge_100"] = g["n_tickets"].apply(lambda s: int((s >= 100).sum())).values

    # rapikan tipe
    for c in ["avg_cluster_size", "median_cluster_size", "p90_cluster_size"]:
        out[c] = out[c].fillna(0).astype(float)

    out = out.sort_values(["n_clusters", "total_tickets"], ascending=[False, False])
    return out

# ======================================================
# 🎛️ Sidebar
# ======================================================
st.sidebar.header("📌 Summary Modeling (All Modul)")

runs = load_runs()
if runs.empty:
    st.warning(f"Tabel {SCHEMA}.{T_RUNS} kosong. Jalankan proses modeling terlebih dahulu.")
    st.stop()

run_label = runs.apply(lambda r: f"{r['run_time']} | {r['approach']} | {r['run_id']}", axis=1)
run_map = dict(zip(run_label, runs["run_id"]))

selected_run_label = st.sidebar.selectbox("Pilih run_id", run_label.tolist(), index=0)
run_id = run_map[selected_run_label]

r = runs[runs["run_id"] == run_id].iloc[0].to_dict()
params = _safe_json_load(r.get("params_json"))

# Ambil parameter kunci (toleran beda nama key)
cos_thr = pick_param(params, ["cosine_threshold", "threshold", "sim_threshold", "cos_thr", "cosine_thr"], default=None)
window_days = pick_param(params, ["window_days", "time_window_days", "window", "window_size_days"], default=None)
ngram = pick_param(params, ["ngram_range", "ngram"], default=None)
min_df = pick_param(params, ["min_df"], default=None)
max_df = pick_param(params, ["max_df"], default=None)
top_k = pick_param(params, ["top_k", "topK"], default=None)

# ✅ Expander 1 (tidak ada expander di dalamnya)
with st.sidebar.expander("ℹ️ Info Run (Parameter)", expanded=True):
    st.write("**Run ID:**", r.get("run_id"))
    st.write("**Run Time:**", r.get("run_time"))
    st.write("**Approach:**", r.get("approach"))
    st.write("**Data Range:**", r.get("data_range"))
    st.write("**Notes:**", r.get("notes"))

    cols = st.columns(2)
    with cols[0]:
        st.write("**Cosine Threshold:**", cos_thr if cos_thr is not None else "-")
        st.write("**Window (days):**", window_days if window_days is not None else "-")
        st.write("**Top K (jika ada):**", top_k if top_k is not None else "-")
    with cols[1]:
        st.write("**n-gram:**", ngram if ngram is not None else "-")
        st.write("**min_df:**", min_df if min_df is not None else "-")
        st.write("**max_df:**", max_df if max_df is not None else "-")

# ✅ Expander 2 terpisah (TIDAK bersarang)
with st.sidebar.expander("🔎 params_json (raw)", expanded=False):
    st.json(params)

# ======================================================
# 📥 Load data
# ======================================================
summary = load_summary(run_id)
if summary.empty:
    st.warning(f"Tidak ada data di {SCHEMA}.{T_SUMMARY} untuk run_id={run_id}.")
    st.stop()

mod_sum = build_modul_summary(summary)
if mod_sum.empty:
    st.warning("Ringkasan modul kosong (cek kolom modul pada cluster_summary).")
    st.stop()

# ======================================================
# 🧾 Header + KPI
# ======================================================
st.title("📊 Summary Hasil Modeling Sintaksis — Semua Modul")
st.caption(
    "Ringkasan hasil klasterisasi per modul untuk 1 run tertentu: parameter run (mis. cosine threshold), "
    "jumlah cluster, statistik ukuran cluster, serta rentang waktu kemunculan. Cocok untuk tabel/figures tesis."
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Run ID", run_id)
k2.metric("Jumlah Modul", f"{mod_sum['modul'].nunique():,}")
k3.metric("Total Cluster", f"{int(mod_sum['n_clusters'].sum()):,}")
k4.metric("Total Tiket (di cluster)", f"{int(mod_sum['total_tickets'].sum()):,}")

# ======================================================
# 🔎 Filter ringkasan
# ======================================================
st.subheader("⚙️ Filter Ringkasan")
c1, c2, c3, c4 = st.columns(4)
with c1:
    min_clusters = st.number_input("Min jumlah cluster per modul", min_value=1, value=1, step=1)
with c2:
    min_total_tickets = st.number_input("Min total tiket per modul", min_value=0, value=0, step=100)
with c3:
    sort_by = st.selectbox(
        "Urutkan berdasarkan",
        ["n_clusters", "total_tickets", "avg_cluster_size", "p90_cluster_size", "max_cluster_size"],
        index=0,
    )
with c4:
    top_n = st.number_input("Tampilkan Top N modul", min_value=5, value=30, step=5)

filtered = mod_sum[
    (mod_sum["n_clusters"] >= int(min_clusters)) &
    (mod_sum["total_tickets"] >= int(min_total_tickets))
].copy()

filtered = filtered.sort_values(sort_by, ascending=False).head(int(top_n))

# ======================================================
# 📤 Download (untuk lampiran / olah lanjut)
# ======================================================
st.download_button(
    "⬇️ Download Summary Per Modul (CSV)",
    data=df_to_csv_bytes(filtered),
    file_name=f"summary_per_modul_{run_id}.csv",
    mime="text/csv",
    use_container_width=True,
)

# ======================================================
# 📋 Tabel utama
# ======================================================
st.subheader("📋 Tabel Ringkasan Per Modul")

view_cols = [
    "modul",
    "n_clusters", "total_tickets",
    "avg_cluster_size", "median_cluster_size", "p90_cluster_size",
    "max_cluster_size", "min_cluster_size",
    "large_clusters_ge_50", "large_clusters_ge_100",
    "first_seen_min", "last_seen_max",
]
st.dataframe(filtered[view_cols], use_container_width=True, height=420)

# ======================================================
# 📈 Visualisasi ringkas
# ======================================================
st.subheader("📈 Visualisasi Ringkas")
left, right = st.columns(2)

with left:
    chart1 = (
        alt.Chart(filtered)
        .mark_bar()
        .encode(
            x=alt.X("n_clusters:Q", title="Jumlah Cluster"),
            y=alt.Y("modul:N", sort="-x", title="Modul"),
            tooltip=["modul", "n_clusters", "total_tickets", "avg_cluster_size", "max_cluster_size"],
        )
        .properties(height=420, title="Top Modul berdasarkan Jumlah Cluster")
    )
    st.altair_chart(chart1, use_container_width=True)

with right:
    chart2 = (
        alt.Chart(filtered)
        .mark_bar()
        .encode(
            x=alt.X("total_tickets:Q", title="Total Tiket (akumulasi n_tickets)"),
            y=alt.Y("modul:N", sort="-x", title="Modul"),
            tooltip=["modul", "total_tickets", "n_clusters", "p90_cluster_size", "max_cluster_size"],
        )
        .properties(height=420, title="Top Modul berdasarkan Total Tiket dalam Cluster")
    )
    st.altair_chart(chart2, use_container_width=True)

# ======================================================
# 🔍 Drill-down per modul (opsional)
# ======================================================
st.divider()
st.subheader("🔎 Drill-down per Modul (opsional)")

modul_opts = filtered["modul"].tolist()
if modul_opts:
    selected_modul = st.selectbox("Pilih modul", modul_opts, index=0)
    dfm = summary[summary["modul"] == selected_modul].copy()
    dfm = dfm.sort_values(["n_tickets", "window_start"], ascending=[False, False])

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Modul", selected_modul)
    a2.metric("Jumlah Cluster", f"{len(dfm):,}")
    a3.metric("Total Tiket", f"{int(dfm['n_tickets'].sum()):,}")
    a4.metric("Rata-rata ukuran cluster", f"{(dfm['n_tickets'].mean() if len(dfm) else 0):.2f}")

    st.markdown("**Distribusi ukuran cluster (n_tickets)**")
    hist = (
        alt.Chart(dfm)
        .mark_bar()
        .encode(
            x=alt.X("n_tickets:Q", bin=alt.Bin(maxbins=30), title="Ukuran Cluster (n_tickets)"),
            y=alt.Y("count():Q", title="Jumlah Cluster"),
            tooltip=[alt.Tooltip("count():Q", title="Jumlah Cluster")],
        )
        .properties(height=280)
    )
    st.altair_chart(hist, use_container_width=True)

    st.markdown("**Daftar cluster pada modul terpilih**")
    st.dataframe(
        dfm[[
            "cluster_id", "window_start", "window_end",
            "n_tickets", "first_seen", "last_seen", "top_terms"
        ]],
        use_container_width=True,
        height=360,
    )

    st.download_button(
        "⬇️ Download Cluster Summary (Modul terpilih) CSV",
        data=df_to_csv_bytes(dfm),
        file_name=f"cluster_summary_{run_id}_{selected_modul}.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
    st.info("Tidak ada modul yang lolos filter untuk drill-down.")
