# pages/modeling_sintaksis_cosine.py
# Viewer Hasil Modeling (MODUL ONLY) — Read-only dari DB
# Membaca: lasis_djp.modeling_runs, lasis_djp.cluster_summary, lasis_djp.cluster_members

import streamlit as st
import pandas as pd
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
T_MEMBERS = "cluster_members"

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
# 🧠 Helpers (DB)
# ======================================================
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
    ORDER BY n_tickets DESC, window_start DESC
    """
    df = pd.read_sql(text(sql), ENGINE, params={"run_id": run_id})
    if not df.empty:
        for c in ["window_start", "window_end", "first_seen", "last_seen"]:
            df[c] = pd.to_datetime(df[c], errors="coerce")
        df["n_tickets"] = pd.to_numeric(df["n_tickets"], errors="coerce").fillna(0).astype(int)
        df["modul"] = df["modul"].astype(str)
        df["cluster_id"] = df["cluster_id"].astype(str)
        df["top_terms"] = df["top_terms"].fillna("").astype(str)
    return df

@st.cache_data(show_spinner=False)
def load_members(run_id: str, cluster_id: str) -> pd.DataFrame:
    sql = f"""
    SELECT
        run_id, cluster_id, incident_number, tgl_submit, site, assignee,
        modul, sub_modul, detailed_decription, text_sintaksis, tgl_preprocessed
    FROM {SCHEMA}.{T_MEMBERS}
    WHERE run_id = :run_id
      AND cluster_id = :cluster_id
    ORDER BY tgl_submit ASC
    """
    df = pd.read_sql(text(sql), ENGINE, params={"run_id": run_id, "cluster_id": cluster_id})
    if not df.empty:
        df["tgl_submit"] = pd.to_datetime(df["tgl_submit"], errors="coerce")
        df["tgl_preprocessed"] = pd.to_datetime(df["tgl_preprocessed"], errors="coerce", utc=True)
    return df

@st.cache_data(show_spinner=False)
def load_cluster_monthly_points(run_id: str, cluster_ids: list[str]) -> pd.DataFrame:
    """
    Menghasilkan titik timeline per bulan untuk cluster terpilih:
    (cluster_id, month, n_incidents)
    """
    if not cluster_ids:
        return pd.DataFrame(columns=["cluster_id", "month", "n_incidents"])

    # Pakai ANY(:cluster_ids) agar efisien
    sql = f"""
    SELECT
      cluster_id,
      date_trunc('month', tgl_submit)::date AS month,
      COUNT(*)::int AS n_incidents
    FROM {SCHEMA}.{T_MEMBERS}
    WHERE run_id = :run_id
      AND cluster_id = ANY(:cluster_ids)
    GROUP BY cluster_id, date_trunc('month', tgl_submit)::date
    ORDER BY month ASC;
    """
    df = pd.read_sql(
        text(sql),
        ENGINE,
        params={"run_id": run_id, "cluster_ids": cluster_ids},
    )
    if not df.empty:
        df["month"] = pd.to_datetime(df["month"], errors="coerce")
        df["cluster_id"] = df["cluster_id"].astype(str)
        df["n_incidents"] = pd.to_numeric(df["n_incidents"], errors="coerce").fillna(0).astype(int)
    return df

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")

def short_topic(top_terms: str, max_len: int = 42) -> str:
    s = (top_terms or "").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3].rstrip() + "..."

# ======================================================
# 🎛️ Sidebar
# ======================================================
st.sidebar.header("⚙️ Viewer Hasil Modeling")

runs = load_runs()
if runs.empty:
    st.warning(f"Tabel {SCHEMA}.{T_RUNS} kosong. Jalankan run_modeling.py terlebih dahulu.")
    st.stop()

run_label = runs.apply(lambda r: f"{r['run_time']} | {r['approach']} | {r['run_id']}", axis=1)
run_map = dict(zip(run_label, runs["run_id"]))
selected_run_label = st.sidebar.selectbox("Pilih run_id", run_label.tolist(), index=0)
run_id = run_map[selected_run_label]

with st.sidebar.expander("📌 Info Run", expanded=False):
    r = runs[runs["run_id"] == run_id].iloc[0].to_dict()
    st.write("**run_id:**", r.get("run_id"))
    st.write("**run_time:**", r.get("run_time"))
    st.write("**approach:**", r.get("approach"))

summary = load_summary(run_id)
if summary.empty:
    st.warning(f"Tidak ada cluster_summary untuk run_id={run_id}.")
    st.stop()

# ======================================================
# ✅ FUNGSI-STYLE VIEW (seperti contoh gambar)
# ======================================================
st.markdown("## 📌 Cluster Overview (Functional View)")

# Application Name = modul
modul_opts = sorted([m for m in summary["modul"].dropna().unique().tolist() if m and m != "nan"])
if not modul_opts:
    st.warning("Kolom modul kosong pada summary.")
    st.stop()

colA, colB = st.columns([1, 2])
with colA:
    app_name = st.selectbox("Application Name", modul_opts, index=0)
with colB:
    st.caption("Tampilan ini merangkum cluster (label/size/topic) dan kemunculannya dari waktu ke waktu (per bulan).")

# Filter khusus view ini
f_app = summary[summary["modul"] == app_name].copy()

max_n = int(max(2, f_app["n_tickets"].max()))
c1, c2, c3 = st.columns(3)
with c1:
    min_cluster = st.number_input("Min Cluster Size", min_value=2, value=2, step=1)
with c2:
    top_k = st.number_input("Top K Clusters", min_value=5, max_value=200, value=30, step=5)
with c3:
    st.metric("Clusters (modul)", f"{len(f_app):,}")

f_app = f_app[f_app["n_tickets"] >= int(min_cluster)].copy()
f_app = f_app.sort_values(["n_tickets", "window_start"], ascending=[False, False]).head(int(top_k))

if f_app.empty:
    st.info("Tidak ada cluster yang memenuhi filter pada modul ini.")
    st.stop()

# Buat label numerik seperti contoh (0..K-1)
f_app = f_app.reset_index(drop=True)
f_app["cluster_label"] = f_app.index.astype(int)

# Table kiri: Cluster Label | Cluster Size | Cluster Topic
left, right = st.columns([1.05, 1.5])

with left:
    table_like = f_app[["cluster_label", "n_tickets", "top_terms"]].copy()
    table_like.columns = ["Cluster Label", "Cluster Size", "Cluster Topic"]
    table_like["Cluster Topic"] = table_like["Cluster Topic"].apply(lambda x: short_topic(x, 52))
    st.dataframe(table_like, use_container_width=True, height=360)

# Timeline kanan: titik kemunculan cluster per bulan
with right:
    # ambil point timeline dari cluster_members untuk cluster_id terpilih (Top K)
    cluster_ids = f_app["cluster_id"].tolist()
    points = load_cluster_monthly_points(run_id, cluster_ids)

    if points.empty:
        st.info("Tidak ada data timeline pada cluster_members (cek tgl_submit atau isi tabel).")
    else:
        # map cluster_id -> cluster_label supaya y-axis rapi
        mapper = f_app.set_index("cluster_id")["cluster_label"].to_dict()
        points["cluster_label"] = points["cluster_id"].map(mapper)

        # chart: x=month, y=cluster_label, size=n_incidents
        chart = (
            alt.Chart(points)
            .mark_circle()
            .encode(
                x=alt.X("month:T", title="Month"),
                y=alt.Y("cluster_label:O", title="Clusters", sort="ascending"),
                size=alt.Size("n_incidents:Q", title="Count"),
                tooltip=[
                    alt.Tooltip("cluster_label:O", title="Cluster Label"),
                    alt.Tooltip("month:T", title="Month"),
                    alt.Tooltip("n_incidents:Q", title="Count"),
                ],
            )
            .properties(height=360)
        )
        st.altair_chart(chart, use_container_width=True)

st.divider()

# ======================================================
# Viewer lengkap (seperti sebelumnya) — untuk drill-down detail
# ======================================================
st.markdown("## 📊 Viewer Hasil Modeling (Detail View)")

# Filter detail view (opsional: gunakan app_name sebagai default)
st.sidebar.divider()
st.sidebar.subheader("Filter Detail View")

detail_modul_opts = ["(Semua)"] + modul_opts
f_modul = st.sidebar.selectbox("Filter modul (detail)", detail_modul_opts, index=detail_modul_opts.index(app_name) if app_name in detail_modul_opts else 0)

max_n_all = int(max(2, summary["n_tickets"].max()))
min_cluster_detail = st.sidebar.slider("Minimal ukuran cluster (detail)", 2, max_n_all, int(min_cluster))

min_w = summary["window_start"].min()
max_w = summary["window_start"].max()
if pd.notna(min_w) and pd.notna(max_w):
    w_start, w_end = st.sidebar.date_input(
        "Rentang window_start (detail)",
        value=(min_w.date(), max_w.date()),
        min_value=min_w.date(),
        max_value=max_w.date(),
    )
else:
    w_start, w_end = None, None

# Apply detail filters
f = summary.copy()
if f_modul != "(Semua)":
    f = f[f["modul"] == f_modul]
f = f[f["n_tickets"] >= int(min_cluster_detail)]
if w_start and w_end:
    f = f[(f["window_start"].dt.date >= w_start) & (f["window_start"].dt.date <= w_end)]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Run ID", run_id)
k2.metric("Jumlah Cluster (filtered)", f"{len(f):,}")
k3.metric("Total Tiket (filtered)", f"{int(f['n_tickets'].sum()):,}")
k4.metric("Rata-rata ukuran cluster", f"{(f['n_tickets'].mean() if len(f) else 0):.2f}")

st.download_button(
    "⬇️ Download Cluster Summary (Filtered) CSV",
    data=df_to_csv_bytes(f),
    file_name=f"cluster_summary_filtered_{run_id}.csv",
    mime="text/csv",
    use_container_width=True,
)

st.markdown("### 📈 Distribusi")
c1, c2 = st.columns(2)

with c1:
    by_modul = (
        f.groupby("modul", dropna=False)
        .agg(n_cluster=("cluster_id", "count"), n_tickets=("n_tickets", "sum"))
        .reset_index()
        .sort_values("n_cluster", ascending=False)
        .head(30)
    )
    chart_modul = (
        alt.Chart(by_modul)
        .mark_bar()
        .encode(
            x=alt.X("n_cluster:Q", title="Jumlah Cluster"),
            y=alt.Y("modul:N", sort="-x", title="Modul"),
            tooltip=["modul", "n_cluster", "n_tickets"],
        )
        .properties(height=380)
    )
    st.altair_chart(chart_modul, use_container_width=True)

with c2:
    hist = (
        alt.Chart(f)
        .mark_bar()
        .encode(
            x=alt.X("n_tickets:Q", bin=alt.Bin(maxbins=30), title="Ukuran Cluster (n_tickets)"),
            y=alt.Y("count():Q", title="Jumlah Cluster"),
            tooltip=[alt.Tooltip("count():Q", title="Jumlah Cluster")],
        )
        .properties(height=380)
    )
    st.altair_chart(hist, use_container_width=True)

st.markdown("### 📋 Daftar Cluster (Filtered)")
view_cols = [
    "cluster_id", "modul", "window_start", "window_end",
    "n_tickets", "first_seen", "last_seen", "top_terms"
]
table_df = f[view_cols].copy()
st.dataframe(table_df, use_container_width=True, height=360)

cluster_ids = f["cluster_id"].tolist()
if not cluster_ids:
    st.info("Tidak ada cluster yang memenuhi filter.")
    st.stop()

st.markdown("### 🔎 Detail Cluster (Drill-down)")
selected_cluster = st.selectbox("Pilih cluster_id", cluster_ids)

members = load_members(run_id, selected_cluster)
if members.empty:
    st.warning("Tidak ada anggota cluster untuk cluster_id ini.")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Cluster ID", selected_cluster)
m2.metric("Jumlah Tiket", f"{len(members):,}")
m3.metric("Tanggal pertama", str(members["tgl_submit"].min()))
m4.metric("Tanggal terakhir", str(members["tgl_submit"].max()))

rep = summary[summary["cluster_id"] == selected_cluster]
if not rep.empty:
    rep_row = rep.iloc[0]
    with st.expander("🧷 Representative Text & Top Terms", expanded=True):
        st.write("**Representative Incident:**", rep_row.get("representative_incident"))
        st.write("**Top Terms:**", rep_row.get("top_terms"))
        st.text_area("Representative Text (text_sintaksis)", rep_row.get("representative_text") or "", height=160)

st.dataframe(
    members[[
        "tgl_submit", "incident_number", "site", "assignee",
        "modul", "sub_modul", "text_sintaksis"
    ]],
    use_container_width=True,
    height=420,
)

d1, d2 = st.columns(2)
with d1:
    st.download_button(
        "⬇️ Download Anggota Cluster (CSV)",
        data=df_to_csv_bytes(members),
        file_name=f"cluster_members_{run_id}_{selected_cluster}.csv",
        mime="text/csv",
        use_container_width=True,
    )
with d2:
    inc_only = members[["incident_number", "tgl_submit", "modul", "sub_modul"]].copy()
    st.download_button(
        "⬇️ Download Incident List (CSV)",
        data=df_to_csv_bytes(inc_only),
        file_name=f"incident_list_{run_id}_{selected_cluster}.csv",
        mime="text/csv",
        use_container_width=True,
    )
