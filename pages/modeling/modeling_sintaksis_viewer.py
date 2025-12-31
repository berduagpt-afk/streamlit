# pages/modeling_sintaksis_offline_viewer.py
# ============================================================
# Viewer Hasil Modeling Sintaksis (Offline) — Read-only dari DB
# VIEW PER JOB_ID + STATISTIK TIAP RUN
#
# PATCH (FULL, CONSISTENT dgn skema terbaru):
# - UndefinedColumn-safe load_runs (kolom adaptif)
# - Tambah n_clusters_multi (clusters >= 2) bila ada
# - Definisi singleton rate eksplisit:
#     singleton_cluster_rate = n_singletons / n_clusters_all
# - Avg cluster size diperbaiki:
#     avg_cluster_size_all = n_rows / n_clusters_all
#     avg_cluster_size_ge2 = (n_rows - n_singletons) / n_clusters_multi
# - Drill-down diperbaiki (indent rapi) + filter singleton cluster benar-benar diterapkan
# - Exclude cluster_size=1 (singleton clusters) di viewer (default ON)
# - Label run & kolom tabel otomatis menyesuaikan kolom tersedia (legacy aman)
# ============================================================

from __future__ import annotations

import json

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

# ======================================================
# ⚙️ Konstanta
# ======================================================
SCHEMA = "lasis_djp"
T_RUNS = "modeling_sintaksis_runs"
T_CLUSTERS = "modeling_sintaksis_clusters"
T_MEMBERS = "modeling_sintaksis_members"


# =========================
# DB
# =========================
@st.cache_resource(show_spinner=False)
def get_engine():
    cfg = st.secrets["connections"]["postgres"]
    url = (
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )
    return create_engine(url, pool_pre_ping=True)


engine = get_engine()


# ======================================================
# Helpers
# ======================================================
def _safe_json(x) -> dict:
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return {}
    return {}


def _fmt_uuid(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
        return ""
    return str(x)


def _num(x, default=None):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _int(x, default=None):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default


def kpi_row(cols: list, labels_values: list):
    for c, (lab, val) in zip(cols, labels_values):
        c.metric(lab, val)


def chart_line(df: pd.DataFrame, x: str, y: str, title: str, tooltip: list[str]):
    if df.empty or x not in df.columns or y not in df.columns:
        return alt.Chart(pd.DataFrame({x: [], y: []})).mark_line()
    return (
        alt.Chart(df, title=title)
        .mark_line(point=True)
        .encode(
            x=alt.X(f"{x}:Q", title=x),
            y=alt.Y(f"{y}:Q", title=y),
            tooltip=tooltip,
        )
        .properties(height=280)
    )


def chart_scatter(df: pd.DataFrame, x: str, y: str, title: str, tooltip: list[str]):
    if df.empty or x not in df.columns or y not in df.columns:
        return alt.Chart(pd.DataFrame({x: [], y: []})).mark_point()
    return (
        alt.Chart(df, title=title)
        .mark_point(filled=True, size=80)
        .encode(
            x=alt.X(f"{x}:Q", title=x),
            y=alt.Y(f"{y}:Q", title=y),
            tooltip=tooltip,
        )
        .properties(height=280)
    )


# ======================================================
# DB Introspection (kolom adaptif)
# ======================================================
@st.cache_data(show_spinner=False, ttl=300)
def get_table_columns(_engine, schema: str, table: str) -> set[str]:
    q = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = :schema
      AND table_name = :table
    """
    df = pd.read_sql(text(q), _engine, params={"schema": schema, "table": table})
    return set(df["column_name"].astype(str).tolist()) if not df.empty else set()


# ======================================================
# Loaders
# ======================================================
@st.cache_data(show_spinner=False, ttl=60)
def load_runs(_engine) -> pd.DataFrame:
    cols = get_table_columns(_engine, SCHEMA, T_RUNS)

    desired = [
        "job_id",
        "modeling_id",
        "run_time",
        "approach",
        "params_json",
        "notes",
        "tfidf_run_id",
        "threshold",
        "knn_k",
        "n_rows",
        "n_clusters_all",
        "n_singletons",
        "n_clusters_multi",   # ✅ baru jika tersedia
        "vocab_size",
        "nnz",
        "elapsed_sec",

        # legacy/opsional
        "window_days",
        "min_cluster_size",
        "n_clusters_recurring",
        "n_noise_tickets",
    ]

    selected = [c for c in desired if c in cols]
    if not selected:
        return pd.DataFrame()

    q = f"""
    SELECT {", ".join(selected)}
    FROM {SCHEMA}.{T_RUNS}
    ORDER BY run_time DESC
    """
    df = pd.read_sql(text(q), _engine)
    if df.empty:
        return df

    for c in desired:
        if c not in df.columns:
            df[c] = pd.NA

    df["run_time"] = pd.to_datetime(df["run_time"], errors="coerce")
    df["params"] = df["params_json"].apply(_safe_json)

    def _fill(row, key):
        v = row.get(key)
        if v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v):
            return row["params"].get(key)
        return v

    # minimal
    df["threshold"] = df.apply(lambda r: _fill(r, "threshold"), axis=1)
    df["knn_k"] = df.apply(lambda r: _fill(r, "knn_k"), axis=1)

    # legacy (aman)
    if "window_days" in df.columns:
        df["window_days"] = df.apply(lambda r: _fill(r, "window_days"), axis=1)
    if "min_cluster_size" in df.columns:
        df["min_cluster_size"] = df.apply(lambda r: _fill(r, "min_cluster_size"), axis=1)

    df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce")
    df["knn_k"] = pd.to_numeric(df["knn_k"], errors="coerce").astype("Int64")

    if "window_days" in df.columns:
        df["window_days"] = pd.to_numeric(df["window_days"], errors="coerce").astype("Int64")
    if "min_cluster_size" in df.columns:
        df["min_cluster_size"] = pd.to_numeric(df["min_cluster_size"], errors="coerce").astype("Int64")

    for c in [
        "n_rows",
        "n_clusters_all",
        "n_singletons",
        "n_clusters_multi",
        "vocab_size",
        "nnz",
        "n_noise_tickets",
        "n_clusters_recurring",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    df["elapsed_sec"] = pd.to_numeric(df["elapsed_sec"], errors="coerce")

    df["job_id"] = df["job_id"].astype("string")
    df["modeling_id"] = df["modeling_id"].astype("string")
    df["tfidf_run_id"] = df["tfidf_run_id"].astype("string")

    df["threshold_r"] = df["threshold"].round(4)
    return df


@st.cache_data(show_spinner=False, ttl=60)
def load_clusters(_engine, modeling_id: str) -> pd.DataFrame:
    q = f"""
    SELECT job_id, modeling_id, cluster_id, cluster_size, min_time, max_time, span_days
    FROM {SCHEMA}.{T_CLUSTERS}
    WHERE modeling_id = :mid
    """
    df = pd.read_sql(text(q), _engine, params={"mid": modeling_id})
    if df.empty:
        return df
    df["min_time"] = pd.to_datetime(df["min_time"], errors="coerce")
    df["max_time"] = pd.to_datetime(df["max_time"], errors="coerce")
    df["span_days"] = pd.to_numeric(df["span_days"], errors="coerce")
    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce").astype("Int64")
    df["cluster_size"] = pd.to_numeric(df["cluster_size"], errors="coerce").astype("Int64")
    return df


@st.cache_data(show_spinner=False, ttl=60)
def load_members_sample(_engine, modeling_id: str, limit_rows: int = 200) -> pd.DataFrame:
    q = f"""
    SELECT job_id, modeling_id, cluster_id, incident_number, tgl_submit, site, assignee, modul, sub_modul
    FROM {SCHEMA}.{T_MEMBERS}
    WHERE modeling_id = :mid
    ORDER BY tgl_submit ASC NULLS LAST
    LIMIT :lim
    """
    df = pd.read_sql(text(q), _engine, params={"mid": modeling_id, "lim": int(limit_rows)})
    if df.empty:
        return df
    df["tgl_submit"] = pd.to_datetime(df["tgl_submit"], errors="coerce")
    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce").astype("Int64")
    return df


@st.cache_data(show_spinner=False, ttl=60)
def load_members_for_cluster(_engine, modeling_id: str, cluster_id: int) -> pd.DataFrame:
    q = f"""
    SELECT job_id, modeling_id, cluster_id, incident_number, tgl_submit, site, assignee, modul, sub_modul
    FROM {SCHEMA}.{T_MEMBERS}
    WHERE modeling_id = :mid AND cluster_id = :cid
    ORDER BY tgl_submit ASC NULLS LAST
    """
    df = pd.read_sql(text(q), _engine, params={"mid": modeling_id, "cid": int(cluster_id)})
    if df.empty:
        return df
    df["tgl_submit"] = pd.to_datetime(df["tgl_submit"], errors="coerce")
    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce").astype("Int64")
    return df


# ======================================================
# UI
# ======================================================
st.title("📌 Viewer Modeling Sintaksis — Per Job (Statistik Tiap Run)")
st.caption(
    "Fokus halaman ini: memilih **job_id** lalu melihat statistik dan perbandingan **semua run** di dalam job tersebut. "
    "Drill-down ke satu run tetap tersedia (opsional)."
)

runs = load_runs(engine)
if runs.empty:
    st.warning(f"Belum ada data atau kolom minimal tidak ditemukan di tabel {SCHEMA}.{T_RUNS}.")
    st.stop()

st.sidebar.header("⚙️ Filter")

exclude_singleton_clusters = st.sidebar.checkbox(
    "Exclude cluster_size=1 (singleton)",
    value=True,
    help="Jika aktif, distribusi klaster & top cluster tidak memasukkan cluster_size=1. Data DB tidak berubah."
)

runs["job_key"] = runs["job_id"].fillna("NO_JOB").astype(str)

job_summary = (
    runs.groupby("job_key", as_index=False)
    .agg(
        n_runs=("modeling_id", "count"),
        min_run_time=("run_time", "min"),
        max_run_time=("run_time", "max"),
        tfidf_run_id=("tfidf_run_id", "first"),
    )
    .sort_values("max_run_time", ascending=False)
)

job_labels = job_summary.apply(
    lambda r: f"{r['job_key']} | runs={int(r['n_runs'])} | "
              f"{(r['min_run_time'].strftime('%Y-%m-%d %H:%M') if pd.notna(r['min_run_time']) else 'NA')} .. "
              f"{(r['max_run_time'].strftime('%Y-%m-%d %H:%M') if pd.notna(r['max_run_time']) else 'NA')}",
    axis=1
).tolist()
job_map = dict(zip(job_labels, job_summary["job_key"].astype(str).tolist()))

sel_job_label = st.sidebar.selectbox("Pilih job_id", options=job_labels, index=0)
job_id = job_map[sel_job_label]

runs_job = runs[runs["job_key"] == job_id].copy().sort_values("run_time", ascending=False)

thr_vals = sorted([float(x) for x in runs_job["threshold_r"].dropna().unique().tolist()])
thr_pick = st.sidebar.multiselect("Filter threshold (opsional)", options=thr_vals, default=thr_vals)
if thr_pick:
    runs_job = runs_job[runs_job["threshold_r"].isin(thr_pick)].copy()

show_raw_params = st.sidebar.checkbox("Tampilkan params_json mentah (di detail run)", value=False)

st.sidebar.divider()
enable_drilldown = st.sidebar.checkbox("Aktifkan drill-down (lihat clusters/members)", value=True)
max_members_preview = st.sidebar.slider("Preview members (drill-down)", 20, 400, 100, 20)
top_n_clusters = st.sidebar.slider("Top cluster (drill-down)", 5, 50, 15, 5)

if runs_job.empty:
    st.warning("Tidak ada run yang cocok dengan filter.")
    st.stop()

# ======================================================
# Ringkasan Job
# ======================================================
st.subheader("Ringkasan Job")
tfidf_run_id = runs_job["tfidf_run_id"].dropna().astype(str).head(1).tolist()
tfidf_run_id = tfidf_run_id[0] if tfidf_run_id else ""

min_rt = runs_job["run_time"].min()
max_rt = runs_job["run_time"].max()

c1, c2, c3, c4, c5 = st.columns(5)
kpi_row(
    [c1, c2, c3, c4, c5],
    [
        ("Job ID", job_id),
        ("Jumlah Run", f"{len(runs_job):,}"),
        ("TF-IDF Run ID", tfidf_run_id or "NA"),
        ("Rentang Waktu", f"{min_rt.strftime('%Y-%m-%d %H:%M') if pd.notna(min_rt) else 'NA'} → "
                          f"{max_rt.strftime('%Y-%m-%d %H:%M') if pd.notna(max_rt) else 'NA'}"),
        ("Approach", str(runs_job["approach"].dropna().unique()[:1].tolist()[0]) if runs_job["approach"].notna().any() else "NA"),
    ],
)

st.divider()

# ======================================================
# Statistik per Run
# ======================================================
st.subheader("Statistik Tiap Run (dalam Job)")

n_rows_num = pd.to_numeric(runs_job["n_rows"], errors="coerce")
n_clusters_all_num = pd.to_numeric(runs_job["n_clusters_all"], errors="coerce")
n_singletons_num = pd.to_numeric(runs_job["n_singletons"], errors="coerce")

# ✅ singleton cluster rate (konsisten)
runs_job["singleton_cluster_rate"] = (
    n_singletons_num / n_clusters_all_num.replace({0: pd.NA})
).astype("Float64")

# ✅ avg cluster size (SEMUA cluster, termasuk singleton)
runs_job["avg_cluster_size_all"] = (
    n_rows_num / n_clusters_all_num.replace({0: pd.NA})
).astype("Float64")

# ✅ cluster multi-member (>=2)
if "n_clusters_multi" in runs_job.columns and runs_job["n_clusters_multi"].notna().any():
    n_clusters_multi_num = pd.to_numeric(runs_job["n_clusters_multi"], errors="coerce")
else:
    n_clusters_multi_num = (n_clusters_all_num - n_singletons_num)

runs_job["n_clusters_multi_calc"] = pd.to_numeric(n_clusters_multi_num, errors="coerce")

# tiket di cluster >=2: total - singleton tickets (singleton tickets = n_singletons)
tickets_multi = (n_rows_num - n_singletons_num).astype("Float64")

# ✅ avg cluster size untuk cluster >=2
runs_job["avg_cluster_size_ge2"] = (
    tickets_multi / runs_job["n_clusters_multi_calc"].replace({0: pd.NA})
).astype("Float64")

# guard anomali
runs_job.loc[tickets_multi < 0, "avg_cluster_size_ge2"] = pd.NA

view_cols = [
    "run_time",
    "threshold_r",
    "knn_k",
    "n_rows",
    "n_clusters_all",
    "n_singletons",
    "n_clusters_multi",
    "n_clusters_multi_calc",
    "avg_cluster_size_all",
    "avg_cluster_size_ge2",
    "singleton_cluster_rate",
    "vocab_size",
    "nnz",
    "elapsed_sec",
    # legacy
    "window_days",
    "min_cluster_size",
    "modeling_id",
]
runs_table = runs_job[[c for c in view_cols if c in runs_job.columns]].copy()

# format tampil
if "singleton_cluster_rate" in runs_table.columns:
    runs_table["singleton_cluster_rate"] = (pd.to_numeric(runs_table["singleton_cluster_rate"], errors="coerce") * 100.0).round(2)

for c in ["avg_cluster_size_all", "avg_cluster_size_ge2"]:
    if c in runs_table.columns:
        runs_table[c] = pd.to_numeric(runs_table[c], errors="coerce").round(2)

if "threshold_r" in runs_table.columns:
    runs_table = runs_table.rename(columns={"threshold_r": "threshold"})

# drop legacy cols kalau semua NA
for legacy in ["window_days", "min_cluster_size"]:
    if legacy in runs_table.columns and runs_table[legacy].isna().all():
        runs_table = runs_table.drop(columns=[legacy])

st.dataframe(runs_table, use_container_width=True)

# ======================================================
# Grafik per threshold
# ======================================================
st.markdown("### Grafik Perbandingan Antar Threshold (dalam Job)")

plot_df = runs_job.dropna(subset=["threshold_r"]).copy()
plot_df["threshold"] = pd.to_numeric(plot_df["threshold_r"], errors="coerce")

plot_df["n_clusters_all_f"] = pd.to_numeric(plot_df["n_clusters_all"], errors="coerce")
plot_df["n_singletons_f"] = pd.to_numeric(plot_df["n_singletons"], errors="coerce")
plot_df["n_clusters_multi_f"] = pd.to_numeric(plot_df.get("n_clusters_multi", pd.NA), errors="coerce")
if plot_df["n_clusters_multi_f"].isna().all():
    plot_df["n_clusters_multi_f"] = (plot_df["n_clusters_all_f"] - plot_df["n_singletons_f"])

plot_df["elapsed_sec_f"] = pd.to_numeric(plot_df["elapsed_sec"], errors="coerce")
plot_df["avg_cluster_size_all_f"] = pd.to_numeric(plot_df["avg_cluster_size_all"], errors="coerce")
plot_df["avg_cluster_size_ge2_f"] = pd.to_numeric(plot_df["avg_cluster_size_ge2"], errors="coerce")
plot_df["singleton_cluster_rate_f"] = (pd.to_numeric(plot_df["singleton_cluster_rate"], errors="coerce") * 100.0)

tooltip = [
    alt.Tooltip("threshold:Q"),
    alt.Tooltip("n_clusters_all_f:Q", title="n_clusters_all"),
    alt.Tooltip("n_clusters_multi_f:Q", title="n_clusters_multi(>=2)"),
    alt.Tooltip("n_singletons_f:Q", title="n_singletons"),
    alt.Tooltip("avg_cluster_size_all_f:Q", title="avg_cluster_size_all"),
    alt.Tooltip("avg_cluster_size_ge2_f:Q", title="avg_cluster_size_ge2"),
    alt.Tooltip("singleton_cluster_rate_f:Q", title="singleton_cluster_rate(%)"),
    alt.Tooltip("elapsed_sec_f:Q", title="elapsed_sec"),
    alt.Tooltip("modeling_id:N"),
]

g1, g2 = st.columns(2)
with g1:
    st.altair_chart(chart_line(plot_df, "threshold", "n_clusters_all_f", "Threshold vs Total Clusters", tooltip), use_container_width=True)
with g2:
    st.altair_chart(chart_line(plot_df, "threshold", "n_clusters_multi_f", "Threshold vs Multi-member Clusters (>=2)", tooltip), use_container_width=True)

g3, g4 = st.columns(2)
with g3:
    st.altair_chart(chart_line(plot_df, "threshold", "n_singletons_f", "Threshold vs Singleton Clusters", tooltip), use_container_width=True)
with g4:
    st.altair_chart(chart_line(plot_df, "threshold", "avg_cluster_size_ge2_f", "Threshold vs Avg Cluster Size (clusters >=2)", tooltip), use_container_width=True)

st.altair_chart(chart_line(plot_df, "threshold", "elapsed_sec_f", "Threshold vs Runtime (elapsed_sec)", tooltip), use_container_width=True)
st.altair_chart(chart_scatter(plot_df, "n_clusters_all_f", "elapsed_sec_f", "Runtime vs Total Clusters", tooltip), use_container_width=True)

st.divider()

# ======================================================
# Detail run (opsional / drill-down)
# ======================================================
st.subheader("Detail Run (Opsional / Drill-down)")
st.caption("Pilih satu run untuk melihat data clusters/members. Ini tidak wajib untuk analisis job-level.")

def _mk_run_label(r: pd.Series) -> str:
    rt = r["run_time"].strftime("%Y-%m-%d %H:%M") if pd.notna(r["run_time"]) else "NA"
    thr = r.get("threshold_r")
    k = r.get("knn_k")
    ncl = r.get("n_clusters_all")
    ns = r.get("n_singletons")
    mid = r.get("modeling_id")
    return f"{rt} | thr={thr} k={k} | clusters={ncl} singletons={ns} | {mid}"

run_labels = runs_job.apply(_mk_run_label, axis=1).tolist()
run_map = dict(zip(run_labels, runs_job["modeling_id"].astype(str).tolist()))

sel_run = st.selectbox("Pilih run untuk detail", options=run_labels, index=0)
mid = run_map[sel_run]
row = runs_job.loc[runs_job["modeling_id"].astype(str) == mid].iloc[0]

c1, c2, c3, c4, c5 = st.columns(5)
kpi_row(
    [c1, c2, c3, c4, c5],
    [
        ("Threshold", str(row.get("threshold_r"))),
        ("kNN-k", str(row.get("knn_k"))),
        ("n_clusters_all", f"{_int(row.get('n_clusters_all'), 0):,}"),
        ("n_singletons", f"{_int(row.get('n_singletons'), 0):,}"),
        ("elapsed_sec", f"{_num(row.get('elapsed_sec'), 0):,.2f}"),
    ]
)

st.write(f"**Modeling ID:** `{mid}`")
st.write(f"**Notes:** {row.get('notes','')}")
st.write(f"**TF-IDF Run ID:** `{_fmt_uuid(row.get('tfidf_run_id'))}`")
if show_raw_params:
    st.json(_safe_json(row.get("params_json")))

# ======================================================
# Drill-down
# ======================================================
if enable_drilldown:
    clusters = load_clusters(engine, mid)

    clusters_view = clusters.copy()
    if exclude_singleton_clusters and not clusters_view.empty and "cluster_size" in clusters_view.columns:
        clusters_view = clusters_view[
            pd.to_numeric(clusters_view["cluster_size"], errors="coerce") >= 2
        ].copy()

    if exclude_singleton_clusters and not clusters_view.empty:
        n_ge2 = int(len(clusters_view))
        tickets_ge2 = int(pd.to_numeric(clusters_view["cluster_size"], errors="coerce").fillna(0).sum())
        n_rows_run = _int(row.get("n_rows"), 0) or 0
        coverage = (tickets_ge2 / n_rows_run * 100.0) if n_rows_run > 0 else 0.0

        st.caption(
            f"🔎 **Viewer filter aktif (cluster_size ≥ 2)** — "
            f"Jumlah klaster: **{n_ge2:,}**, "
            f"Tiket dalam klaster: **{tickets_ge2:,}** "
            f"(coverage **{coverage:.2f}%** dari total tiket run)"
        )

    members_sample = load_members_sample(engine, mid, limit_rows=int(max_members_preview))
    if exclude_singleton_clusters and not clusters_view.empty and not members_sample.empty:
        valid_ids = set(clusters_view["cluster_id"].dropna().astype(int).tolist())
        members_sample = members_sample[members_sample["cluster_id"].isin(list(valid_ids))].copy()

    tab1, tab2, tab3 = st.tabs(["Clusters", "Members (sample)", "Top Clusters (detail)"])

    with tab1:
        if clusters_view.empty:
            st.info("Tidak ada data clusters (setelah filter) untuk run ini.")
        else:
            st.altair_chart(
                alt.Chart(clusters_view, title="Distribusi Ukuran Klaster (detail run)")
                .mark_bar()
                .encode(
                    x=alt.X("cluster_size:Q", bin=alt.Bin(maxbins=40), title="cluster_size"),
                    y=alt.Y("count():Q", title="jumlah klaster"),
                    tooltip=[alt.Tooltip("cluster_size:Q"), alt.Tooltip("count():Q", title="n_clusters")],
                )
                .properties(height=260),
                use_container_width=True,
            )
            st.dataframe(
                clusters_view.sort_values(["cluster_size", "span_days"], ascending=[False, True]).head(50),
                use_container_width=True,
            )

    with tab2:
        if members_sample.empty:
            st.info("Tidak ada data members (setelah filter) untuk run ini.")
        else:
            st.dataframe(members_sample, use_container_width=True)
            st.caption("Catatan: tabel ini hanya sample (LIMIT) untuk menjaga performa.")

    with tab3:
        if clusters_view.empty:
            st.info("Data cluster kosong (setelah filter).")
        else:
            topc = clusters_view.sort_values(["cluster_size", "span_days"], ascending=[False, True]).head(top_n_clusters)
            st.dataframe(topc, use_container_width=True)

            options = topc["cluster_id"].dropna().astype(int).tolist()
            if not options:
                st.info("Tidak ada cluster_id yang valid.")
            else:
                pick = st.selectbox("Pilih cluster_id", options=options, index=0, key="pick_cluster")
                mem = load_members_for_cluster(engine, mid, int(pick))

                st.markdown(f"##### Members untuk cluster {pick} (n={len(mem)})")
                if mem.empty:
                    st.info("Tidak ada member untuk cluster ini.")
                else:
                    st.dataframe(mem.head(max_members_preview), use_container_width=True)

                    if mem["tgl_submit"].notna().any():
                        tmp = mem.dropna(subset=["tgl_submit"]).copy()
                        tmp["date"] = tmp["tgl_submit"].dt.date
                        ts = tmp.groupby("date", as_index=False).size().rename(columns={"size": "n_tickets"})
                        st.altair_chart(
                            alt.Chart(ts, title="Timeline dalam cluster (per hari)")
                            .mark_line(point=True)
                            .encode(
                                x=alt.X("date:T", title="Tanggal"),
                                y=alt.Y("n_tickets:Q", title="Jumlah tiket"),
                                tooltip=["date", "n_tickets"],
                            )
                            .properties(height=240),
                            use_container_width=True,
                        )

with st.expander("ℹ️ Debug"):
    st.write("job_id:", job_id)
    st.write("runs in job:", len(runs_job))
    st.write("selected modeling_id:", mid)
    st.write("exclude_singleton_clusters:", exclude_singleton_clusters)
    st.write("enable_drilldown:", enable_drilldown)
