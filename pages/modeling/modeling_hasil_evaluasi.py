from __future__ import annotations

import math
from typing import Optional, List

import pandas as pd
import numpy as np
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
# ⚙️ KONSTANTA DB
# ======================================================
SCHEMA = "lasis_djp"

T_EVAL = "modeling_evaluation_results"          # gabungan sintaksis/semantik (silhouette, dbi, dbcv)
T_SEM_RUNS = "modeling_semantik_hdbscan_runs"   # dbcv disimpan di sini
T_SYN_RUNS = "modeling_sintaksis_runs"          # opsional (kalau mau tampilkan run sintaksis juga)


# ======================================================
# 🔌 DB Connection
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


engine = get_engine()


# ======================================================
# 🧰 Helpers
# ======================================================
def fmt_float(x: Optional[float], nd: int = 4) -> str:
    if x is None:
        return "-"
    try:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return "-"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "-"


def as_list_unique(x: pd.Series) -> List[str]:
    return sorted([v for v in x.dropna().astype(str).unique().tolist() if v != ""])


# ======================================================
# 📥 Loaders
# ======================================================
@st.cache_data(show_spinner=False, ttl=120)
def load_eval_table(_engine, limit: int = 2000) -> pd.DataFrame:
    q = text(
        f"""
        SELECT
            run_time,
            jenis_pendekatan,
            job_id::text           AS job_id,
            modeling_id::text      AS modeling_id,
            embedding_run_id::text AS embedding_run_id,
            temporal_id,
            silhouette_score,
            dbi,
            dbcv,
            threshold,
            notes,
            meta_json
        FROM {SCHEMA}.{T_EVAL}
        ORDER BY run_time DESC
        LIMIT :lim
        """
    )
    with _engine.begin() as conn:
        df = pd.read_sql(q, conn, params={"lim": int(limit)})
    return df


@st.cache_data(show_spinner=False, ttl=120)
def load_semantik_runs(_engine, limit: int = 1000) -> pd.DataFrame:
    q = text(
        f"""
        SELECT
            modeling_id::text      AS modeling_id,
            run_time,
            embedding_run_id::text AS embedding_run_id,
            n_rows,
            n_clusters,
            n_noise,
            silhouette,
            dbi,
            dbcv,
            params_json,
            notes
        FROM {SCHEMA}.{T_SEM_RUNS}
        ORDER BY run_time DESC
        LIMIT :lim
        """
    )
    with _engine.begin() as conn:
        df = pd.read_sql(q, conn, params={"lim": int(limit)})
    return df


# ======================================================
# 🧭 UI
# ======================================================
st.title("📊 Viewer Hasil Evaluasi Modeling")
st.caption(
    "Menampilkan metrik evaluasi (Silhouette, DBI, DBCV) dari tabel evaluasi gabungan "
    "dan dari runs semantik (HDBSCAN)."
)

with st.sidebar:
    st.header("⚙️ Filter Umum")
    tab_default = st.radio("Tab", ["Evaluasi Gabungan", "DBCV Semantik (Runs)", "Compare"], index=0)

    max_rows = st.number_input("Maks baris (load)", 200, 10000, 2000, 200)
    show_meta = st.checkbox("Tampilkan meta_json", value=False)
    show_notes = st.checkbox("Tampilkan notes", value=True)

tabs = st.tabs(["🧾 Evaluasi Gabungan", "🧠 DBCV Semantik (Runs)", "🔎 Compare"])

# ======================================================
# TAB 1 — EVALUASI GABUNGAN
# ======================================================
with tabs[0]:
    df = load_eval_table(engine, limit=int(max_rows))

    if df.empty:
        st.warning(f"Tabel {SCHEMA}.{T_EVAL} kosong / belum ada data.")
        st.stop()

    # --- filter UI ---
    c1, c2, c3 = st.columns([1.2, 1.2, 1.2])
    with c1:
        jenis_opts = as_list_unique(df["jenis_pendekatan"])
        jenis_sel = st.multiselect("jenis_pendekatan", options=jenis_opts, default=jenis_opts)
    with c2:
        mid_sel = st.selectbox("modeling_id (opsional)", options=["(all)"] + as_list_unique(df["modeling_id"]), index=0)
    with c3:
        emb_sel = st.selectbox("embedding_run_id (opsional)", options=["(all)"] + as_list_unique(df["embedding_run_id"]), index=0)

    c4, c5, c6 = st.columns([1.2, 1.2, 1.2])
    with c4:
        temporal_sel = st.selectbox("temporal_id (opsional)", options=["(all)"] + as_list_unique(df["temporal_id"]), index=0)
    with c5:
        thr_mode = st.selectbox("Filter threshold", options=["(all)", "only -1.0 (semantik)", "only not -1.0 (sintaksis)"], index=0)
    with c6:
        q_notes = st.text_input("Cari notes (contains)", value="")

    dff = df.copy()
    if jenis_sel:
        dff = dff[dff["jenis_pendekatan"].isin(jenis_sel)]
    if mid_sel != "(all)":
        dff = dff[dff["modeling_id"] == mid_sel]
    if emb_sel != "(all)":
        dff = dff[dff["embedding_run_id"] == emb_sel]
    if temporal_sel != "(all)":
        dff = dff[dff["temporal_id"] == temporal_sel]
    if thr_mode == "only -1.0 (semantik)":
        dff = dff[dff["threshold"].fillna(-9999.0) == -1.0]
    elif thr_mode == "only not -1.0 (sintaksis)":
        dff = dff[dff["threshold"].fillna(-1.0) != -1.0]
    if q_notes.strip():
        dff = dff[dff["notes"].fillna("").str.contains(q_notes.strip(), case=False, regex=False)]

    # KPI
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows", f"{len(dff):,}")
    k2.metric("Silhouette (avg)", fmt_float(dff["silhouette_score"].mean(), 4))
    k3.metric("DBI (avg)", fmt_float(dff["dbi"].mean(), 4))
    k4.metric("DBCV (avg)", fmt_float(dff["dbcv"].mean(), 4))

    st.divider()

    # Charts
    ch_base = alt.Chart(dff).properties(height=280)

    ch_scatter = (
        ch_base.mark_circle(size=70)
        .encode(
            x=alt.X("dbi:Q", title="DBI (lebih kecil lebih baik)"),
            y=alt.Y("silhouette_score:Q", title="Silhouette (lebih besar lebih baik)"),
            tooltip=[
                "run_time:T", "jenis_pendekatan:N", "modeling_id:N", "embedding_run_id:N",
                "temporal_id:N", "threshold:Q", "silhouette_score:Q", "dbi:Q", "dbcv:Q"
            ],
            color="jenis_pendekatan:N",
        )
    )
    st.subheader("Scatter: Silhouette vs DBI")
    st.altair_chart(ch_scatter, use_container_width=True)

    if dff["dbcv"].notna().any():
        ch_dbcv = (
            ch_base.mark_line()
            .encode(
                x=alt.X("run_time:T", title="run_time"),
                y=alt.Y("dbcv:Q", title="DBCV"),
                color="jenis_pendekatan:N",
                tooltip=["run_time:T", "jenis_pendekatan:N", "dbcv:Q", "modeling_id:N", "embedding_run_id:N"],
            )
        )
        st.subheader("Trend: DBCV per waktu")
        st.altair_chart(ch_dbcv, use_container_width=True)

    st.divider()

    # Table
    cols = [
        "run_time", "jenis_pendekatan", "job_id", "modeling_id", "embedding_run_id",
        "temporal_id", "threshold", "silhouette_score", "dbi", "dbcv"
    ]
    if show_notes:
        cols.append("notes")
    if show_meta:
        cols.append("meta_json")

    st.subheader("Tabel Hasil Evaluasi (Gabungan)")
    st.dataframe(dff[cols], use_container_width=True, height=480)

    # download
    st.download_button(
        "⬇️ Download CSV (hasil terfilter)",
        data=dff[cols].to_csv(index=False).encode("utf-8"),
        file_name="evaluation_results_filtered.csv",
        mime="text/csv",
    )

# ======================================================
# TAB 2 — DBCV SEMANTIK RUNS
# ======================================================
with tabs[1]:
    df_runs = load_semantik_runs(engine, limit=min(int(max_rows), 5000))

    if df_runs.empty:
        st.warning(f"Tabel {SCHEMA}.{T_SEM_RUNS} kosong / belum ada data.")
        st.stop()

    c1, c2, c3 = st.columns([1.2, 1.2, 1.2])
    with c1:
        mid_sel = st.selectbox("modeling_id", options=["(all)"] + as_list_unique(df_runs["modeling_id"]), index=0)
    with c2:
        emb_sel = st.selectbox("embedding_run_id", options=["(all)"] + as_list_unique(df_runs["embedding_run_id"]), index=0)
    with c3:
        only_has_dbcv = st.checkbox("Hanya yang sudah ada DBCV", value=True)

    dff = df_runs.copy()
    if mid_sel != "(all)":
        dff = dff[dff["modeling_id"] == mid_sel]
    if emb_sel != "(all)":
        dff = dff[dff["embedding_run_id"] == emb_sel]
    if only_has_dbcv:
        dff = dff[dff["dbcv"].notna()]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows", f"{len(dff):,}")
    k2.metric("DBCV (avg)", fmt_float(dff["dbcv"].mean(), 4))
    k3.metric("n_clusters (avg)", fmt_float(dff["n_clusters"].mean(), 2))
    k4.metric("noise (avg)", fmt_float(dff["n_noise"].mean(), 2))

    st.divider()

    ch = (
        alt.Chart(dff).mark_circle(size=80)
        .encode(
            x=alt.X("n_clusters:Q", title="n_clusters"),
            y=alt.Y("dbcv:Q", title="DBCV"),
            tooltip=[
                "run_time:T", "modeling_id:N", "embedding_run_id:N",
                "dbcv:Q", "n_clusters:Q", "n_noise:Q", "silhouette:Q", "dbi:Q"
            ],
        )
        .properties(height=320)
    )
    st.subheader("Scatter: DBCV vs n_clusters")
    st.altair_chart(ch, use_container_width=True)

    cols = [
        "run_time", "modeling_id", "embedding_run_id",
        "n_rows", "n_clusters", "n_noise",
        "silhouette", "dbi", "dbcv"
    ]
    if show_notes:
        cols.append("notes")

    st.subheader("Tabel Runs Semantik (HDBSCAN)")
    st.dataframe(dff[cols], use_container_width=True, height=520)

    st.download_button(
        "⬇️ Download CSV (runs semantik terfilter)",
        data=dff[cols].to_csv(index=False).encode("utf-8"),
        file_name="semantik_runs_dbcv_filtered.csv",
        mime="text/csv",
    )

# ======================================================
# TAB 3 — COMPARE
# ======================================================
with tabs[2]:
    st.caption("Pilih beberapa baris untuk dibandingkan (dari evaluasi gabungan).")
    df = load_eval_table(engine, limit=min(int(max_rows), 5000))
    if df.empty:
        st.warning("Tabel evaluasi gabungan kosong.")
        st.stop()

    # pilih subset supaya ringan
    df_small = df.copy()
    df_small["key"] = (
        df_small["jenis_pendekatan"].astype(str) + " | " +
        df_small["modeling_id"].astype(str) + " | " +
        df_small["embedding_run_id"].astype(str) + " | " +
        df_small["temporal_id"].fillna("").astype(str) + " | " +
        df_small["threshold"].fillna(-9999).astype(str)
    )

    keys = as_list_unique(df_small["key"])
    sel = st.multiselect("Pilih run untuk dibandingkan", options=keys, default=keys[: min(5, len(keys))])

    dff = df_small[df_small["key"].isin(sel)].copy()
    if dff.empty:
        st.info("Belum ada yang dipilih.")
        st.stop()

    st.subheader("Tabel Compare")
    cols = ["run_time", "jenis_pendekatan", "modeling_id", "embedding_run_id", "temporal_id",
            "threshold", "silhouette_score", "dbi", "dbcv", "notes"]
    cols = [c for c in cols if c in dff.columns]
    st.dataframe(dff[cols].sort_values("run_time", ascending=False), use_container_width=True, height=420)

    st.subheader("Radar-like (Bar) Compare")
    melt = dff.melt(
        id_vars=["key"],
        value_vars=[c for c in ["silhouette_score", "dbi", "dbcv"] if c in dff.columns],
        var_name="metric",
        value_name="value",
    ).dropna()

    ch = (
        alt.Chart(melt).mark_bar()
        .encode(
            x=alt.X("key:N", title="run", sort="-y"),
            y=alt.Y("value:Q", title="value"),
            column=alt.Column("metric:N", title=None),
            tooltip=["key:N", "metric:N", "value:Q"],
        )
        .properties(height=280)
    )
    st.altair_chart(ch, use_container_width=True)

# Sidebar tab switching (biar UX enak)
if tab_default == "Evaluasi Gabungan":
    st.session_state["_tab_hint"] = 0
elif tab_default == "DBCV Semantik (Runs)":
    st.session_state["_tab_hint"] = 1
else:
    st.session_state["_tab_hint"] = 2
