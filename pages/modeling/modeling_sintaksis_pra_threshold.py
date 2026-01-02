# pages/modeling_sintaksis_cosine_dist_viewer.py
# ============================================================
# Streamlit Viewer — Cosine Similarity Distribution (Pra-Threshold)
# Sumber:
# - lasis_djp.modeling_sintaksis_cosine_dist (hist_json, stats_json)
#
# Fitur:
# 1) Histogram cosine similarity (pra-threshold) dari hist_json
# 2) Tabel kuantil + statistik ringkas dari stats_json
# 3) Overlay garis threshold (0.6/0.7/0.8/0.9) untuk justifikasi
# ============================================================

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import altair as alt
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ============================================================
# Guard login (sesuaikan dengan proyek kamu)
# ============================================================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login terlebih dahulu.")
    st.stop()

SCHEMA = "lasis_djp"
T_DIST = "modeling_sintaksis_cosine_dist"

DEFAULT_THRESHOLDS = [0.6, 0.7, 0.8, 0.9]


# ============================================================
# DB connection (mengikuti pola secrets.toml proyek kamu)
# ============================================================
@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    cfg = st.secrets["connections"]["postgres"]
    url = (
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )
    return create_engine(url, pool_pre_ping=True)


# ============================================================
# Helpers
# ============================================================
def _as_dict(x: Any) -> Dict[str, Any]:
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            obj = json.loads(x)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def load_dist_runs(engine: Engine) -> pd.DataFrame:
    q = text(f"""
        SELECT
            job_id,
            tfidf_run_id,
            run_time,
            knn_k,
            n_rows,
            n_pairs,
            stats_json,
            hist_json,
            sample_note
        FROM {SCHEMA}.{T_DIST}
        ORDER BY run_time DESC
        LIMIT 200
    """)
    with engine.begin() as conn:
        df = pd.read_sql(q, conn)

    if not df.empty:
        df["run_time"] = pd.to_datetime(df["run_time"], errors="coerce")
    return df


def build_hist_df(hist_json: Dict[str, Any]) -> pd.DataFrame:
    """
    hist_json:
      - bin_edges: list length (bins+1)
      - counts: list length bins
    Kita ubah ke tabel dengan bin_left, bin_right, bin_mid, count.
    """
    edges = hist_json.get("bin_edges", []) or []
    counts = hist_json.get("counts", []) or []

    if len(edges) < 2 or len(counts) < 1:
        return pd.DataFrame(columns=["bin_left", "bin_right", "bin_mid", "count"])

    # safe: bins = min(len(counts), len(edges)-1)
    bins = min(len(counts), len(edges) - 1)
    rows = []
    for i in range(bins):
        left = float(edges[i])
        right = float(edges[i + 1])
        mid = (left + right) / 2.0
        rows.append({"bin_left": left, "bin_right": right, "bin_mid": mid, "count": int(counts[i])})

    return pd.DataFrame(rows)


def build_quantile_df(stats_json: Dict[str, Any]) -> pd.DataFrame:
    q = stats_json.get("quantiles", {}) or {}
    # q keys bisa string "0.25", dll
    rows = []
    for k, v in q.items():
        try:
            qf = float(k)
        except Exception:
            continue
        try:
            vf = float(v)
        except Exception:
            continue
        rows.append({"quantile": qf, "cosine_similarity": vf})

    dfq = pd.DataFrame(rows)
    if not dfq.empty:
        dfq = dfq.sort_values("quantile").reset_index(drop=True)
        dfq["quantile"] = dfq["quantile"].map(lambda x: f"{x:.2f}")
    return dfq


def build_summary_kpis(stats_json: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "mean": stats_json.get("mean", None),
        "std": stats_json.get("std", None),
        "min": stats_json.get("min", None),
        "max": stats_json.get("max", None),
        "n_pairs_total": stats_json.get("n_pairs_total", None),
        "n_pairs_used": stats_json.get("n_pairs_used", None),
        "sample_note": stats_json.get("sample_note", ""),
        "pair_mode": stats_json.get("pair_mode", ""),
        "knn_k_effective": stats_json.get("knn_k_effective", None),
    }


# ============================================================
# UI
# ============================================================
st.title("Distribusi Cosine Similarity Pra-Threshold")
st.caption(
    "Halaman ini menampilkan histogram nilai cosine similarity kandidat (berbasis kNN), "
    "tabel kuantil, serta overlay garis threshold untuk mendukung justifikasi pemilihan threshold."
)

engine = get_engine()
df_runs = load_dist_runs(engine)

if df_runs.empty:
    st.warning(f"Belum ada data pada {SCHEMA}.{T_DIST}. Jalankan script offline yang menyimpan cosine_dist terlebih dahulu.")
    st.stop()

# Sidebar controls
with st.sidebar:
    st.header("Filter & Parameter")

    # pilih job_id / run
    # tampilkan label ringkas
    def label_row(r):
        rt = r["run_time"]
        rt_s = rt.strftime("%Y-%m-%d %H:%M:%S") if pd.notnull(rt) else "NA"
        return f"{rt_s} | job={str(r['job_id'])[:8]}… | tfidf={str(r['tfidf_run_id'])[:12]}… | k={r['knn_k']} | rows={r['n_rows']}"

    options = df_runs.to_dict("records")
    labels = [label_row(r) for r in options]

    sel_idx = st.selectbox(
        "Pilih hasil distribusi (job_id)",
        options=list(range(len(options))),
        format_func=lambda i: labels[i],
        index=0,
    )
    sel = options[sel_idx]

    thresholds = st.multiselect(
        "Overlay threshold",
        options=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
        default=DEFAULT_THRESHOLDS,
    )

    bins_override = st.checkbox("Override jumlah bins (visual saja)", value=False)
    viz_bins = st.slider("Bins", min_value=10, max_value=200, value=50, step=5) if bins_override else None

# Parse JSONs
stats_json = _as_dict(sel.get("stats_json"))
hist_json = _as_dict(sel.get("hist_json"))

# Jika user override bins, kita re-bin dari bin_mid approximations tidak ideal;
# jadi override hanya untuk chart step: kita pakai data existing saja. (lebih aman)
hist_df = build_hist_df(hist_json)
q_df = build_quantile_df(stats_json)
kpis = build_summary_kpis(stats_json)

# Header info
c1, c2, c3, c4 = st.columns(4)
c1.metric("TF-IDF Run ID", str(sel.get("tfidf_run_id")))
c2.metric("kNN k", int(sel.get("knn_k") or 0))
c3.metric("Jumlah Tiket (rows)", f"{int(sel.get('n_rows') or 0):,}".replace(",", "."))
c4.metric("Jumlah Pasangan (kandidat)", f"{int(sel.get('n_pairs') or 0):,}".replace(",", "."))

st.divider()

# KPI stats
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Mean", f"{kpis['mean']:.4f}" if isinstance(kpis["mean"], (int, float)) else "-")
k2.metric("Std", f"{kpis['std']:.4f}" if isinstance(kpis["std"], (int, float)) else "-")
k3.metric("Min", f"{kpis['min']:.4f}" if isinstance(kpis["min"], (int, float)) else "-")
k4.metric("Max", f"{kpis['max']:.4f}" if isinstance(kpis["max"], (int, float)) else "-")
k5.metric("Sample", str(kpis.get("sample_note") or ""))

with st.expander("Metadata perhitungan (untuk penulisan tesis)", expanded=False):
    st.write({
        "pair_mode": kpis.get("pair_mode"),
        "knn_k_effective": kpis.get("knn_k_effective"),
        "n_pairs_total": kpis.get("n_pairs_total"),
        "n_pairs_used": kpis.get("n_pairs_used"),
        "catatan": "Distribusi dihitung dari kandidat pasangan kNN (directed), bukan full pairwise O(N^2).",
    })

# ============================================================
# Histogram + threshold overlay
# ============================================================
st.subheader("Histogram Cosine Similarity (Pra-Threshold)")

if hist_df.empty:
    st.warning("hist_json kosong / tidak valid.")
else:
    base = alt.Chart(hist_df).mark_bar().encode(
        x=alt.X("bin_mid:Q", title="Cosine Similarity", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("count:Q", title="Jumlah pasangan (kandidat)"),
        tooltip=[
            alt.Tooltip("bin_left:Q", title="Bin left", format=".3f"),
            alt.Tooltip("bin_right:Q", title="Bin right", format=".3f"),
            alt.Tooltip("count:Q", title="Count"),
        ],
    ).properties(height=360)

    # overlay threshold lines (tanpa warna spesifik)
    thr_df = pd.DataFrame({"threshold": thresholds})
    thr_lines = (
        alt.Chart(thr_df)
        .mark_rule(strokeDash=[6, 4])
        .encode(
            x=alt.X("threshold:Q"),
            tooltip=[alt.Tooltip("threshold:Q", format=".2f")]
        )
    )

    thr_text = (
        alt.Chart(thr_df)
        .mark_text(align="left", dx=5, dy=-5)
        .encode(
            x="threshold:Q",
            y=alt.value(0),
            text=alt.Text("threshold:Q", format=".2f"),
        )
    )

    st.altair_chart((base + thr_lines + thr_text).interactive(), use_container_width=True)

    st.caption(
        "Garis vertikal putus-putus menunjukkan nilai threshold kandidat. "
        "Area histogram di kanan garis menggambarkan proporsi pasangan dengan kemiripan ≥ threshold."
    )

# ============================================================
# Quantiles table
# ============================================================
st.subheader("Tabel Kuantil Cosine Similarity (Pra-Threshold)")

if q_df.empty:
    st.info("stats_json tidak memiliki quantiles atau formatnya tidak sesuai.")
else:
    st.dataframe(q_df, use_container_width=True)

# ============================================================
# Quick justification helper (opsional)
# ============================================================
st.subheader("Ringkasan untuk Justifikasi Threshold (berbasis kuantil)")

if q_df.empty:
    st.write("Tambahkan quantiles pada stats_json agar ringkasan ini muncul.")
else:
    # coba ambil median dan P90/P95
    q_map = {float(row["quantile"]): float(row["cosine_similarity"]) for _, row in q_df.assign(
        quantile=q_df["quantile"].astype(float)
    ).iterrows()}

    median = q_map.get(0.50, None)
    p90 = q_map.get(0.90, None)
    p95 = q_map.get(0.95, None)

    cols = st.columns(3)
    cols[0].metric("Median (Q0.50)", f"{median:.4f}" if isinstance(median, (int, float)) else "-")
    cols[1].metric("Q0.90", f"{p90:.4f}" if isinstance(p90, (int, float)) else "-")
    cols[2].metric("Q0.95", f"{p95:.4f}" if isinstance(p95, (int, float)) else "-")

    st.markdown(
        """
**Interpretasi singkat (siap dipakai di Bab IV):**
- Distribusi cosine similarity pra-threshold menggambarkan sebaran kemiripan redaksional antar tiket (kandidat kNN).
- Nilai threshold dapat diposisikan relatif terhadap kuantil (mis. Q0.90/Q0.95) untuk menyeimbangkan **spesifisitas** klaster dan **fragmentasi** (singleton).
- Semakin tinggi threshold, semakin sedikit pasangan yang lolos (histogram sisi kanan garis makin kecil), sehingga klaster cenderung lebih kecil dan lebih homogen.
        """
    )

# ============================================================
# Debug view (optional)
# ============================================================
with st.expander("Lihat raw JSON (debug)", expanded=False):
    st.write("stats_json", stats_json)
    st.write("hist_json", hist_json)
