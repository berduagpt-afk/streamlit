# pages/evaluasi_clustering_metrics.py
# Evaluasi Clustering — Silhouette Score & Davies-Bouldin Index (DBI)
# Membaca: lasis_djp.modeling_runs, lasis_djp.cluster_members
#
# FIX:
# - modeling_runs tidak wajib punya kolom "modul"
# - evaluasi per-modul dibaca dari cluster_members (jika kolom modul tersedia)

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sqlalchemy import create_engine, text
from sklearn.metrics import silhouette_score, davies_bouldin_score

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
T_RUNS = "modeling_runs"
T_MEMBERS = "cluster_members"

#st.set_page_config(page_title="Evaluasi Clustering (Silhouette & DBI)", layout="wide")

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

engine = get_engine()

# ======================================================
# 🧠 Helpers
# ======================================================
def _safe_json_to_vec(x) -> np.ndarray | None:
    """Parse embedding/vector yang mungkin berupa list, string JSON, atau None."""
    if x is None:
        return None
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x, dtype=float)
        return arr
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            obj = json.loads(s)
            arr = np.asarray(obj, dtype=float)
            return arr
        except Exception:
            # fallback untuk format array Postgres seperti "{0.1,0.2}"
            try:
                s2 = s.replace("{", "[").replace("}", "]")
                obj = json.loads(s2)
                arr = np.asarray(obj, dtype=float)
                return arr
            except Exception:
                return None
    return None


def pick_label_col(df: pd.DataFrame) -> str:
    """Cari nama kolom label cluster yang umum."""
    candidates = ["cluster_id", "cluster", "label", "cluster_label"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError("Kolom label cluster tidak ditemukan. Harus ada salah satu: cluster_id / cluster / label / cluster_label.")


def pick_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, str, np.ndarray]:
    """
    Pilih matrix fitur X dari kolom yang tersedia.
    Return:
      X (np.ndarray), sumber_fitur (str), mask_valid (np.ndarray bool)
    mask_valid dipakai kalau ada baris embedding yang gagal diparse.
    """
    # 1) embedding / vector (kolom umum)
    candidate_vec_cols = [
        "embedding", "embeddings",
        "vector", "vectors",
        "text_embedding", "sentence_embedding",
        "vec", "vec_json", "embedding_json"
    ]
    for c in candidate_vec_cols:
        if c in df.columns:
            vecs = df[c].apply(_safe_json_to_vec)
            ok = vecs.notna().to_numpy()
            if ok.mean() >= 0.95 and ok.sum() >= 3:
                X = np.vstack(vecs[ok].to_list())
                return X, c, ok

    # 2) UMAP 2D
    if {"umap_x", "umap_y"}.issubset(df.columns):
        X = df[["umap_x", "umap_y"]].astype(float).to_numpy()
        ok = np.ones(len(df), dtype=bool)
        return X, "umap_x/umap_y", ok

    # 3) t-SNE 2D
    if {"tsne_x", "tsne_y"}.issubset(df.columns):
        X = df[["tsne_x", "tsne_y"]].astype(float).to_numpy()
        ok = np.ones(len(df), dtype=bool)
        return X, "tsne_x/tsne_y", ok

    raise ValueError(
        "Tidak menemukan fitur numerik untuk evaluasi. "
        "Pastikan cluster_members menyimpan salah satu: embedding/vector, umap_x&umap_y, atau tsne_x&tsne_y."
    )


def to_int_labels(s: pd.Series) -> np.ndarray:
    """Ubah label ke int; kalau string campur, mapping jadi int."""
    try:
        return s.astype(int).to_numpy()
    except Exception:
        uniq = s.astype(str).unique().tolist()
        mapping = {k: i for i, k in enumerate(sorted(uniq))}
        return s.astype(str).map(mapping).astype(int).to_numpy()


def compute_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    """
    Hitung Silhouette & DBI dengan guard.
    Silhouette: lebih tinggi lebih baik
    DBI: lebih rendah lebih baik
    """
    note = ""
    if X is None or labels is None or len(labels) != X.shape[0]:
        return {"silhouette": np.nan, "dbi": np.nan, "note": "Ukuran X dan label tidak selaras."}

    if X.shape[0] < 3:
        return {"silhouette": np.nan, "dbi": np.nan, "note": "Jumlah data terlalu sedikit untuk evaluasi."}

    unique = np.unique(labels)
    if len(unique) < 2:
        return {"silhouette": np.nan, "dbi": np.nan, "note": "Hanya 1 cluster (atau semua label sama)."}

    try:
        sil = float(silhouette_score(X, labels, metric="euclidean"))
    except Exception as e:
        sil = np.nan
        note += f"Silhouette gagal: {e} | "

    try:
        dbi = float(davies_bouldin_score(X, labels))
    except Exception as e:
        dbi = np.nan
        note += f"DBI gagal: {e}"

    return {"silhouette": sil, "dbi": dbi, "note": note.strip(" |")}


def fmt(x) -> str:
    return "-" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.4f}"


# ======================================================
# 🧾 UI — Header
# ======================================================
st.title("📊 Evaluasi Modeling Clustering")
st.caption("Metrik internal: Silhouette Score (↑ lebih baik) dan Davies–Bouldin Index / DBI (↓ lebih baik).")

# ======================================================
# 🔎 Sidebar — pilih run
# ======================================================
with st.sidebar:
    st.header("Filter Run")
    max_runs = st.number_input("Maks. run ditampilkan", min_value=10, max_value=5000, value=300, step=10)

    # FIX: jangan select kolom yang belum tentu ada (modul)
    q_runs = text(f"""
        SELECT
            run_id, approach, run_time, threshold, window_days, notes
        FROM {SCHEMA}.{T_RUNS}
        ORDER BY run_time DESC
        LIMIT :max_runs
    """)
    runs = pd.read_sql(q_runs, engine, params={"max_runs": int(max_runs)})

    if runs.empty:
        st.warning("Tabel modeling_runs kosong atau belum ada run.")
        st.stop()

    approach_opts = ["(Semua)"] + sorted([x for x in runs["approach"].dropna().unique().tolist()])
    approach = st.selectbox("Approach", approach_opts, index=0)

    runs_f = runs[runs["approach"] == approach].copy() if approach != "(Semua)" else runs.copy()

    def run_label(r):
        rt = pd.to_datetime(r["run_time"]).strftime("%Y-%m-%d %H:%M:%S") if pd.notna(r["run_time"]) else "-"
        thr = r.get("threshold", None)
        win = r.get("window_days", None)
        # notes optional
        notes = r.get("notes", "")
        notes = "" if pd.isna(notes) else str(notes)
        notes_short = (notes[:40] + "…") if len(notes) > 40 else notes
        return f"{r['run_id']} | {r['approach']} | {rt} | thr={thr} | win={win} | {notes_short}"

    runs_f["label"] = runs_f.apply(run_label, axis=1)

    chosen = st.selectbox("Pilih run", runs_f["label"].tolist(), index=0)
    run_id = chosen.split("|")[0].strip()

    st.divider()
    ignore_noise = st.checkbox("Abaikan noise label -1 (HDBSCAN)", value=True)
    per_modul = st.checkbox("Hitung juga per-modul (jika kolom modul tersedia di cluster_members)", value=True)
    sample_cap = st.number_input("Batas sampel evaluasi (0 = tanpa batas)", min_value=0, max_value=200000, value=0, step=1000)

# ======================================================
# 📥 Load cluster members
# ======================================================
st.subheader("Data Cluster Members")

q_members = text(f"""
    SELECT *
    FROM {SCHEMA}.{T_MEMBERS}
    WHERE run_id = :run_id
""")
dfm = pd.read_sql(q_members, engine, params={"run_id": run_id})

if dfm.empty:
    st.error("Tidak ada data di cluster_members untuk run_id tersebut.")
    st.stop()

# sampling
if sample_cap and int(sample_cap) > 0 and len(dfm) > int(sample_cap):
    dfm = dfm.sample(int(sample_cap), random_state=42).reset_index(drop=True)

# pick label column
try:
    label_col = pick_label_col(dfm)
except Exception as e:
    st.error(str(e))
    st.stop()

labels_all = to_int_labels(dfm[label_col])

# pick feature matrix
try:
    X_all, feat_src, ok_mask = pick_feature_matrix(dfm)
except Exception as e:
    st.error(str(e))
    st.stop()

# align labels with feature rows (khusus embedding parsing)
dfm_eval = dfm.copy()
dfm_eval["_label_int"] = labels_all

if ok_mask is not None and ok_mask.sum() != len(ok_mask):
    dfm_eval = dfm_eval.loc[ok_mask].reset_index(drop=True)
    labels_all = dfm_eval["_label_int"].to_numpy()

# Noise count (sebelum filter ignore_noise)
noise_count = int((labels_all == -1).sum())

# header metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Run ID", run_id)
c2.metric("Rows (setelah validasi fitur)", f"{len(dfm_eval):,}")
c3.metric("Cluster unik", f"{pd.Series(labels_all).nunique():,}")
c4.metric("Noise (-1)", f"{noise_count:,}")

with st.expander("Lihat sample data (cluster_members)", expanded=False):
    st.dataframe(dfm_eval.head(50), use_container_width=True)

# ======================================================
# ✅ Compute overall metrics
# ======================================================
st.subheader("Metrik Evaluasi (Global)")

# Apply ignore noise
mask = np.ones(len(dfm_eval), dtype=bool)
if ignore_noise:
    mask = labels_all != -1

# X_all harus sesuai dfm_eval:
# - jika feat_src embedding: X_all sudah vstack dari ok_mask
# - jika umap/tsne: X_all diambil dari dfm, tetapi dfm_eval sudah ok_mask (all True), jadi aman
if feat_src in ["umap_x/umap_y", "tsne_x/tsne_y"]:
    # rebuild X from dfm_eval (lebih aman)
    if feat_src == "umap_x/umap_y":
        X_use = dfm_eval[["umap_x", "umap_y"]].astype(float).to_numpy()
    else:
        X_use = dfm_eval[["tsne_x", "tsne_y"]].astype(float).to_numpy()
else:
    X_use = X_all  # sudah sejalan dengan dfm_eval

X2 = X_use[mask]
y2 = labels_all[mask]

res = compute_metrics(X2, y2)

k1, k2, k3 = st.columns([1, 1, 2])
k1.metric("Silhouette Score", fmt(res["silhouette"]))
k2.metric("Davies–Bouldin Index (DBI)", fmt(res["dbi"]))
k3.write(f"**Sumber fitur:** `{feat_src}`")

if res.get("note"):
    st.info(res["note"])

# ======================================================
# 📈 Distribusi ukuran cluster
# ======================================================
st.subheader("Distribusi Ukuran Cluster")

tmp = pd.DataFrame({"cluster_id": labels_all})
if ignore_noise:
    tmp = tmp[tmp["cluster_id"] != -1].copy()

sizes = tmp.value_counts("cluster_id").reset_index()
sizes.columns = ["cluster_id", "n"]
sizes = sizes.sort_values("n", ascending=False)

if sizes.empty:
    st.warning("Tidak ada cluster untuk ditampilkan (mungkin semua data noise).")
else:
    topk = min(30, len(sizes))
    chart = (
        alt.Chart(sizes.head(topk))
        .mark_bar()
        .encode(
            x=alt.X("cluster_id:O", sort="-y", title="Cluster ID"),
            y=alt.Y("n:Q", title="Jumlah anggota"),
            tooltip=["cluster_id", "n"],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(sizes.head(200), use_container_width=True)

# ======================================================
# 🧩 Per-modul metrics (opsional)
# ======================================================
if per_modul and ("modul" in dfm_eval.columns):
    st.subheader("Metrik Evaluasi per Modul")

    rows = []
    for mod, g in dfm_eval.groupby("modul"):
        y = g["_label_int"].to_numpy()

        # build X per group sesuai sumber fitur
        if feat_src == "umap_x/umap_y" and {"umap_x", "umap_y"}.issubset(g.columns):
            Xg = g[["umap_x", "umap_y"]].astype(float).to_numpy()
        elif feat_src == "tsne_x/tsne_y" and {"tsne_x", "tsne_y"}.issubset(g.columns):
            Xg = g[["tsne_x", "tsne_y"]].astype(float).to_numpy()
        else:
            # embedding/vector column name == feat_src
            if feat_src in g.columns:
                vecs = g[feat_src].apply(_safe_json_to_vec)
                ok = vecs.notna().to_numpy()
                if ok.sum() < 3:
                    continue
                Xg = np.vstack(vecs[ok].to_list())
                y = y[ok]
            else:
                continue

        if ignore_noise:
            m = y != -1
            Xg = Xg[m]
            y = y[m]

        r = compute_metrics(Xg, y)

        rows.append({
            "modul": mod,
            "n_rows": int(len(g)),
            "n_eval": int(len(y)),
            "n_clusters": int(pd.Series(y).nunique()) if len(y) else 0,
            "silhouette": r["silhouette"],
            "dbi": r["dbi"],
        })

    df_mod = pd.DataFrame(rows)

    if df_mod.empty:
        st.warning("Tidak cukup data per-modul untuk menghitung metrik (butuh >=2 cluster dan sampel memadai).")
    else:
        df_mod = df_mod.sort_values(["silhouette", "dbi"], ascending=[False, True])

        cA, cB = st.columns(2)
        with cA:
            ch1 = (
                alt.Chart(df_mod.head(40))
                .mark_bar()
                .encode(
                    x=alt.X("silhouette:Q", title="Silhouette Score (↑)"),
                    y=alt.Y("modul:N", sort="-x", title="Modul"),
                    tooltip=[
                        "modul", "n_rows", "n_eval", "n_clusters",
                        alt.Tooltip("silhouette:Q", format=".4f"),
                        alt.Tooltip("dbi:Q", format=".4f")
                    ],
                )
                .properties(height=460)
            )
            st.altair_chart(ch1, use_container_width=True)

        with cB:
            ch2 = (
                alt.Chart(df_mod.head(40))
                .mark_bar()
                .encode(
                    x=alt.X("dbi:Q", title="DBI (↓)"),
                    y=alt.Y("modul:N", sort="x", title="Modul"),
                    tooltip=[
                        "modul", "n_rows", "n_eval", "n_clusters",
                        alt.Tooltip("silhouette:Q", format=".4f"),
                        alt.Tooltip("dbi:Q", format=".4f")
                    ],
                )
                .properties(height=460)
            )
            st.altair_chart(ch2, use_container_width=True)

        st.dataframe(df_mod, use_container_width=True)

        csv = df_mod.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download metrik per-modul (CSV)",
            data=csv,
            file_name=f"metrics_per_modul_run_{run_id}.csv",
            mime="text/csv",
        )
else:
    if per_modul:
        st.info("Evaluasi per-modul dilewati karena kolom `modul` tidak tersedia di cluster_members.")

# ======================================================
# 🧾 Interpretasi singkat
# ======================================================
with st.expander("Cara membaca metrik (ringkas)"):
    st.markdown(
        """
- **Silhouette Score** (rentang -1 s/d 1): makin mendekati **1** → cluster makin “rapi”.
- **DBI** (>=0): makin mendekati **0** → kualitas cluster makin baik.
- Untuk algoritma yang menghasilkan noise (mis. HDBSCAN), evaluasi internal sering lebih stabil jika **noise `-1` diabaikan**.
        """
    )
