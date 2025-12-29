# pages/evaluasi_silhouette_dbi_sintaksis.py
# ======================================================
# Evaluasi Clustering (Silhouette + Davies-Bouldin Index) — Sintaksis
# - Dropdown pilih 1 run (modeling_id)
# - Ambil label cluster dari modeling_sintaksis_members
# - Ambil fitur TF-IDF dari incident_tfidf_vectors (tfidf_json)
# - Hitung:
#   1) Silhouette Score
#   2) Davies–Bouldin Index (DBI)
#
# Catatan penting:
# - Silhouette untuk cosine pada data besar bisa berat -> disediakan sampling.
# - DBI standar berbasis Euclidean. Untuk TF-IDF, kita pakai:
#   - Normalisasi L2 + TruncatedSVD (opsional) -> lalu DBI Euclidean pada ruang tereduksi.
# ======================================================

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from typing import Dict, Any, List, Tuple

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score, davies_bouldin_score


# ======================================================
# 🔐 Guard login (sesuaikan dengan app kamu)
# ======================================================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login terlebih dahulu.")
    st.stop()


SCHEMA = "lasis_djp"
T_RUNS = "modeling_sintaksis_runs"
T_MEMBERS = "modeling_sintaksis_members"
T_TFIDF = "incident_tfidf_vectors"
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
# 📦 Loaders (cache-safe: _engine)
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
      n_clusters_all,
      n_clusters_recurring,
      n_noise_tickets,
      params_json,
      notes
    FROM {SCHEMA}.{T_RUNS}
    ORDER BY run_time DESC
    LIMIT :lim
    """
    df = pd.read_sql(text(sql), _engine, params={"lim": int(limit_runs)})
    df["run_time"] = pd.to_datetime(df["run_time"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_members_and_tfidf(
    _engine: Engine,
    modeling_id: str,
    tfidf_run_id: str,
    include_noise: bool,
) -> pd.DataFrame:
    noise_clause = "" if include_noise else f"AND m.cluster_id <> {NOISE_ID}"
    sql = f"""
    SELECT
      m.incident_number,
      m.cluster_id,
      m.tgl_submit,
      m.is_recurring,
      v.tfidf_json
    FROM {SCHEMA}.{T_MEMBERS} m
    JOIN {SCHEMA}.{T_TFIDF} v
      ON v.incident_number = m.incident_number
     AND v.run_id::text = :rid
    WHERE m.modeling_id::text = :mid
      {noise_clause}
      AND v.tfidf_json IS NOT NULL
    ORDER BY m.tgl_submit NULLS LAST
    """
    df = pd.read_sql(text(sql), _engine, params={"mid": str(modeling_id), "rid": str(tfidf_run_id)})
    df["tgl_submit"] = pd.to_datetime(df["tgl_submit"], errors="coerce")
    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce").fillna(NOISE_ID).astype(int)
    return df


# ======================================================
# ✅ Build CSR matrix from tfidf_json list
# ======================================================
def build_csr_matrix(tfidf_json_list: List[Any]) -> Tuple[csr_matrix, Dict[str, int]]:
    vocab: Dict[str, int] = {}
    docs: List[Dict[str, float]] = []

    for x in tfidf_json_list:
        d = x if isinstance(x, dict) else {}
        docs.append(d)
        for term in d.keys():
            if term not in vocab:
                vocab[term] = len(vocab)

    indptr = [0]
    indices: List[int] = []
    data: List[float] = []

    for d in docs:
        for term, val in d.items():
            j = vocab.get(term)
            if j is None:
                continue
            try:
                fv = float(val)
            except Exception:
                continue
            if fv == 0.0:
                continue
            indices.append(j)
            data.append(fv)
        indptr.append(len(indices))

    X = csr_matrix(
        (np.array(data, dtype=np.float32),
         np.array(indices, dtype=np.int32),
         np.array(indptr, dtype=np.int32)),
        shape=(len(docs), len(vocab)),
        dtype=np.float32
    )
    return X, vocab


# ======================================================
# Metrics helpers
# ======================================================
def compute_metrics(
    X: csr_matrix,
    labels: np.ndarray,
    use_svd: bool,
    svd_components: int,
    silhouette_metric: str,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    - Silhouette:
        * 'cosine' bisa langsung pada sparse (tapi berat untuk n besar).
        * 'euclidean' biasanya lebih cepat.
    - DBI:
        * sklearn DBI berbasis euclidean. Kita hitung pada ruang (opsional) hasil SVD.
    """
    out: Dict[str, Any] = {}

    # minimal syarat
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return {"error": "Silhouette/DBI butuh >= 2 cluster."}

    # normalisasi L2 agar TF-IDF stabil
    Xn = X.copy()
    normalize(Xn, norm="l2", axis=1, copy=False)

    X_eval = Xn
    svd_model = None
    if use_svd:
        k = int(max(2, min(int(svd_components), Xn.shape[1] - 1)))
        svd_model = TruncatedSVD(n_components=k, random_state=int(random_state))
        X_eval = svd_model.fit_transform(Xn)  # -> dense (n,k)
        out["svd_components_used"] = k
        out["svd_explained_var_sum"] = float(np.sum(svd_model.explained_variance_ratio_))

    # Silhouette
    try:
        if silhouette_metric == "cosine":
            # jika X_eval sudah dense akibat SVD, tetap bisa metric cosine
            sil = silhouette_score(X_eval, labels, metric="cosine")
        else:
            sil = silhouette_score(X_eval, labels, metric="euclidean")
        out["silhouette_score"] = float(sil)
        out["silhouette_metric"] = silhouette_metric
    except Exception as e:
        out["silhouette_error"] = str(e)

    # DBI (euclidean) — standar
    try:
        # DBI butuh dense; kalau masih sparse, ubah dulu (untuk ukuran kecil) atau pakai SVD.
        if hasattr(X_eval, "toarray"):
            Xe = X_eval.toarray()
        else:
            Xe = X_eval
        dbi = davies_bouldin_score(Xe, labels)
        out["dbi"] = float(dbi)
        out["dbi_metric"] = "euclidean"
    except Exception as e:
        out["dbi_error"] = str(e)

    return out


# ======================================================
# UI
# ======================================================
st.title("📊 Evaluasi Clustering: Silhouette Score & Davies–Bouldin Index (DBI)")
st.caption(
    "Halaman ini mengevaluasi kualitas cluster hasil modeling sintaksis. "
    "Gunakan sampling bila data besar agar perhitungan Silhouette tidak berat."
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

pick = st.selectbox("Pilih run (modeling_id)", runs["label"].tolist(), index=0)
row = runs.loc[runs["label"] == pick].iloc[0]

modeling_id = str(row["modeling_id"])
tfidf_run_id = str(row["tfidf_run_id"]) if pd.notna(row["tfidf_run_id"]) else ""

# Sidebar settings
st.sidebar.header("⚙️ Parameter Evaluasi")
include_noise = st.sidebar.checkbox("Include noise (cluster_id=-1)", value=False)
only_recurring = st.sidebar.checkbox("Hanya recurring (is_recurring=1)", value=True)

sample_n = st.sidebar.slider("Sampling N dokumen (untuk metric)", 200, 20000, 3000, 200)
random_state = st.sidebar.number_input("Random seed", value=42, step=1)

sil_metric = st.sidebar.selectbox("Silhouette metric", ["cosine", "euclidean"], index=0)

use_svd = st.sidebar.checkbox("Gunakan TruncatedSVD (recommended untuk DBI & speed)", value=True)
svd_k = st.sidebar.slider("SVD components", 20, 300, 80, 10)

min_cluster_size_eval = st.sidebar.slider("Min size cluster ikut evaluasi", 2, 200, 5, 1)

st.sidebar.markdown("---")
st.sidebar.write("ℹ️ Interpretasi singkat:")
st.sidebar.write("- Silhouette: semakin mendekati 1 semakin bagus; ~0 overlap; negatif buruk.")
st.sidebar.write("- DBI: semakin kecil semakin bagus.")


# Run info
c1, c2, c3, c4 = st.columns(4)
c1.metric("Modeling ID", modeling_id[:8] + "…")
c2.metric("TFIDF run_id", (tfidf_run_id[:8] + "…") if tfidf_run_id else "-")
c3.metric("Threshold", float(row["threshold"]) if pd.notna(row.get("threshold")) else np.nan)
c4.metric("Window days", int(row["window_days"]) if pd.notna(row.get("window_days")) else 0)

if not tfidf_run_id or tfidf_run_id.lower() == "none":
    st.error("tfidf_run_id kosong pada modeling_sintaksis_runs. Pastikan script offline menyimpan tfidf_run_id.")
    st.stop()

with st.spinner("Memuat label cluster + tfidf_json..."):
    df = load_members_and_tfidf(engine, modeling_id, tfidf_run_id, include_noise=include_noise)

if df.empty:
    st.warning("Data kosong (join members ↔ tfidf_vectors gagal atau tfidf_json null).")
    st.stop()

# optional: only recurring
df["is_recurring"] = pd.to_numeric(df["is_recurring"], errors="coerce").fillna(0).astype(int)
if only_recurring:
    df = df[df["is_recurring"] == 1]

if df.empty:
    st.warning("Data kosong setelah filter recurring/noise.")
    st.stop()

# filter cluster by size (untuk evaluasi)
sz = df.groupby("cluster_id")["incident_number"].size().reset_index(name="cluster_size")
keep_clusters = sz.loc[sz["cluster_size"] >= int(min_cluster_size_eval), "cluster_id"].tolist()
df_eval = df[df["cluster_id"].isin(keep_clusters)].copy()

if df_eval["cluster_id"].nunique() < 2:
    st.warning("Cluster yang memenuhi min_cluster_size_eval kurang dari 2. Turunkan min_cluster_size_eval.")
    st.stop()

# sampling dokumen
n_all = len(df_eval)
if n_all > sample_n:
    rs = np.random.RandomState(int(random_state))
    idx = rs.choice(df_eval.index.to_numpy(), size=int(sample_n), replace=False)
    df_eval = df_eval.loc[idx].copy()
    df_eval = df_eval.reset_index(drop=True)

# build matrix
with st.spinner("Membangun matriks TF-IDF..."):
    X, vocab = build_csr_matrix(df_eval["tfidf_json"].tolist())

labels = df_eval["cluster_id"].to_numpy(dtype=np.int64)

# compute metrics
with st.spinner("Menghitung Silhouette & DBI..."):
    res = compute_metrics(
        X=X,
        labels=labels,
        use_svd=bool(use_svd),
        svd_components=int(svd_k),
        silhouette_metric=str(sil_metric),
        random_state=int(random_state),
    )

# show results
st.subheader("Hasil Evaluasi")

if "error" in res:
    st.error(res["error"])
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Docs evaluated", len(df_eval))
m2.metric("Clusters evaluated", int(pd.Series(labels).nunique()))
m3.metric("Vocab size", len(vocab))
m4.metric("NNZ", int(X.nnz))

colA, colB = st.columns(2)

with colA:
    if "silhouette_score" in res:
        st.metric(f"Silhouette ({res.get('silhouette_metric')})", f"{res['silhouette_score']:.4f}")
    else:
        st.warning(f"Silhouette gagal: {res.get('silhouette_error')}")

with colB:
    if "dbi" in res:
        st.metric("Davies–Bouldin Index (lebih kecil lebih baik)", f"{res['dbi']:.4f}")
    else:
        st.warning(f"DBI gagal: {res.get('dbi_error')}")

if use_svd:
    st.info(
        f"SVD digunakan: k={res.get('svd_components_used')} | "
        f"Σ explained variance ratio ≈ {res.get('svd_explained_var_sum', 0.0):.4f}"
    )

# Distribution summary per cluster
st.subheader("Distribusi ukuran cluster (data yang dievaluasi)")
sz_eval = (
    pd.DataFrame({"cluster_id": labels})
    .value_counts()
    .reset_index(name="cluster_size")
    .sort_values("cluster_size", ascending=False)
)

# bar chart
chart = alt.Chart(sz_eval).mark_bar().encode(
    x=alt.X("cluster_id:N", sort="-y", title="Cluster ID"),
    y=alt.Y("cluster_size:Q", title="Jumlah dokumen (evaluated sample)"),
    tooltip=["cluster_id:N", "cluster_size:Q"],
).properties(height=320)

st.altair_chart(chart, use_container_width=True)

with st.expander("📋 Detail cluster size (top 50)"):
    st.dataframe(sz_eval.head(50), use_container_width=True)

with st.expander("🧾 Preview data evaluasi (sample)"):
    st.dataframe(
        df_eval[["incident_number", "cluster_id", "tgl_submit", "is_recurring"]].sort_values("tgl_submit").head(500),
        use_container_width=True
    )
