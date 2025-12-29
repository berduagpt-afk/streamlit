# pages/tfidf_unigram_viewer.py
# ======================================================
# Viewer + Grafik (t-SNE + Overlay Cluster Cosine + Temporal Bubble + Topic Summary)
# untuk hasil TF-IDF Unigram dari:
#   - lasis_djp.incident_modeling_tfidf_runs
#   - lasis_djp.incident_modeling_tfidf_vectors
#
# Fitur:
# ✅ Pilih run_id (terbaru)
# ✅ Filter modul/site/range tanggal + limit viewer
# ✅ Detail tiket + vektor TF-IDF (JSONB {feature: weight})
# ✅ Grafik:
#   1) Distribusi jumlah feature aktif per tiket
#   2) Top feature (doc frequency) dari sample viewer
#   3) t-SNE 2D dari TF-IDF sample
#   4) Overlay cluster cosine similarity threshold (connected components)
#   5) Cluster Summary: label, size, topic (top terms)
#   6) Temporal Bubble Chart: FULL data via DB (bukan sample t-SNE)
#   7) Ringkasan waktu per cluster (sample): first_seen, last_seen, duration_days
#
# Patch penting:
# - Cosine overlay menggunakan L2-normalize (lebih valid)
# - Perplexity t-SNE aman (< n_samples), handle sampel kecil
# - TSNE arg kompatibel (n_iter vs max_iter)
# - Overlay sample: pakai full cosine matrix tapi tanpa nested loop python
# - Temporal FULL: pakai graph-based clustering (radius graph / kNN) -> scalable (tanpa O(n^2) matrix)
# - Altair disable max rows
# - Bubble chart FULL mengikuti filter tanggal (mis. 2 tahun)
# ======================================================

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sqlalchemy import create_engine, text

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


# ======================================================
# Altair: hindari limit baris default
# ======================================================
alt.data_transformers.disable_max_rows()


# ======================================================
# 🔐 Guard login (ikuti pola app kamu)
# ======================================================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login terlebih dahulu.")
    st.stop()


# ======================================================
# ⚙️ Konstanta
# ======================================================
SCHEMA = "lasis_djp"
T_RUNS = "incident_modeling_tfidf_runs"
T_VEC = "incident_modeling_tfidf_vectors"


# ======================================================
# 🔌 DB Connection (secrets.toml)
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
# JSON safe
# ======================================================
def safe_json(v):
    """Normalize JSONB field into python dict."""
    if v is None:
        return {}
    if isinstance(v, dict):
        return v
    if isinstance(v, str) and v.strip():
        try:
            return json.loads(v)
        except Exception:
            return {}
    return {}


# ======================================================
# DB loaders
# ======================================================
@st.cache_data(show_spinner=False, ttl=120)
def load_runs(limit: int = 300) -> pd.DataFrame:
    sql = f"""
    SELECT run_id, run_time, approach, params_json, data_range, notes
    FROM {SCHEMA}.{T_RUNS}
    ORDER BY run_time DESC
    LIMIT :limit
    """
    df = pd.read_sql(text(sql), engine, params={"limit": int(limit)})
    df["params_json"] = df["params_json"].apply(safe_json)
    df["data_range"] = df["data_range"].apply(safe_json)
    return df


@st.cache_data(show_spinner=False, ttl=120)
def load_moduls(run_id: str) -> list[str]:
    sql = f"""
    SELECT DISTINCT modul
    FROM {SCHEMA}.{T_VEC}
    WHERE run_id = :run_id AND COALESCE(modul,'') <> ''
    ORDER BY modul
    """
    s = pd.read_sql(text(sql), engine, params={"run_id": run_id})
    return s["modul"].tolist()


@st.cache_data(show_spinner=False, ttl=120)
def load_sites(run_id: str) -> list[str]:
    sql = f"""
    SELECT DISTINCT site
    FROM {SCHEMA}.{T_VEC}
    WHERE run_id = :run_id AND COALESCE(site,'') <> ''
    ORDER BY site
    """
    s = pd.read_sql(text(sql), engine, params={"run_id": run_id})
    return s["site"].tolist()


@st.cache_data(show_spinner=False, ttl=120)
def load_vectors(
    run_id: str,
    modul: str | None,
    site: str | None,
    start: date | None,
    end: date | None,
    limit_rows: int,
) -> pd.DataFrame:
    where = ["run_id = :run_id"]
    params: dict[str, object] = {"run_id": run_id, "limit_rows": int(limit_rows)}

    if modul and modul != "Semua":
        where.append("modul = :modul")
        params["modul"] = modul

    if site and site != "Semua":
        where.append("site = :site")
        params["site"] = site

    if start:
        where.append("tgl_submit >= :start_ts")
        params["start_ts"] = pd.to_datetime(start)

    if end:
        where.append("tgl_submit < :end_ts")
        params["end_ts"] = pd.to_datetime(end) + pd.Timedelta(days=1)

    where_sql = " AND ".join(where)

    sql = f"""
    SELECT
      incident_number, tgl_submit, site, assignee, modul, sub_modul,
      text_sintaksis, tfidf_json
    FROM {SCHEMA}.{T_VEC}
    WHERE {where_sql}
    ORDER BY tgl_submit DESC
    LIMIT :limit_rows
    """
    df = pd.read_sql(text(sql), engine, params=params)
    df["tgl_submit"] = pd.to_datetime(df["tgl_submit"], errors="coerce")
    df["tfidf_json"] = df["tfidf_json"].apply(safe_json)
    return df


@st.cache_data(show_spinner=False, ttl=300)
def load_vectors_all(
    run_id: str,
    modul: str | None,
    site: str | None,
    start: date | None,
    end: date | None,
    max_n: int = 12000,
) -> pd.DataFrame:
    """
    Ambil data untuk analisis FULL (temporal) dengan safety limit.
    """
    where = ["run_id = :run_id"]
    params: dict[str, object] = {"run_id": run_id, "limit_rows": int(max_n)}

    if modul and modul != "Semua":
        where.append("modul = :modul")
        params["modul"] = modul

    if site and site != "Semua":
        where.append("site = :site")
        params["site"] = site

    if start:
        where.append("tgl_submit >= :start_ts")
        params["start_ts"] = pd.to_datetime(start)

    if end:
        where.append("tgl_submit < :end_ts")
        params["end_ts"] = pd.to_datetime(end) + pd.Timedelta(days=1)

    where_sql = " AND ".join(where)

    sql = f"""
    SELECT
      incident_number, tgl_submit, modul, sub_modul, site, assignee, tfidf_json
    FROM {SCHEMA}.{T_VEC}
    WHERE {where_sql}
    ORDER BY tgl_submit ASC
    LIMIT :limit_rows
    """
    df = pd.read_sql(text(sql), engine, params=params)
    df["tgl_submit"] = pd.to_datetime(df["tgl_submit"], errors="coerce")
    df["tfidf_json"] = df["tfidf_json"].apply(safe_json)
    return df


# ======================================================
# TF-IDF JSONB -> Dense matrix
# ======================================================
def build_dense_matrix_from_json(df: pd.DataFrame, max_features: int = 2000):
    """
    Convert tfidf_json dict into dense matrix [n_samples, n_features]
    Feature selection: top document frequency on this df sample.
    """
    dfreq: dict[str, int] = {}
    for v in df["tfidf_json"]:
        if isinstance(v, dict):
            for k in v.keys():
                kk = str(k)
                dfreq[kk] = dfreq.get(kk, 0) + 1

    if not dfreq:
        X = np.zeros((len(df), 1), dtype=np.float32)
        return X, ["__empty__"]

    top = sorted(dfreq.items(), key=lambda kv: -kv[1])[: int(max_features)]
    features = [k for k, _ in top]
    f_index = {f: i for i, f in enumerate(features)}

    X = np.zeros((len(df), len(features)), dtype=np.float32)
    for i, v in enumerate(df["tfidf_json"]):
        if not isinstance(v, dict):
            continue
        for k, w in v.items():
            j = f_index.get(str(k))
            if j is not None:
                try:
                    X[i, j] = float(w)
                except Exception:
                    pass

    return X, features


# ======================================================
# Union-Find
# ======================================================
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

    def groups(self):
        out = {}
        for i in range(len(self.parent)):
            r = self.find(i)
            out.setdefault(r, []).append(i)
        return out


# ======================================================
# Cosine threshold clustering (sample: full matrix, cepat tanpa nested loop)
# ======================================================
def cosine_threshold_clusters_fullmatrix(X_cos: np.ndarray, threshold: float, min_cluster_size: int = 2) -> np.ndarray:
    """
    Sample-only: gunakan cosine_similarity full matrix (O(n^2)), tapi cepat (np.argwhere).
    X_cos diasumsikan sudah L2-normalized.
    """
    n = int(X_cos.shape[0])
    if n <= 1:
        return np.zeros(n, dtype=int)

    S = cosine_similarity(X_cos)
    np.fill_diagonal(S, 0.0)

    uf = UnionFind(n)
    pairs = np.argwhere(S >= float(threshold))
    for i, j in pairs:
        i = int(i); j = int(j)
        if i < j:
            uf.union(i, j)

    groups = uf.groups()
    labels = np.full(n, -1, dtype=int)

    roots = sorted(groups.keys(), key=lambda r: (-len(groups[r]), r))
    label = 0
    for r in roots:
        members = groups[r]
        if len(members) < int(min_cluster_size):
            continue
        for idx in members:
            labels[idx] = label
        label += 1

    return labels


# ======================================================
# Cosine threshold clustering (scalable: radius graph / kNN)
# ======================================================
def cosine_threshold_clusters_graph(
    X_dense: np.ndarray,
    threshold: float,
    min_cluster_size: int = 2,
    method: str = "radius",   # "radius" | "knn"
    knn_k: int = 30,
) -> np.ndarray:
    """
    Graph-based connected components clustering untuk cosine threshold.
    Lebih scalable dibanding full cosine matrix.

    - X_dense: TF-IDF dense (belum dinormalisasi)
    - threshold: cosine similarity threshold
    - method:
        * "radius": connect jika cosine_dist <= 1-threshold (recommended)
        * "knn": ambil k tetangga terdekat lalu filter yang sim >= threshold
    """
    n = int(X_dense.shape[0])
    if n <= 1:
        return np.zeros(n, dtype=int)

    # L2 normalize supaya cosine distance valid
    X_cos = normalize(X_dense, norm="l2")

    uf = UnionFind(n)

    thr = float(threshold)
    radius = float(max(0.0, 1.0 - thr))  # cosine_dist <= 1-thr

    m = (method or "radius").lower().strip()

    if m == "radius":
        # radius graph: semua neighbor dalam radius (cosine distance)
        nn = NearestNeighbors(metric="cosine", algorithm="brute")
        nn.fit(X_cos)

        neigh_ind = nn.radius_neighbors(X_cos, radius=radius, return_distance=False)
        for i, js in enumerate(neigh_ind):
            if js is None:
                continue
            for j in js:
                j = int(j)
                if i < j:
                    uf.union(i, j)

    else:
        # kNN graph + filter threshold
        k = int(max(2, min(int(knn_k), n)))
        nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=k)
        nn.fit(X_cos)

        dists, inds = nn.kneighbors(X_cos, n_neighbors=k, return_distance=True)
        sims = 1.0 - dists

        for i in range(n):
            for jj in range(1, k):  # skip self at 0
                j = int(inds[i, jj])
                if j == i:
                    continue
                if float(sims[i, jj]) >= thr:
                    a, b = (i, j) if i < j else (j, i)
                    uf.union(a, b)

    groups = uf.groups()
    labels = np.full(n, -1, dtype=int)
    roots = sorted(groups.keys(), key=lambda r: (-len(groups[r]), r))

    label = 0
    for r in roots:
        members = groups[r]
        if len(members) < int(min_cluster_size):
            continue
        for idx in members:
            labels[idx] = label
        label += 1

    return labels


# ======================================================
# Cluster topic summary + temporal aggregation
# ======================================================
def compute_cluster_topics(
    df: pd.DataFrame,
    cluster_col: str = "cluster_label",
    tfidf_col: str = "tfidf_json",
    topk: int = 8,
    exclude_noise: bool = True,
) -> pd.DataFrame:
    """
    Output: Cluster Label, Cluster Size, Cluster Topic (top terms)
    Terms computed by sum TF-IDF weights per feature within cluster.
    """
    df2 = df.copy()
    if exclude_noise:
        df2 = df2[df2[cluster_col].astype(str).str.lower() != "noise"]

    if df2.empty:
        return pd.DataFrame(columns=["Cluster Label", "Cluster Size", "Cluster Topic"])

    rows = []
    for cl, g in df2.groupby(cluster_col):
        agg: dict[str, float] = {}
        for v in g[tfidf_col]:
            if isinstance(v, dict):
                for k, w in v.items():
                    try:
                        kk = str(k)
                        agg[kk] = agg.get(kk, 0.0) + float(w)
                    except Exception:
                        pass

        top = sorted(agg.items(), key=lambda kv: -kv[1])[: int(topk)] if agg else []
        topic = ", ".join([t for t, _ in top])
        rows.append({"Cluster Label": cl, "Cluster Size": int(len(g)), "Cluster Topic": topic})

    out = pd.DataFrame(rows).sort_values("Cluster Size", ascending=False).reset_index(drop=True)
    return out


def compute_cluster_temporal_counts(
    df: pd.DataFrame,
    cluster_col: str = "cluster_label",
    date_col: str = "tgl_submit",
    freq: str = "M",  # "M","W","D"
    exclude_noise: bool = True,
) -> pd.DataFrame:
    """
    Output: period (datetime), period_label (str), cluster_label, Count
    """
    df2 = df.copy()
    df2[date_col] = pd.to_datetime(df2[date_col], errors="coerce")
    df2 = df2.dropna(subset=[date_col])

    if exclude_noise:
        df2 = df2[df2[cluster_col].astype(str).str.lower() != "noise"]

    if df2.empty:
        return pd.DataFrame(columns=["period", "period_label", "cluster_label", "Count"])

    f = freq.upper()
    if f == "M":
        df2["period"] = df2[date_col].dt.to_period("M").dt.start_time
        df2["period_label"] = df2["period"].dt.strftime("%Y-%m")
    elif f == "W":
        df2["period"] = df2[date_col].dt.to_period("W").dt.start_time
        df2["period_label"] = df2["period"].dt.strftime("%Y-%m-%d")
    else:
        df2["period"] = df2[date_col].dt.floor("D")
        df2["period_label"] = df2["period"].dt.strftime("%Y-%m-%d")

    agg = (
        df2.groupby(["period", "period_label", cluster_col])["incident_number"]
        .count()
        .reset_index(name="Count")
        .rename(columns={cluster_col: "cluster_label"})
    )
    return agg


def cluster_ordering(labels: list[str]) -> list[str]:
    """Sort cluster labels C00, C01,..., noise last."""
    def key(cl):
        s = str(cl)
        if s.lower() == "noise":
            return (2, 10**9)
        if s.startswith("C"):
            try:
                return (0, int(s[1:]))
            except Exception:
                return (1, 10**8)
        return (1, 10**7)
    return sorted(labels, key=key)


# ======================================================
# Cached heavy computation: t-SNE + overlay (SAMPLE)
# ======================================================
@st.cache_data(show_spinner=True, ttl=600)
def compute_tsne_and_overlay(
    df_small: pd.DataFrame,
    max_features: int,
    perplexity: int,
    n_iter: int,
    do_overlay: bool,
    cosine_thr: float,
    min_cluster_size: int,
    use_scaler_for_tsne: bool = True,
) -> pd.DataFrame:
    X_dense, _ = build_dense_matrix_from_json(df_small, max_features=int(max_features))

    out = df_small.copy()
    if X_dense.size == 0 or np.allclose(X_dense, 0):
        out["tsne_x"] = 0.0
        out["tsne_y"] = 0.0
        out["cluster_id"] = -1
        out["cluster_label"] = "noise"
        return out

    n_samples = int(X_dense.shape[0])
    if n_samples < 3:
        out["tsne_x"] = 0.0
        out["tsne_y"] = 0.0
        out["cluster_id"] = -1
        out["cluster_label"] = "noise"
        return out

    # cosine space (VALID)
    X_cos = normalize(X_dense, norm="l2")

    # t-SNE space (opsional)
    if use_scaler_for_tsne:
        X_tsne = StandardScaler(with_mean=True, with_std=True).fit_transform(X_dense)
    else:
        X_tsne = X_dense

    # perplexity harus < n_samples
    perp = min(int(perplexity), max(1, n_samples - 1))
    if perp >= n_samples:
        perp = max(1, n_samples - 1)

    tsne_kwargs = dict(
        n_components=2,
        perplexity=perp,
        learning_rate="auto",
        init="pca",
        random_state=42,
    )

    try:
        tsne = TSNE(**tsne_kwargs, n_iter=int(n_iter))
    except TypeError:
        tsne = TSNE(**tsne_kwargs, max_iter=int(n_iter))

    emb = tsne.fit_transform(X_tsne)
    out["tsne_x"] = emb[:, 0]
    out["tsne_y"] = emb[:, 1]

    if do_overlay:
        labels = cosine_threshold_clusters_fullmatrix(
            X_cos,
            threshold=float(cosine_thr),
            min_cluster_size=int(min_cluster_size),
        )
        out["cluster_id"] = labels
        out["cluster_label"] = out["cluster_id"].apply(
            lambda x: "noise" if int(x) < 0 else f"C{int(x):02d}"
        )
    else:
        out["cluster_id"] = -1
        out["cluster_label"] = "off"

    return out


# ======================================================
# Temporal FULL via DB (scalable graph)
# ======================================================
@st.cache_data(show_spinner=True, ttl=900)
def load_temporal_counts_full_db(
    run_id: str,
    modul: str | None,
    site: str | None,
    start: date | None,
    end: date | None,
    freq: str,                 # "M","W","D"
    cosine_thr: float,
    min_cluster_size: int,
    max_features: int,
    max_n_full: int = 12000,
    exclude_noise: bool = True,
    graph_method: str = "radius",
    knn_k: int = 30,
) -> pd.DataFrame:
    """
    Bubble chart temporal untuk FULL data (sesuai filter tanggal) dengan graph-based clustering.
    Pipeline:
    1) load full (dibatasi max_n_full untuk keamanan)
    2) build dense matrix (vocab by doc freq pada full)
    3) cluster cosine threshold via radius graph / kNN graph (connected components)
    4) aggregate period x cluster
    """
    df_all = load_vectors_all(
        run_id=run_id,
        modul=modul,
        site=site,
        start=start,
        end=end,
        max_n=max_n_full,
    )

    if df_all.empty:
        return pd.DataFrame(columns=["period", "period_label", "cluster_label", "Count"])

    X_dense, _ = build_dense_matrix_from_json(df_all, max_features=int(max_features))

    labels = cosine_threshold_clusters_graph(
        X_dense,
        threshold=float(cosine_thr),
        min_cluster_size=int(min_cluster_size),
        method=str(graph_method),
        knn_k=int(knn_k),
    )

    df_all = df_all.copy()
    df_all["cluster_label"] = pd.Series(labels).apply(
        lambda x: "noise" if int(x) < 0 else f"C{int(x):02d}"
    )

    df_temp = compute_cluster_temporal_counts(
        df_all,
        cluster_col="cluster_label",
        date_col="tgl_submit",
        freq=freq,
        exclude_noise=bool(exclude_noise),
    )
    return df_temp


# ======================================================
# UI
# ======================================================
st.title("📌 Viewer & Analisis TF-IDF Unigram — incident_clean")
st.caption("Visualisasi t-SNE + overlay clustering cosine + ringkasan topik + pola temporal per cluster.")

runs = load_runs()
if runs.empty:
    st.warning("Belum ada run TF-IDF di tabel runs. Jalankan script offline dulu.")
    st.stop()

# Sidebar
st.sidebar.header("Parameter Viewer")

run_labels = [
    f"{r.run_time.strftime('%Y-%m-%d %H:%M:%S')} | {r.run_id}" if pd.notna(r.run_time) else f"(no time) | {r.run_id}"
    for r in runs.itertuples(index=False)
]
picked = st.sidebar.selectbox("Pilih Run", run_labels, index=0)
run_id = picked.split("|")[-1].strip()

row_run = runs[runs["run_id"] == run_id].iloc[0]
params = row_run.get("params_json", {}) or {}
drange = row_run.get("data_range", {}) or {}

# Filters
moduls = ["Semua"] + load_moduls(run_id)
sites = ["Semua"] + load_sites(run_id)

modul = st.sidebar.selectbox("Filter Modul", moduls, index=0)
site = st.sidebar.selectbox("Filter Site", sites, index=0)

# Date input (lebih aman dari typo)
sd = drange.get("start_date")
ed = drange.get("end_date")
start_default = pd.to_datetime(sd, errors="coerce").date() if sd else None
end_default = pd.to_datetime(ed, errors="coerce").date() if ed else None

start = st.sidebar.date_input("Start date", value=start_default)
end = st.sidebar.date_input("End date", value=end_default)

# Viewer limit
limit_rows = st.sidebar.slider("Limit tiket untuk viewer", 200, 20000, 3000, step=100)

# t-SNE controls (SAMPLE)
st.sidebar.divider()
st.sidebar.header("Grafik: t-SNE (Sample)")
enable_tsne = st.sidebar.checkbox("Aktifkan t-SNE", value=True)
tsne_sample = st.sidebar.slider("Jumlah tiket (sampling)", 200, 3000, 700, step=100)
tsne_max_features = st.sidebar.slider("Batas feature global", 300, 5000, 1400, step=100)
tsne_perplexity = st.sidebar.slider("Perplexity", 5, 50, 18, step=1)
tsne_n_iter = st.sidebar.slider("Iterasi", 500, 3000, 1000, step=100)
use_scaler_for_tsne = st.sidebar.checkbox("StandardScaler untuk t-SNE", value=True)

# Overlay controls (sample)
st.sidebar.divider()
st.sidebar.header("Overlay: Cluster Cosine (Sample)")
enable_overlay = st.sidebar.checkbox("Overlay cluster cosine", value=True)
cosine_threshold = st.sidebar.slider("Cosine threshold", 0.50, 0.95, 0.75, step=0.01)
min_cluster_size = st.sidebar.slider("Min ukuran cluster", 2, 30, 3, step=1)

# Topic + temporal controls
st.sidebar.divider()
st.sidebar.header("Ringkasan Cluster & Temporal")
show_noise_in_table = st.sidebar.checkbox("Tampilkan noise (tabel)", value=False)
show_noise_in_chart = st.sidebar.checkbox("Tampilkan noise (grafik)", value=False)
topk_terms = st.sidebar.slider("Top terms per cluster", 5, 20, 8, 1)
temporal_grain = st.sidebar.selectbox("Granularity waktu", ["Bulanan", "Mingguan", "Harian"], index=0)
freq = {"Bulanan": "M", "Mingguan": "W", "Harian": "D"}[temporal_grain]

st.sidebar.caption("⚠️ Temporal FULL dihitung dari DB (scalable graph). Batas aman berikut:")
max_n_full = st.sidebar.slider("Batas tiket (FULL temporal)", 2000, 200000, 30000, step=5000)

graph_method = st.sidebar.selectbox("Metode graph (FULL temporal)", ["radius", "knn"], index=0)
knn_k = st.sidebar.slider("k (untuk kNN)", 10, 300, 30, step=10)

# Metadata
with st.expander("ℹ️ Metadata Run", expanded=True):
    c1, c2, c3 = st.columns(3)
    c1.metric("Run Time", str(row_run["run_time"]))
    c2.metric("Approach", str(row_run["approach"]))
    c3.metric("Run ID", run_id)
    st.write("**Params (TF-IDF):**", params.get("tfidf", {}))
    st.write("**Data Range:**", drange)

# Load viewer data (berdasarkan filter)
dfv = load_vectors(run_id, modul, site, start or None, end or None, int(limit_rows))

st.subheader("📄 Daftar Tiket (sample viewer)")
st.caption("Kolom `tfidf_json` adalah JSONB `{feature: weight}` (sparse).")

if dfv.empty:
    st.warning("Tidak ada tiket pada filter ini.")
    st.stop()

show_cols = ["incident_number", "tgl_submit", "site", "assignee", "modul", "sub_modul"]
st.dataframe(dfv[show_cols], use_container_width=True, height=280)

# ======================================================
# Grafik 1: Distribusi jumlah feature aktif
# ======================================================
st.subheader("📊 Distribusi Jumlah Feature Aktif per Tiket")
df_feat = dfv.copy()
df_feat["n_features_active"] = df_feat["tfidf_json"].apply(lambda x: len(x) if isinstance(x, dict) else 0)

hist = (
    alt.Chart(df_feat)
    .mark_bar()
    .encode(
        x=alt.X("n_features_active:Q", bin=alt.Bin(maxbins=40), title="Jumlah feature non-zero"),
        y=alt.Y("count()", title="Jumlah tiket"),
        tooltip=[alt.Tooltip("count()", title="Jumlah tiket")],
    )
    .properties(height=300)
)
st.altair_chart(hist, use_container_width=True)

# ======================================================
# Grafik 2: Top feature doc frequency dari viewer
# ======================================================
st.subheader("📌 Top Feature (Document Frequency) — dari sample viewer")
st.caption("Doc frequency = berapa tiket dalam sample viewer yang mengandung feature tersebut.")

freq_map: dict[str, int] = {}
for v in dfv["tfidf_json"]:
    if isinstance(v, dict):
        for k in v.keys():
            kk = str(k)
            freq_map[kk] = freq_map.get(kk, 0) + 1

df_freq = (
    pd.DataFrame([{"feature": k, "doc_freq": v} for k, v in freq_map.items()])
    .sort_values("doc_freq", ascending=False)
    .head(30)
)

if df_freq.empty:
    st.info("Tidak ada feature yang bisa dihitung dari sample ini.")
else:
    bar = (
        alt.Chart(df_freq)
        .mark_bar()
        .encode(
            x=alt.X("doc_freq:Q", title="Doc frequency"),
            y=alt.Y("feature:N", sort="-x", title="Feature"),
            tooltip=["feature", "doc_freq"],
        )
        .properties(height=420)
    )
    st.altair_chart(bar, use_container_width=True)

# ======================================================
# Detail tiket + vektor TF-IDF
# ======================================================
st.subheader("🔎 Detail Tiket & Vektor TF-IDF (JSONB)")
inc = st.selectbox("Pilih incident_number", dfv["incident_number"].tolist())
row = dfv[dfv["incident_number"] == inc].iloc[0]

st.markdown(
    f"**Incident:** `{row['incident_number']}`  \n"
    f"**Tanggal:** {row['tgl_submit']}  \n"
    f"**Modul:** {row['modul']} | **Sub:** {row['sub_modul']}  \n"
    f"**Site:** {row['site']} | **Assignee:** {row['assignee']}"
)
st.text_area("Text Sintaksis", value=str(row["text_sintaksis"]), height=140)

vec = row["tfidf_json"] if isinstance(row["tfidf_json"], dict) else {}
df_vec = (
    pd.DataFrame([{"feature": str(k), "weight": float(v)} for k, v in vec.items()])
    .sort_values("weight", ascending=False)
)
if df_vec.empty:
    st.info("Vektor TF-IDF kosong untuk tiket ini.")
else:
    st.dataframe(df_vec.head(80), use_container_width=True, height=260)

# ======================================================
# t-SNE + Overlay (SAMPLE)
# ======================================================
st.divider()
st.subheader("🧭 t-SNE (2D) + Overlay Cluster Cosine (Sample)")

if not enable_tsne:
    st.info("t-SNE dimatikan. Aktifkan dari sidebar untuk melihat visualisasi.")
    st.stop()

if tsne_sample > 1500:
    st.warning(
        "Sampling > 1500 bisa berat untuk t-SNE & cosine full-matrix. "
        "Kalau terasa lama, turunkan 'Jumlah tiket (sampling)'."
    )

df_tsne_in = dfv.sample(n=min(int(tsne_sample), len(dfv)), random_state=42).copy()

cA, cB, cC, cD = st.columns([1, 1, 1, 1])
with cA:
    run_btn = st.button("▶️ Jalankan t-SNE (+ overlay)", use_container_width=True)
with cB:
    st.write(f"Sampling: **{len(df_tsne_in):,}** tiket")
with cC:
    st.write(f"Overlay: **{'ON' if enable_overlay else 'OFF'}**")
with cD:
    st.write(f"Temporal FULL: **{temporal_grain}**")

st.caption(
    "Overlay cluster cosine = pewarnaan titik berdasarkan cluster hasil cosine similarity threshold "
    "di ruang TF-IDF (bukan cluster dari t-SNE)."
)

if not run_btn:
    st.info("Klik tombol **Jalankan t-SNE (+ overlay)** untuk menghitung dan menampilkan plot.")
    st.stop()

with st.spinner("Menghitung t-SNE dan (opsional) overlay cluster cosine (sample)..."):
    df_tsne_out = compute_tsne_and_overlay(
        df_tsne_in[
            [
                "incident_number",
                "tgl_submit",
                "site",
                "assignee",
                "modul",
                "sub_modul",
                "text_sintaksis",
                "tfidf_json",
            ]
        ].copy(),
        max_features=int(tsne_max_features),
        perplexity=int(tsne_perplexity),
        n_iter=int(tsne_n_iter),
        do_overlay=bool(enable_overlay),
        cosine_thr=float(cosine_threshold),
        min_cluster_size=int(min_cluster_size),
        use_scaler_for_tsne=bool(use_scaler_for_tsne),
    )

# Plot t-SNE
n_unique_modul = df_tsne_out["modul"].nunique(dropna=True)
color_fallback = "modul" if n_unique_modul <= 15 else "site"

if enable_overlay:
    color_enc = alt.Color("cluster_label:N", legend=alt.Legend(title="Cluster (Cosine)"))
    title = f"t-SNE + Overlay Cluster Cosine (thr={cosine_threshold:.2f}, min_size={min_cluster_size})"
else:
    color_enc = alt.Color(f"{color_fallback}:N", legend=alt.Legend(title=color_fallback))
    title = f"t-SNE (warna = {color_fallback})"

chart_tsne = (
    alt.Chart(df_tsne_out)
    .mark_circle(size=70, opacity=0.78)
    .encode(
        x=alt.X("tsne_x:Q", axis=None),
        y=alt.Y("tsne_y:Q", axis=None),
        color=color_enc,
        tooltip=[
            "incident_number",
            "modul",
            "sub_modul",
            "site",
            "cluster_label",
            alt.Tooltip("tgl_submit:T", title="Tanggal"),
        ],
    )
    .properties(height=520, title=title)
)
st.altair_chart(chart_tsne, use_container_width=True)

# Ringkasan ukuran cluster (sample)
if enable_overlay:
    st.subheader("📌 Ringkasan ukuran cluster (Cosine) — sample")
    df_c = (
        df_tsne_out.groupby("cluster_label")["incident_number"]
        .count()
        .reset_index(name="n_tickets")
        .sort_values("n_tickets", ascending=False)
    )
    st.dataframe(df_c, use_container_width=True, height=260)
else:
    st.info("Overlay OFF — ringkasan cluster tidak ditampilkan.")

# ======================================================
# Cluster summary (topic) + temporal bubble chart (FULL)
# ======================================================
st.divider()
st.subheader("📍 Ringkasan Cluster (Topik) + Pola Temporal")
st.caption(
    "Bubble Chart dihitung dari FULL data via DB (sesuai filter tanggal) "
    "dengan graph-based clustering (radius/kNN)."
)

left, right = st.columns([1.05, 1.7])

with left:
    st.markdown("### 📋 Cluster Summary (dari sample t-SNE)")
    df_topic = compute_cluster_topics(
        df_tsne_out,
        cluster_col="cluster_label",
        tfidf_col="tfidf_json",
        topk=int(topk_terms),
        exclude_noise=(not show_noise_in_table),
    )

    if df_topic.empty:
        st.info("Belum ada cluster yang tampil (threshold/min_cluster_size mungkin terlalu ketat).")
    else:
        st.dataframe(df_topic, use_container_width=True, height=360)

with right:
    st.markdown("### 🫧 Pola Temporal per Cluster (Bubble Chart) — FULL via DB")

    if not enable_overlay:
        st.info("Overlay OFF — Bubble Chart butuh cluster_label. Aktifkan Overlay cluster cosine.")
    else:
        with st.spinner("Menghitung pola temporal dari FULL data (DB)..."):
            df_temp = load_temporal_counts_full_db(
                run_id=run_id,
                modul=modul,
                site=site,
                start=start if isinstance(start, date) else None,
                end=end if isinstance(end, date) else None,
                freq=freq,
                cosine_thr=float(cosine_threshold),
                min_cluster_size=int(min_cluster_size),
                max_features=int(tsne_max_features),
                max_n_full=int(max_n_full),
                exclude_noise=(not show_noise_in_chart),
                graph_method=str(graph_method),
                knn_k=int(knn_k),
            )

        if df_temp.empty:
            st.info("Tidak ada data temporal untuk ditampilkan (atau semua masuk noise).")
        else:
            cluster_order = cluster_ordering(df_temp["cluster_label"].unique().tolist())
            x_title = "Month" if freq == "M" else ("Week" if freq == "W" else "Date")

            bubble = (
                alt.Chart(df_temp)
                .mark_circle(opacity=0.75)
                .encode(
                    x=alt.X("period:T", title=x_title),
                    y=alt.Y("cluster_label:N", title="Clusters", sort=cluster_order),
                    size=alt.Size("Count:Q", title="Count"),
                    tooltip=[
                        alt.Tooltip("cluster_label:N", title="Cluster"),
                        alt.Tooltip("period_label:N", title="Periode"),
                        alt.Tooltip("Count:Q", title="Jumlah tiket"),
                    ],
                )
                .properties(height=360)
            )
            st.altair_chart(bubble, use_container_width=True)

            st.caption(
                f"FULL via DB: max_n_full={max_n_full:,}, method={graph_method}, k={knn_k}, "
                f"thr={cosine_threshold:.2f}, min_size={min_cluster_size}."
            )

# ======================================================
# Ringkasan waktu per cluster (sample)
# ======================================================
st.markdown("### ⏱️ Ringkasan Waktu per Cluster (first_seen / last_seen) — sample t-SNE")
df_time = df_tsne_out.copy()
df_time["tgl_submit"] = pd.to_datetime(df_time["tgl_submit"], errors="coerce")
df_time = df_time.dropna(subset=["tgl_submit"])

if not show_noise_in_table:
    df_time = df_time[df_time["cluster_label"].astype(str).str.lower() != "noise"]

if df_time.empty:
    st.info("Tidak ada data untuk ringkasan waktu.")
else:
    df_time_sum = (
        df_time.groupby("cluster_label")
        .agg(
            n_tickets=("incident_number", "count"),
            first_seen=("tgl_submit", "min"),
            last_seen=("tgl_submit", "max"),
        )
        .reset_index()
    )
    df_time_sum["duration_days"] = (df_time_sum["last_seen"] - df_time_sum["first_seen"]).dt.days.astype(int)
    df_time_sum = df_time_sum.sort_values("n_tickets", ascending=False)
    st.dataframe(df_time_sum, use_container_width=True, height=280)

# ======================================================
# Bonus: contoh tiket dari cluster terbesar (sample)
# ======================================================
if enable_overlay:
    non_noise = df_tsne_out[df_tsne_out["cluster_label"].astype(str).str.lower() != "noise"]
    if not non_noise.empty:
        top_cluster = (
            non_noise.groupby("cluster_label")["incident_number"]
            .count()
            .sort_values(ascending=False)
            .index[0]
        )

        st.markdown(f"### 🧩 Contoh tiket dari cluster terbesar (sample): {top_cluster}")
        ex = non_noise[non_noise["cluster_label"] == top_cluster].sort_values("tgl_submit", ascending=False).head(25)
        st.dataframe(
            ex[["incident_number", "tgl_submit", "modul", "sub_modul", "site", "assignee", "cluster_label"]],
            use_container_width=True,
            height=320,
        )
