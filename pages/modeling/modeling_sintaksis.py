# pages/modeling_sintaksis.py
# Modeling Sintaksis (TF-IDF) — Viewer + Similarity + Clustering (tanpa tfidf_vec_json)
# Sumber:
#   - lasis_djp.incident_tfidf_runs
#   - lasis_djp.incident_tfidf_vectors (pakai tfidf_json term->value)
#
# Catatan:
# - tfidf_vec_json memang kosong -> dibangun on-the-fly dari tfidf_json + feature_names_json
# - clustering scalable: kNN (NearestNeighbors) + threshold edge -> connected components
# - FIX: persist hasil clustering di session_state agar t-SNE tidak error saat rerun

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sqlalchemy import create_engine, text

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD


# =========================
# Guard login
# =========================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login terlebih dahulu.")
    st.stop()

# =========================
# Konstanta
# =========================
SCHEMA = "lasis_djp"
T_RUNS = "incident_tfidf_runs"
T_VECS = "incident_tfidf_vectors"

# =========================
# Session state defaults
# =========================
ss = st.session_state
ss.setdefault("tfidf_cluster_ready", False)
ss.setdefault("tfidf_cluster_run_id", None)
ss.setdefault("tfidf_cluster_dfc", None)          # df_valid + cluster_id
ss.setdefault("tfidf_cluster_summary", None)      # ringkasan cluster
ss.setdefault("tfidf_cluster_edges", None)        # jumlah edge
ss.setdefault("tfidf_cluster_meta", {})           # thr/k/min_size/n_docs


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


# =========================
# Helpers JSONB & array
# =========================
def ensure_list(obj: Any) -> Optional[List[Any]]:
    if obj is None:
        return None
    if isinstance(obj, list):
        return obj
    if isinstance(obj, str):
        try:
            out = json.loads(obj)
            return out if isinstance(out, list) else None
        except Exception:
            return None
    return None


def ensure_dict(obj: Any) -> Optional[Dict[str, Any]]:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        try:
            out = json.loads(obj)
            return out if isinstance(out, dict) else None
        except Exception:
            return None
    return None


def to_1d(a: Any) -> np.ndarray:
    """
    Convert matrix/array/sparse result to 1D ndarray.
    Menghindari .A1 agar kompatibel lintas versi SciPy/Numpy.
    """
    if hasattr(a, "toarray"):
        a = a.toarray()
    return np.asarray(a).ravel()


def build_term_index(feature_names_json: Any) -> Tuple[Optional[Dict[str, int]], int, Optional[List[str]]]:
    feats = ensure_list(feature_names_json)
    if not feats:
        return None, 0, None
    feats2 = [str(x) for x in feats]
    term_to_idx = {t: i for i, t in enumerate(feats2)}
    return term_to_idx, len(feats2), feats2


def build_tfidf_csr(
    df_docs: pd.DataFrame,
    term_to_idx: Dict[str, int],
    n_features: int,
    col_tfidf_json: str = "tfidf_json",
) -> Tuple[Optional[csr_matrix], Optional[pd.DataFrame]]:
    """
    Buat CSR matrix dari kolom tfidf_json (dict term->value).
    Mengembalikan:
      X: csr_matrix shape=(n_valid_docs, n_features)
      df_valid: dataframe subset dokumen yang valid (baris yang punya minimal 1 term yang match)
    """
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    keep_rows: List[int] = []

    data_list = df_docs[col_tfidf_json].tolist()

    for src_row_idx, raw in enumerate(data_list):
        d = ensure_dict(raw)
        if not isinstance(d, dict) or len(d) == 0:
            continue

        nnz = 0
        row_id = len(keep_rows)

        for term, v in d.items():
            j = term_to_idx.get(str(term))
            if j is None:
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            if fv == 0.0 or (isinstance(fv, float) and math.isnan(fv)):
                continue
            rows.append(row_id)
            cols.append(j)
            vals.append(fv)
            nnz += 1

        if nnz > 0:
            keep_rows.append(src_row_idx)

    if len(keep_rows) == 0:
        return None, None

    X = csr_matrix((vals, (rows, cols)), shape=(len(keep_rows), n_features), dtype=np.float32)
    df_valid = df_docs.iloc[keep_rows].reset_index(drop=True)
    return X, df_valid


def top_terms_from_tfidf_dict(tfidf_doc: Any, topn: int = 20) -> pd.DataFrame:
    d = ensure_dict(tfidf_doc)
    if not isinstance(d, dict):
        return pd.DataFrame(columns=["term", "tfidf"])
    items: List[Tuple[str, float]] = []
    for k, v in d.items():
        try:
            items.append((str(k), float(v)))
        except Exception:
            continue
    items.sort(key=lambda x: x[1], reverse=True)
    return pd.DataFrame(items[:topn], columns=["term", "tfidf"])


def compute_pairwise_similarity_sample_sparse(X: csr_matrix, m: int = 600, seed: int = 7) -> np.ndarray:
    """
    Sampling cosine similarity untuk pasangan acak (sparse-safe).
    sim(i,j) = (xi·xj) / (||xi|| ||xj||)
    """
    n = X.shape[0]
    if n < 2:
        return np.array([], dtype=np.float32)

    pairs = n * (n - 1) // 2
    m = min(m, pairs)
    if m <= 0:
        return np.array([], dtype=np.float32)

    rng = np.random.default_rng(seed)
    i1 = rng.integers(0, n, size=m)
    i2 = rng.integers(0, n, size=m)

    # row norms (sparse-safe)
    row_norm = np.sqrt(to_1d(X.multiply(X).sum(axis=1))) + 1e-9

    # dot produk pasangan (sparse)
    num = to_1d((X[i1].multiply(X[i2])).sum(axis=1))
    den = row_norm[i1] * row_norm[i2]
    sim = (num / den).astype(np.float32)
    sim = np.clip(sim, -1.0, 1.0)
    return sim


def connected_components_from_edges(n: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    """
    Union-Find untuk connected components.
    """
    parent = np.arange(n, dtype=np.int32)
    rank = np.zeros(n, dtype=np.int8)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for a, b in edges:
        union(a, b)

    roots = np.array([find(i) for i in range(n)], dtype=np.int32)

    # compress to 0..k-1
    uniq: Dict[int, int] = {}
    labels = np.empty(n, dtype=np.int32)
    cid = 0
    for i, r in enumerate(roots):
        rr = int(r)
        if rr not in uniq:
            uniq[rr] = cid
            cid += 1
        labels[i] = uniq[rr]
    return labels


# =========================
# Data loaders
# =========================
@st.cache_data(show_spinner=False)
def load_runs(limit: int = 200) -> pd.DataFrame:
    q = text(f"""
        SELECT run_id, run_time, approach, params_json, data_range, notes,
               idf_json, feature_names_json
        FROM {SCHEMA}.{T_RUNS}
        ORDER BY run_time DESC
        LIMIT :lim
    """)
    return pd.read_sql(q, engine, params={"lim": limit})


@st.cache_data(show_spinner=False)
def load_vectors(run_id: str) -> pd.DataFrame:
    q = text(f"""
        SELECT run_id, incident_number, tgl_submit, site, assignee, modul, sub_modul,
               tokens_sintaksis_json, text_sintaksis,
               tfidf_json
        FROM {SCHEMA}.{T_VECS}
        WHERE run_id = :run_id
    """)
    return pd.read_sql(q, engine, params={"run_id": run_id})


# =========================
# UI Header
# =========================
st.title("Modeling Sintaksis (TF-IDF)")
st.caption("Viewer + Similarity + Clustering berbasis `tfidf_json` (tanpa `tfidf_vec_json`).")

runs = load_runs()
if runs.empty:
    st.warning("Belum ada data di `incident_tfidf_runs`.")
    st.stop()


# =========================
# Sidebar controls
# =========================
with st.sidebar:
    st.header("1) Pilih Run TF-IDF")
    run_id = st.selectbox("run_id", runs["run_id"].tolist())
    run_row = runs.loc[runs["run_id"] == run_id].iloc[0]

    st.caption(f"Run time: {run_row['run_time']}")
    st.write(f"Approach: `{run_row['approach']}`")
    if run_row.get("notes"):
        st.info(run_row["notes"])

    st.divider()
    st.header("2) Filter Data (opsional)")
    sample_n = st.slider("Max sampel dokumen (untuk similarity/cluster)", 200, 8000, 1500, 100)
    mod_filter_on = st.checkbox("Filter modul", value=False)
    date_filter_on = st.checkbox("Filter tanggal", value=False)

    st.divider()
    st.header("3) Parameter Klasterisasi")
    thr = st.slider("Cosine similarity threshold", 0.70, 0.95, 0.80, 0.01)
    k_neighbors = st.slider("k Nearest Neighbors (graph)", 5, 60, 20, 1)
    min_cluster_size = st.slider("Min cluster size (opsional)", 1, 50, 2, 1)

    run_cluster = st.button("Jalankan Klasterisasi", type="primary")

    st.divider()
    st.header("4) Visualisasi t-SNE")
    show_tsne = st.checkbox("Tampilkan t-SNE (sample)", value=False)
    max_tsne = st.slider("Maks titik t-SNE", 200, 1500, 600, 100)
    # perplexity & svd_dim akan disesuaikan setelah tahu n_tsne


# =========================
# Load vectors
# =========================
df = load_vectors(run_id)
if df.empty:
    st.warning("Tidak ada data di `incident_tfidf_vectors` untuk run_id ini.")
    st.stop()

term_to_idx, n_features, feature_names = build_term_index(run_row["feature_names_json"])
if term_to_idx is None or n_features <= 0:
    st.error("feature_names_json kosong/tidak valid. Tidak bisa membangun vektor TF-IDF.")
    st.stop()

# Apply optional filters
df_work = df.copy()

if mod_filter_on:
    mods = sorted([m for m in df_work["modul"].dropna().unique().tolist()])
    sel_mods = st.sidebar.multiselect("Pilih modul", mods, default=mods[: min(10, len(mods))])
    if sel_mods:
        df_work = df_work[df_work["modul"].isin(sel_mods)]

if date_filter_on:
    dmin = pd.to_datetime(df_work["tgl_submit"], errors="coerce").min()
    dmax = pd.to_datetime(df_work["tgl_submit"], errors="coerce").max()
    if pd.isna(dmin) or pd.isna(dmax):
        st.sidebar.warning("tgl_submit kosong; filter tanggal tidak bisa dipakai.")
    else:
        dr = st.sidebar.date_input("Rentang tanggal", value=(dmin.date(), dmax.date()))
        # streamlit bisa mengembalikan date atau tuple(date,date) tergantung versi/aksi user
        if isinstance(dr, tuple) and len(dr) == 2:
            start, end = dr
            df_work = df_work[
                (pd.to_datetime(df_work["tgl_submit"], errors="coerce").dt.date >= start)
                & (pd.to_datetime(df_work["tgl_submit"], errors="coerce").dt.date <= end)
            ]

df_work = df_work.dropna(subset=["tfidf_json"])
if df_work.empty:
    st.warning("Setelah filter, tidak ada dokumen dengan `tfidf_json`.")
    st.stop()

if len(df_work) > sample_n:
    df_sample = df_work.sample(sample_n, random_state=7).reset_index(drop=True)
else:
    df_sample = df_work.reset_index(drop=True)

X, df_valid = build_tfidf_csr(df_sample, term_to_idx, n_features, col_tfidf_json="tfidf_json")
if X is None or df_valid is None or X.shape[0] < 2:
    st.warning("Vektor TF-IDF valid < 2. Tidak bisa similarity/clustering.")
    st.stop()


# Jika user ganti run_id, invalidate hasil cluster lama
if ss.get("tfidf_cluster_run_id") != run_id:
    ss["tfidf_cluster_ready"] = False
    ss["tfidf_cluster_run_id"] = run_id
    ss["tfidf_cluster_dfc"] = None
    ss["tfidf_cluster_summary"] = None
    ss["tfidf_cluster_edges"] = None
    ss["tfidf_cluster_meta"] = {}


# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["4.4 Ringkasan Run", "Fitur & IDF", "Eksplor Tiket", "Similarity", "Klasterisasi + t-SNE"]
)

# -------------------------
# Tab 1: Ringkasan Run
# -------------------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah dokumen (filter)", f"{len(df_work):,}")
    c2.metric("Dokumen dianalisis (valid)", f"{X.shape[0]:,}")
    c3.metric("Jumlah fitur", f"{n_features:,}")
    c4.metric("Modul unik", f"{df_work['modul'].nunique(dropna=True):,}")

    st.subheader("Metadata run")
    st.json(
        {
            "run_id": run_id,
            "run_time": str(run_row["run_time"]),
            "approach": run_row["approach"],
            "params_json": run_row["params_json"],
            "data_range": run_row["data_range"],
            "notes": run_row["notes"],
        }
    )

    st.subheader("Cek kelengkapan data")
    st.write(
        {
            "% tfidf_json NULL (setelah dropna tfidf_json)": float(df_work["tfidf_json"].isna().mean()) * 100,
            "% text_sintaksis NULL": float(df_work["text_sintaksis"].isna().mean()) * 100,
            "% tokens_sintaksis_json NULL": float(df_work["tokens_sintaksis_json"].isna().mean()) * 100,
        }
    )

    st.subheader("Preview data")
    st.dataframe(
        df_work[["incident_number", "tgl_submit", "modul", "sub_modul", "site", "assignee"]].head(20),
        use_container_width=True,
        height=360,
    )


# -------------------------
# Tab 2: Fitur & IDF
# -------------------------
with tab2:
    st.subheader("Analisis IDF (global)")
    idf_list = ensure_list(run_row["idf_json"])
    if not isinstance(feature_names, list) or not isinstance(idf_list, list) or len(idf_list) != len(feature_names):
        st.warning("idf_json / feature_names_json tidak valid atau panjangnya tidak sama.")
    else:
        idf = np.array([float(x) for x in idf_list], dtype=np.float32)
        feat = np.array(feature_names, dtype=object)

        tmp = pd.DataFrame({"term": feat, "idf": idf})
        colA, colB = st.columns(2)

        with colA:
            st.write("Term paling spesifik (IDF tinggi):")
            st.dataframe(tmp.sort_values("idf", ascending=False).head(30), use_container_width=True)

        with colB:
            st.write("Term paling umum (IDF rendah):")
            st.dataframe(tmp.sort_values("idf", ascending=True).head(30), use_container_width=True)

        st.divider()
        term_q = st.text_input("Cari term (mengandung)")
        if term_q:
            hit = (
                tmp[tmp["term"].str.contains(term_q, case=False, na=False)]
                .sort_values("idf", ascending=False)
                .head(100)
            )
            st.dataframe(hit, use_container_width=True, height=420)


# -------------------------
# Tab 3: Eksplor Tiket
# -------------------------
with tab3:
    st.subheader("Eksplor 1 Tiket")
    pick = st.selectbox("Pilih incident_number", df_valid["incident_number"].astype(str).tolist())
    d = df_valid.loc[df_valid["incident_number"].astype(str) == str(pick)].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Modul", str(d.get("modul", "")))
    c2.metric("Sub Modul", str(d.get("sub_modul", "")))
    c3.metric("Site", str(d.get("site", "")))
    c4.metric("Assignee", str(d.get("assignee", "")))

    st.write("**Tanggal submit**:", str(d.get("tgl_submit")))
    st.write("**text_sintaksis**")
    st.code((d.get("text_sintaksis") or "")[:4000])

    st.write("**Top TF-IDF terms (dokumen ini)**")
    st.dataframe(top_terms_from_tfidf_dict(d.get("tfidf_json"), topn=25), use_container_width=True)

    st.divider()
    st.subheader("Cari tiket serupa (cosine similarity)")
    topk = st.slider("Top-K similar", 3, 30, 10, 1)

    d_dict = ensure_dict(d.get("tfidf_json")) or {}
    rows, cols, vals = [], [], []
    for term, v in d_dict.items():
        j = term_to_idx.get(str(term))
        if j is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if fv == 0.0:
            continue
        rows.append(0)
        cols.append(j)
        vals.append(fv)

    if len(vals) == 0:
        st.warning("Dokumen ini tidak punya term TF-IDF valid untuk similarity.")
    else:
        qvec = csr_matrix((vals, (rows, cols)), shape=(1, n_features), dtype=np.float32)

        row_norm = np.sqrt(to_1d(X.multiply(X).sum(axis=1))) + 1e-9
        q_norm = float(math.sqrt(qvec.multiply(qvec).sum())) + 1e-9

        # dot: (X @ qvec.T) -> (n, 1)
        dots = to_1d(X @ qvec.T)
        sim = (dots / (row_norm * q_norm)).astype(np.float32)

        idx = np.argsort(-sim)[: topk + 1]
        out = df_valid.iloc[idx][["incident_number", "tgl_submit", "modul", "text_sintaksis"]].copy()
        out["similarity"] = sim[idx]
        out = out.sort_values("similarity", ascending=False)

        st.dataframe(out, use_container_width=True, height=420)


# -------------------------
# Tab 4: Similarity
# -------------------------
with tab4:
    st.subheader("Distribusi Cosine Similarity (sampling pasangan)")
    m_pairs = st.slider("Jumlah pasangan acak", 200, 3000, 800, 100)
    sim = compute_pairwise_similarity_sample_sparse(X, m=m_pairs, seed=7)

    if sim.size == 0:
        st.warning("Tidak cukup dokumen untuk sampling similarity.")
    else:
        sim_df = pd.DataFrame({"similarity": sim})
        hist = alt.Chart(sim_df).mark_bar().encode(
            x=alt.X("similarity:Q", bin=alt.Bin(maxbins=30)),
            y="count()"
        )
        st.altair_chart(hist, use_container_width=True)
        st.write(sim_df["similarity"].describe(percentiles=[0.5, 0.9, 0.95]).to_frame().T)


# -------------------------
# Tab 5: Klasterisasi + t-SNE
# -------------------------
with tab5:
    st.subheader("Klasterisasi TF-IDF (kNN Graph + Threshold)")
    st.caption("Bangun graph dari k tetangga terdekat (cosine), edge jika similarity ≥ threshold, lalu connected components.")

    # Jalankan clustering jika tombol diklik
    if run_cluster:
        if X.shape[0] < 2:
            st.warning("Dokumen valid < 2. Tidak bisa klasterisasi.")
        else:
            k = min(k_neighbors, max(2, X.shape[0] - 1))
            nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
            nn.fit(X)

            dist, ind = nn.kneighbors(X, return_distance=True)

            edges_set = set()
            for i in range(X.shape[0]):
                for jpos in range(1, ind.shape[1]):  # skip dirinya sendiri
                    j = int(ind[i, jpos])
                    sim_ij = float(1.0 - dist[i, jpos])
                    if sim_ij >= thr:
                        a, b = (i, j) if i < j else (j, i)
                        edges_set.add((a, b))

            edges = list(edges_set)

            if len(edges) == 0:
                st.warning("Tidak ada edge yang memenuhi threshold. Turunkan threshold atau naikkan kNN.")
            else:
                labels = connected_components_from_edges(X.shape[0], edges)
                dfc = df_valid.copy()
                dfc["cluster_id"] = labels

                counts = dfc["cluster_id"].value_counts()
                keep_clusters = counts[counts >= min_cluster_size].index.tolist()
                dfc["cluster_id"] = dfc["cluster_id"].where(dfc["cluster_id"].isin(keep_clusters), other=-1)

                summary = (
                    dfc[dfc["cluster_id"] != -1]
                    .groupby("cluster_id")
                    .agg(
                        n_members=("incident_number", "count"),
                        modul_top=("modul", lambda x: x.value_counts().index[0] if x.notna().any() else None),
                        first_date=("tgl_submit", "min"),
                        last_date=("tgl_submit", "max"),
                    )
                    .reset_index()
                    .sort_values("n_members", ascending=False)
                )

                # Persist ke session_state (FIX t-SNE)
                ss["tfidf_cluster_ready"] = True
                ss["tfidf_cluster_run_id"] = run_id
                ss["tfidf_cluster_dfc"] = dfc
                ss["tfidf_cluster_summary"] = summary
                ss["tfidf_cluster_edges"] = int(len(edges))
                ss["tfidf_cluster_meta"] = {
                    "thr": float(thr),
                    "k_neighbors": int(k),
                    "min_cluster_size": int(min_cluster_size),
                    "n_docs": int(X.shape[0]),
                }

    # Tampilkan hasil clustering terakhir (kalau ada)
    if not ss.get("tfidf_cluster_ready", False) or ss.get("tfidf_cluster_dfc") is None:
        st.info("Klik **Jalankan Klasterisasi** di sidebar untuk membentuk cluster.")
    else:
        dfc = ss["tfidf_cluster_dfc"]
        summary = ss["tfidf_cluster_summary"]
        meta = ss.get("tfidf_cluster_meta", {})
        edges_n = ss.get("tfidf_cluster_edges", 0)

        colA, colB, colC, colD = st.columns(4)
        colA.metric("Threshold", f"{meta.get('thr', thr):.2f}")
        colB.metric("kNN", f"{meta.get('k_neighbors', k_neighbors):,}")
        colC.metric("Jumlah edge", f"{edges_n:,}")
        colD.metric("Dokumen valid", f"{meta.get('n_docs', X.shape[0]):,}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Jumlah cluster", f"{(summary.shape[0] if isinstance(summary, pd.DataFrame) else 0):,}")
        col2.metric("Anggota terklaster", f"{(dfc['cluster_id'] != -1).sum():,}")
        col3.metric("Noise / cluster kecil", f"{(dfc['cluster_id'] == -1).sum():,}")

        st.write("**Ringkasan cluster**")
        if summary is None or summary.empty:
            st.warning("Semua cluster terfilter oleh min_cluster_size. Turunkan min_cluster_size lalu jalankan ulang.")
        else:
            st.dataframe(summary, use_container_width=True, height=380)

            pick_c = st.selectbox("Pilih cluster untuk drill-down", summary["cluster_id"].tolist())
            members = (
                dfc[dfc["cluster_id"] == pick_c][["incident_number", "tgl_submit", "modul", "text_sintaksis", "tfidf_json"]]
                .sort_values("tgl_submit", ascending=False)
                .reset_index(drop=True)
            )

            st.write("**Anggota cluster**")
            st.dataframe(
                members[["incident_number", "tgl_submit", "modul", "text_sintaksis"]],
                use_container_width=True,
                height=420
            )

            st.subheader("Top terms klaster (agregasi TF-IDF)")
            agg: Dict[str, float] = {}
            for raw in members["tfidf_json"].tolist():
                dct = ensure_dict(raw) or {}
                for term, v in dct.items():
                    try:
                        agg[str(term)] = agg.get(str(term), 0.0) + float(v)
                    except Exception:
                        continue
            top_terms = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:25]
            st.dataframe(pd.DataFrame(top_terms, columns=["term", "tfidf_sum"]), use_container_width=True)

            st.subheader("Timeline kemunculan klaster")
            if members["tgl_submit"].notna().any():
                tdf = members.copy()
                tdf["date"] = pd.to_datetime(tdf["tgl_submit"], errors="coerce").dt.date
                tcount = tdf.dropna(subset=["date"]).groupby("date").size().reset_index(name="count")

                chart = alt.Chart(tcount).mark_bar().encode(
                    x="date:T",
                    y="count:Q",
                    tooltip=["date:T", "count:Q"]
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("tgl_submit kosong; timeline tidak dapat ditampilkan.")

        # ---------- t-SNE ----------
        st.divider()
        st.subheader("Visualisasi t-SNE (opsional)")
        st.caption("Menggunakan sample dari dokumen valid. t-SNE berbasis SVD→2D; warna = cluster_id.")

        if show_tsne:
            n_tsne = min(int(max_tsne), int(X.shape[0]))
            if n_tsne < 50:
                st.warning("Jumlah sampel terlalu kecil untuk t-SNE (minimal ~50).")
            else:
                max_perp = max(5, min(50, (n_tsne - 1) // 3))
                perp = st.slider("Perplexity", 5, max_perp, min(30, max_perp), 1)
                svd_dim = st.slider("Reduksi awal (SVD dim)", 10, 100, 50, 5)

                # SVD dim aman: <= min(n_features-1, n_tsne-1)
                svd_dim_eff = min(int(svd_dim), int(X.shape[1] - 1), int(n_tsne - 1))
                if svd_dim_eff < 2:
                    st.warning("SVD dim terlalu kecil setelah penyesuaian. Naikkan sampel atau cek fitur.")
                else:
                    with st.spinner("Menjalankan t-SNE..."):
                        try:
                            # Optional: ambil sample random agar tidak bias urutan
                            rng = np.random.default_rng(7)
                            idx = rng.choice(X.shape[0], size=n_tsne, replace=False)
                            idx = np.sort(idx)  # agar dfc.iloc[idx] konsisten & mudah dibaca

                            Xs = X[idx]
                            dfc_local = dfc.iloc[idx].reset_index(drop=True)
                            labels_tsne = dfc_local["cluster_id"].astype(str).values

                            svd = TruncatedSVD(n_components=svd_dim_eff, random_state=7)
                            Xr = svd.fit_transform(Xs)

                            ts = TSNE(
                                n_components=2,
                                perplexity=float(perp),
                                init="random",
                                learning_rate="auto",
                                random_state=7
                            )
                            emb = ts.fit_transform(Xr)

                            p = pd.DataFrame({
                                "x": emb[:, 0],
                                "y": emb[:, 1],
                                "cluster_id": labels_tsne,
                            })

                            chart = alt.Chart(p).mark_circle(size=50).encode(
                                x="x:Q",
                                y="y:Q",
                                color="cluster_id:N",
                                tooltip=["cluster_id:N"]
                            ).interactive()

                            st.altair_chart(chart, use_container_width=True)

                        except Exception as e:
                            st.error("t-SNE gagal dijalankan.")
                            st.exception(e)
        else:
            st.info("Centang **Tampilkan t-SNE** di sidebar untuk menampilkan proyeksi 2D.")
