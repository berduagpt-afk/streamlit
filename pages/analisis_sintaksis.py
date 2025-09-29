# analisis_sintaksis.py
import math
import re
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.title("🧮 Analisis Sintaksis (TF-IDF + Cosine Similarity)")
st.caption(
    "Mencari kemiripan antar tiket berbasis kata (TF-IDF), lalu memberi label insiden berulang dengan aturan temporal."
)

# ========================= Helpers =========================
def pick_date_column(df: pd.DataFrame) -> str | None:
    # Prioritas: timestamp (dari preprocessing) -> Tanggal_Parsed -> kolom bertipe datetime -> coba parse 'Tanggal'
    if "timestamp" in df.columns:
        return "timestamp"
    if "Tanggal_Parsed" in df.columns:
        return "Tanggal_Parsed"
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            return c
    for c in df.columns:
        if re.search(r"tanggal|date|waktu|created|time", str(c), flags=re.I):
            try:
                _ = pd.to_datetime(df[c], errors="raise")
                df[c] = _
                return c
            except Exception:
                pass
    return None

@st.cache_data(show_spinner=False)
def tfidf_fit_transform(
    texts: list[str],
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    max_features=None,
    use_idf=True,
    sublinear_tf=True,
    norm="l2",
    lowercase=False,
    token_pattern=r"(?u)\b\w+\b",
):
    vec = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        use_idf=use_idf,
        sublinear_tf=sublinear_tf,
        norm=norm,
        lowercase=lowercase,
        token_pattern=token_pattern,
    )
    X = vec.fit_transform(texts)
    return vec, X

def upper_triangle_pairs_sparse(gram: "scipy.sparse.spmatrix", threshold: float, limit: int | None = 5000):
    """
    Ambil pasangan i<j dengan similarity >= threshold dari matriks Gram (X*X.T), format sparse.
    Mengembalikan list (i, j, sim) terurut desc.
    """
    gram = gram.tocsr()
    n = gram.shape[0]
    pairs = []
    for i in range(n):
        start_ptr, end_ptr = gram.indptr[i], gram.indptr[i + 1]
        cols = gram.indices[start_ptr:end_ptr]
        vals = gram.data[start_ptr:end_ptr]
        for j, sim in zip(cols, vals):
            if j <= i:
                continue
            if sim >= threshold:
                pairs.append((i, j, float(sim)))
    pairs.sort(key=lambda x: x[2], reverse=True)
    if limit is not None:
        pairs = pairs[:limit]
    return pairs

def rule_based_label(df: pd.DataFrame, pairs: list[tuple[int, int, float]], date_col: str, days_window: int) -> pd.Series:
    """
    Label 'Berulang' jika tiket i punya pasangan j lebih awal dengan sim>=thr dan selisih tanggal <= window.
    """
    labels = pd.Series(["Tidak Berulang"] * len(df), index=df.index)
    # Urut waktu agar j < i mewakili tiket lebih awal
    order = df[date_col].sort_values(kind="stable").index.tolist()
    pos = {idx: k for k, idx in enumerate(order)}  # posisi waktu
    for i, j, _s in pairs:
        idx_i, idx_j = df.index[i], df.index[j]
        earlier, later = (idx_j, idx_i) if pos[idx_j] < pos[idx_i] else (idx_i, idx_j)
        dt = abs((df.loc[later, date_col] - df.loc[earlier, date_col]).days)
        if dt <= days_window:
            labels.loc[later] = "Berulang"
    return labels

def top_contrib_terms_for(doc_row, q_vec, vec, topm=5):
    """
    Ambil term kontribusi teratas untuk satu dokumen terhadap query:
    lakukan elemen-wise product doc_row * q_vec, ambil fitur terbesar.
    """
    prod = doc_row.multiply(q_vec)  # 1 x n_features
    if prod.nnz == 0:
        return []
    idx_sorted = np.argsort(prod.data)[-topm:][::-1]
    feats = vec.get_feature_names_out()
    return [feats[prod.indices[k]] for k in idx_sorted]

# ========================= Ambil data =========================
df_clean = st.session_state.get("df_clean", None)  # hasil preprocessing

# ❌ Hapus/baris lama yang bikin error:
# df_raw = st.session_state.get("df_raw") or st.session_state.get("dataset")

# ✅ Ganti dengan ini:
df_raw = st.session_state.get("df_raw", None)
if df_raw is None:
    df_raw = st.session_state.get("dataset", None)

if df_clean is None and df_raw is None:
    st.warning("⚠️ Tidak ada dataset di sesi. Silakan upload & preprocessing terlebih dulu.")
    st.stop()

df_source = df_clean if df_clean is not None else df_raw
df = df_source.copy()


# kolom teks: utamakan tokens_str (TF-IDF friendly) -> Deskripsi_Bersih -> isi
text_col_candidates = [c for c in ["tokens_str", "Deskripsi_Bersih", "isi"] if c in df.columns]
if not text_col_candidates:
    st.error("Tidak menemukan kolom teks yang sesuai. Harus ada salah satu dari: tokens_str / Deskripsi_Bersih / isi.")
    st.stop()

# ========================= Sidebar =========================
with st.sidebar:
    st.header("⚙️ Pengaturan TF-IDF")
    text_col = st.selectbox("Kolom teks", text_col_candidates, index=0,
                            help="Disarankan: tokens_str (hasil tokenisasi).")
    ngram_choice = st.selectbox("N-gram", ["1", "1–2", "1–3"], index=1)
    ngram_map = {"1": (1, 1), "1–2": (1, 2), "1–3": (1, 3)}
    ngram = ngram_map[ngram_choice]

    min_df = st.number_input("min_df (dokumen minimal)", min_value=1, max_value=100, value=2, step=1)
    max_df = st.slider("max_df (proporsi dokumen maksimal)", min_value=0.5, max_value=1.0, value=0.95, step=0.01)
    max_features = st.number_input("max_features (0=tanpa batas)", min_value=0, max_value=100000, value=0, step=1000)
    use_idf = st.checkbox("Gunakan IDF", value=True)
    sublinear_tf = st.checkbox("Sublinear TF", value=True)
    norm = st.selectbox("Normalisasi", ["l2", "l1", None], index=0)
    lowercase = st.checkbox("Lowercase di TF-IDF", value=False, help="Biasanya tidak perlu karena sudah dibersihkan.")

    st.divider()
    st.header("🔎 Kemiripan")
    mode = st.radio("Mode", ["Semua pasangan >= threshold", "Top-k per tiket"], index=0)
    threshold = st.slider("Threshold cosine", 0.0, 1.0, 0.80, 0.01)
    topk = st.number_input("k (untuk Top-k per tiket)", min_value=1, max_value=50, value=5, step=1)
    sample_size = st.number_input(
        "Batas jumlah baris (0=semua)", min_value=0, max_value=20000, value=3000, step=100,
        help="Untuk performa, batasi jumlah baris saat eksplorasi."
    )

    st.divider()
    st.header("🕒 Pelabelan Temporal")
    date_col = pick_date_column(df)
    window_days = st.number_input("Jendela hari insiden berulang", min_value=1, max_value=180, value=30, step=1)

    run = st.button("🚀 Jalankan Analisis", use_container_width=True)

st.write(
    f"**Dataset aktif:** {len(df):,} baris. Kolom teks: `{text_col}`"
    + (f" | Kolom tanggal: `{date_col}`" if date_col else " | Kolom tanggal: (tidak tersedia)")
)

if not run:
    st.info("Atur opsi di sidebar, lalu klik **Jalankan Analisis**.")
    st.stop()

# ========================= Sampling =========================
if sample_size and sample_size > 0 and len(df) > sample_size:
    st.warning(f"Dataset {len(df):,} baris dibatasi ke {sample_size:,} baris untuk performa.")
    df = df.sort_index().head(sample_size)

# Pastikan ada teks valid
texts = df[text_col].fillna("").astype(str).tolist()
if not any(t.strip() for t in texts):
    st.error("Semua teks kosong pada kolom terpilih.")
    st.stop()

# ========================= TF-IDF =========================
vec, X = tfidf_fit_transform(
    texts,
    ngram_range=ngram,
    min_df=min_df,
    max_df=(None if max_df >= 0.9999 else max_df),
    max_features=(None if max_features == 0 else max_features),
    use_idf=use_idf,
    sublinear_tf=sublinear_tf,
    norm=norm,
    lowercase=lowercase,
    token_pattern=r"(?u)\b\w+\b",
)
st.success(f"Vectorizer terlatih: {X.shape[0]} dokumen × {X.shape[1]} fitur.")

# Simpan untuk pemakaian lanjutan (opsional)
st.session_state["syntax_vec"] = vec
st.session_state["syntax_X"] = X
st.session_state["syntax_df_ref"] = df
st.session_state["syntax_text_col"] = text_col
st.session_state["syntax_date_col"] = date_col

# ========================= Similarity (antar dokumen) =========================
# Gram = X * X.T (cosine sim bila norm != None)
gram = (X * X.T).tocsr()

pairs = []
if mode == "Semua pasangan >= threshold":
    pairs = upper_triangle_pairs_sparse(gram, threshold=threshold, limit=10000)
else:
    # Top-k per i (hindari self-match)
    n = gram.shape[0]
    tmp = []
    for i in range(n):
        row = gram.getrow(i)
        cols = row.indices
        vals = row.data
        cand = [(j, float(s)) for j, s in zip(cols, vals) if j != i]
        cand.sort(key=lambda x: x[1], reverse=True)
        for j, s in cand[:topk]:
            if i < j:
                tmp.append((i, j, s))
            elif j < i:
                tmp.append((j, i, s))
    # unique pairs & threshold
    pairs = list({(i, j): s for i, j, s in tmp if s >= threshold}.items())
    pairs = [(i, j, s) for (i, j), s in pairs]
    pairs.sort(key=lambda x: x[2], reverse=True)

# ========================= Tabel pasangan mirip =========================
if not pairs:
    st.warning("Tidak ada pasangan tiket yang memenuhi kriteria.")
else:
    id_col_candidates = [c for c in ["id_bugtrack", "id", "ID", "ticket_id"] if c in df.columns]
    id_col = id_col_candidates[0] if id_col_candidates else None
    tanggal_col = date_col

    data_rows = []
    for i, j, s in pairs[:5000]:  # batasi tampilan
        row = {"i": int(i), "j": int(j), "similarity": round(float(s), 4)}
        if id_col:
            row["id_i"] = df.iloc[i][id_col]
            row["id_j"] = df.iloc[j][id_col]
        if tanggal_col:
            row["tgl_i"] = df.iloc[i][tanggal_col]
            row["tgl_j"] = df.iloc[j][tanggal_col]
        row["text_i"] = df.iloc[i][text_col]
        row["text_j"] = df.iloc[j][text_col]
        data_rows.append(row)

    pairs_df = pd.DataFrame(data_rows)
    st.subheader("🔗 Pasangan Tiket Mirip")
    st.dataframe(pairs_df, use_container_width=True, hide_index=True)

    # ========================= Rule-based labeling =========================
    if date_col is None:
        st.info("Kolom tanggal tidak tersedia, pelabelan temporal dilewati.")
        labels = pd.Series(["Tidak Berulang"] * len(df), index=df.index)
    else:
        labels = rule_based_label(df, pairs, date_col=date_col, days_window=window_days)

    df_out = df.copy()
    df_out["Label_Sintaksis"] = labels.values

    st.subheader("📊 Ringkasan Label")
    st.write(df_out["Label_Sintaksis"].value_counts())

    if date_col is not None:
        st.subheader("📈 Tren Insiden Berulang (per bulan)")
        trend = (
            df_out.groupby([df_out[date_col].dt.to_period("M"), "Label_Sintaksis"])
            .size()
            .unstack(fill_value=0)
        )
        st.line_chart(trend)

    # simpan ke session & unduh
    st.session_state["syntax_pairs"] = pairs_df
    st.session_state["df_syntax"] = df_out

    csv = df_out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Hasil Analisis Sintaksis (CSV)",
        data=csv,
        file_name="hasil_analisis_sintaksis.csv",
        mime="text/csv",
    )

# ========================= 🔍 Pencarian Interaktif (TF-IDF) =========================
st.subheader("🔍 Pencarian Interaktif (TF-IDF)")
st.caption("Ketik deskripsi tiket/keluhan untuk menemukan tiket paling mirip berdasarkan model TF-IDF di atas.")

with st.form("search_form", clear_on_submit=False):
    query = st.text_area("Deskripsi tiket / keluhan", height=120, placeholder="Contoh: Tidak bisa login ke aplikasi setelah update…")
    k_res = st.number_input("Top-k hasil", min_value=1, max_value=50, value=10, step=1)
    show_text_col = st.selectbox("Tampilkan kolom teks", [c for c in ["isi", "Deskripsi_Bersih", "tokens_str"] if c in df.columns], index=0)
    do_search = st.form_submit_button("🔎 Cari", use_container_width=True)

if do_search:
    q = (query or "").strip()
    if not q:
        st.warning("Masukkan deskripsi untuk pencarian.")
    else:
        # vektorkan query dengan vectorizer yang sama
        q_vec = vec.transform([q])
        # cosine sim = X * q_vec.T  (asumsi TF-IDF normalized sesuai pengaturan)
        sims = (X @ q_vec.T).toarray().ravel()
        if np.allclose(sims.max(), 0.0):
            st.warning("Tidak ada token query yang cocok dengan vocabulary TF-IDF. Coba ubah kata kunci.")
        else:
            top_idx = np.argsort(-sims)[:k_res]
            rows = []
            for rank, idx in enumerate(top_idx, start=1):
                r = {
                    "rank": rank,
                    "index": int(idx),
                    "score": round(float(sims[idx]), 4),
                }
                # ID & tanggal (jika ada)
                for cand in ["id_bugtrack", "id", "ID", "ticket_id"]:
                    if cand in df.columns:
                        r["id"] = df.iloc[idx][cand]
                        break
                if date_col:
                    r["tanggal"] = df.iloc[idx][date_col]
                r["teks"] = df.iloc[idx][show_text_col]
                # term kontribusi
                try:
                    r["top_terms"] = ", ".join(top_contrib_terms_for(X.getrow(idx), q_vec, vec, topm=5))
                except Exception:
                    r["top_terms"] = ""
                rows.append(r)
            res_df = pd.DataFrame(rows)
            st.dataframe(res_df, use_container_width=True, hide_index=True)
