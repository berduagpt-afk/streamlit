# pages/analisis_sintaksis.py
# Analisis Sintaksis (TF-IDF + Cosine Similarity) langsung dari PostgreSQL (schema: lasis_djp)

import re
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer

st.title("🧮 Analisis Sintaksis (TF-IDF + Cosine Similarity)")
st.caption("Analisis kemiripan antar tiket berbasis kata dari data di database (schema: lasis_djp).")

# ======================================================
# 🔐 Koneksi Database PostgreSQL
# ======================================================
def get_connection():
    cfg = st.secrets["connections"]["postgres"]
    url = (
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )
    return create_engine(url)

@st.cache_data(show_spinner=False)
def load_data_from_db(table_name="incident_clean", schema="lasis_djp"):
    engine = get_connection()
    df = pd.read_sql_table(table_name, con=engine, schema=schema)
    engine.dispose()
    return df

def save_dataframe(df: pd.DataFrame, table_name: str, schema: str = "lasis_djp", if_exists: str = "replace"):
    engine = get_connection()
    df.to_sql(table_name, engine, schema=schema, if_exists=if_exists, index=False)
    engine.dispose()

# ======================================================
# ⚙️ Fungsi bantu
# ======================================================
def pick_date_column(df: pd.DataFrame) -> str | None:
    for c in ["tgl_submit", "timestamp"]:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
                return c
            except Exception:
                pass
    for c in df.select_dtypes(include="datetime").columns:
        return c
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
        token_pattern=r"(?u)\b\w+\b",
    )
    X = vec.fit_transform(texts)
    return vec, X

def upper_triangle_pairs_sparse(gram, threshold: float, limit: int | None = 5000):
    gram = gram.tocsr()
    n = gram.shape[0]
    pairs = []
    for i in range(n):
        start, end = gram.indptr[i], gram.indptr[i + 1]
        cols = gram.indices[start:end]
        vals = gram.data[start:end]
        for j, s in zip(cols, vals):
            if j <= i:
                continue
            if s >= threshold:
                pairs.append((i, j, float(s)))
    pairs.sort(key=lambda x: x[2], reverse=True)
    if limit:
        pairs = pairs[:limit]
    return pairs

def rule_based_label(df, pairs, date_col, days_window):
    labels = pd.Series(["Tidak Berulang"] * len(df), index=df.index)
    if date_col not in df.columns:
        return labels
    order = df[date_col].sort_values(kind="stable").index.tolist()
    pos = {idx: k for k, idx in enumerate(order)}
    for i, j, _s in pairs:
        idx_i, idx_j = df.index[i], df.index[j]
        earlier, later = (idx_j, idx_i) if pos[idx_j] < pos[idx_i] else (idx_i, idx_j)
        dt = abs((df.loc[later, date_col] - df.loc[earlier, date_col]).days)
        if dt <= days_window:
            labels.loc[later] = "Berulang"
    return labels

# ======================================================
# 🧭 Sidebar Parameter
# ======================================================
with st.sidebar:
    st.header("⚙️ Pengaturan Analisis Sintaksis")
    table_name = st.text_input("Nama tabel sumber data", "incident_clean")
    schema = "lasis_djp"
    ngram = st.selectbox("N-gram", ["1", "1–2", "1–3"], index=1)
    ngram_map = {"1": (1, 1), "1–2": (1, 2), "1–3": (1, 3)}
    min_df = st.number_input("min_df", 1, 100, 2)
    max_df = st.slider("max_df", 0.5, 1.0, 0.95, 0.01)
    threshold = st.slider("Threshold cosine", 0.0, 1.0, 0.8, 0.01)
    window_days = st.number_input("Jendela waktu berulang (hari)", 1, 180, 30)
    sample_size = st.number_input("Batas jumlah data", 0, 10000, 3000, step=100)
    run = st.button("🚀 Jalankan Analisis", use_container_width=True)

# ======================================================
# 📥 Ambil Data dari Database
# ======================================================
try:
    df = load_data_from_db(table_name=table_name, schema=schema)
    st.success(f"✅ Berhasil memuat {len(df):,} baris dari {schema}.{table_name}")
except Exception as e:
    st.error(f"Gagal memuat data dari database: {e}")
    st.stop()

if df.empty:
    st.warning("Dataset kosong.")
    st.stop()

# Kolom teks utama
text_col_candidates = [c for c in ["tokens_str", "Deskripsi_Bersih", "isi_permasalahan", "text_sintaksis"] if c in df.columns]
if not text_col_candidates:
    st.error("Tidak ada kolom teks yang sesuai. Harus ada salah satu dari tokens_str / Deskripsi_Bersih / isi_permasalahan / text_sintaksis.")
    st.stop()

text_col = text_col_candidates[0]
date_col = pick_date_column(df)
id_col = "Incident_Number" if "Incident_Number" in df.columns else None

st.info(f"📄 Kolom teks: `{text_col}` | Kolom tanggal: `{date_col}` | Kolom ID: `{id_col or '(tidak ada)'}`")

if sample_size and len(df) > sample_size:
    df = df.head(sample_size)
    st.warning(f"Dataset dibatasi ke {sample_size:,} baris untuk efisiensi.")

# ======================================================
# 🧮 TF-IDF Vectorization
# ======================================================
texts = df[text_col].fillna("").astype(str).tolist()
if not any(texts):
    st.error("Semua teks kosong pada kolom terpilih.")
    st.stop()

vec, X = tfidf_fit_transform(
    texts,
    ngram_range=ngram_map[ngram],
    min_df=min_df,
    max_df=max_df,
    use_idf=True,
    sublinear_tf=True,
    norm="l2",
)
st.success(f"Vectorizer terlatih: {X.shape[0]} dokumen × {X.shape[1]} fitur.")

# ======================================================
# 🔗 Kemiripan antar dokumen
# ======================================================
gram = (X * X.T).tocsr()
pairs = upper_triangle_pairs_sparse(gram, threshold=threshold, limit=5000)

if not pairs:
    st.warning("Tidak ditemukan pasangan tiket yang mirip di atas threshold.")
    st.stop()

pairs_df = pd.DataFrame(
    [
        {
            "i": int(i), "j": int(j), "similarity": round(float(s), 4),
            "id_i": df.iloc[i][id_col] if id_col else i,
            "id_j": df.iloc[j][id_col] if id_col else j,
            "tgl_i": df.iloc[i][date_col] if date_col else None,
            "tgl_j": df.iloc[j][date_col] if date_col else None,
        }
        for i, j, s in pairs
    ]
)

st.subheader("🔗 Pasangan Tiket Mirip")
st.dataframe(pairs_df, use_container_width=True, hide_index=True)

# ======================================================
# 🏷️ Label Temporal: Berulang / Tidak Berulang
# ======================================================
labels = rule_based_label(df, pairs, date_col=date_col, days_window=window_days)
df["Label_Sintaksis"] = labels
df["tgl_analisis"] = datetime.now()

st.subheader("📊 Distribusi Label")
st.bar_chart(df["Label_Sintaksis"].value_counts())

if date_col:
    st.subheader("📈 Tren Insiden Berulang per Bulan")
    trend = df.groupby([df[date_col].dt.to_period("M"), "Label_Sintaksis"]).size().unstack(fill_value=0)
    st.line_chart(trend)

# ======================================================
# 💾 Simpan ke Database
# ======================================================
try:
    with st.spinner("Menyimpan hasil analisis ke database..."):
        save_dataframe(df, table_name="incident_sintaksis", schema="lasis_djp", if_exists="replace")
    st.success("✅ Hasil analisis disimpan ke lasis_djp.incident_sintaksis.")
except Exception as e:
    st.error(f"Gagal menyimpan ke database: {e}")

# ======================================================
# 📤 Unduh hasil CSV
# ======================================================
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Unduh Hasil Analisis Sintaksis (CSV)",
    data=csv,
    file_name="hasil_analisis_sintaksis.csv",
    mime="text/csv",
)
