# pages/tfidf_extraction.py
# Feature Extraction: TF-IDF dari kolom text_sintaksis (incident_clean)

import io
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer

# ======================================================
# 🔐 Guard login
# ======================================================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login terlebih dahulu untuk mengakses halaman ini.")
    st.stop()

# Inisialisasi state TF-IDF
if "tfidf_X" not in st.session_state:
    st.session_state["tfidf_X"] = None
    st.session_state["tfidf_vocab"] = None
    st.session_state["tfidf_n_docs"] = None
    st.session_state["tfidf_n_terms"] = None

# ======================================================
# 🔌 Koneksi PostgreSQL
# ======================================================
def get_connection():
    cfg = st.secrets["connections"]["postgres"]
    url = (
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )
    return create_engine(url, pool_pre_ping=True)


@st.cache_data(show_spinner=False)
def load_incident_clean(schema="lasis_djp", table="incident_clean") -> pd.DataFrame:
    eng = get_connection()
    try:
        df = pd.read_sql_table(table, con=eng, schema=schema)
    finally:
        eng.dispose()
    return df


# ======================================================
# 🧭 Setup halaman
# ======================================================
st.title("🧮 Feature Extraction – TF-IDF (Pendekatan Sintaksis)")
st.caption(
    "Menghitung representasi vektor TF-IDF dari kolom **text_sintaksis** "
    "pada tabel `lasis_djp.incident_clean`."
)

# ======================================================
# 📦 Load data
# ======================================================
with st.spinner("📦 Memuat data dari incident_clean..."):
    try:
        df = load_incident_clean()
    except Exception as e:
        st.error(f"Gagal memuat data dari database: {e}")
        st.stop()

if df.empty:
    st.warning("Dataset incident_clean kosong. Jalankan dulu tahap preprocessing.")
    st.stop()

# Validasi kolom wajib
required_cols = ["incident_number", "text_sintaksis"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Kolom wajib tidak ditemukan: {', '.join(missing)}")
    st.stop()

# Pastikan teks bertipe string
df["text_sintaksis"] = df["text_sintaksis"].fillna("").astype(str)

# ======================================================
# 🔍 Filter data sebelum TF-IDF (modul & waktu)
# ======================================================
st.subheader("Filter Data Sebelum TF-IDF (Opsional)")

with st.expander("Tampilkan filter data (modul / rentang waktu)", expanded=False):

    # Filter berdasarkan modul (jika kolom modul tersedia)
    if "modul" in df.columns:
        all_modul = (
            df["modul"]
            .dropna()
            .astype(str)
            .sort_values()
            .unique()
            .tolist()
        )

        selected_modul = st.multiselect(
            "Pilih modul (kosongkan jika ingin semua modul)",
            options=all_modul,
        )

        if selected_modul:
            df = df[df["modul"].astype(str).isin(selected_modul)].copy()

    else:
        st.info("Kolom **modul** tidak ditemukan di incident_clean, filter modul dilewati.")

    # Filter berdasarkan rentang tanggal (jika kolom tgl_submit tersedia)
    if "tgl_submit" in df.columns:
        df["tgl_submit"] = pd.to_datetime(df["tgl_submit"], errors="coerce")

        valid_dates = df["tgl_submit"].dropna()
        if not valid_dates.empty:
            min_date = valid_dates.min().date()
            max_date = valid_dates.max().date()

            start_date, end_date = st.date_input(
                "Rentang tanggal berdasarkan kolom tgl_submit",
                value=(min_date, max_date),
            )

            if start_date > end_date:
                st.warning("Tanggal awal lebih besar dari tanggal akhir. Rentang tanggal tidak diterapkan.")
            else:
                mask = df["tgl_submit"].dt.date.between(start_date, end_date)
                df = df[mask].copy()
        else:
            st.info("Semua nilai tgl_submit tidak valid/NaT, filter tanggal dilewati.")
    else:
        st.info("Kolom **tgl_submit** tidak ditemukan di incident_clean, filter tanggal dilewati.")

# Cek lagi setelah filter
if df.empty:
    st.error("Tidak ada data yang tersisa setelah filter modul/waktu. Atur ulang filter Anda.")
    st.stop()

# ======================================================
# 🔧 Pengaturan TF-IDF (Sidebar)
# ======================================================
with st.sidebar:
    st.header("⚙️ Parameter TF-IDF")

    max_features = st.number_input(
        "Maksimal jumlah fitur (0 = semua)",
        min_value=0,
        max_value=50_000,
        value=5_000,
        step=500,
    )

    ngram_max = st.selectbox(
        "N-gram maksimum",
        options=[1, 2],
        index=0,
        format_func=lambda x: "Unigram saja" if x == 1 else "Unigram + Bigram",
    )

    min_df = st.number_input(
        "min_df (dokumen minimal suatu term muncul)",
        min_value=1,
        max_value=100,
        value=3,
        step=1,
    )

    max_df_ratio = st.slider(
        "max_df (proporsi dokumen maksimal term muncul)",
        min_value=0.1,
        max_value=1.0,
        value=0.9,
        step=0.05,
    )

    use_idf = st.checkbox("Gunakan IDF (tf-idf klasik)", True)
    norm_choice = st.selectbox(
        "Normalisasi vektor",
        options=["l2", "l1", None],
        index=0,
        format_func=lambda x: "Tanpa normalisasi" if x is None else x,
    )

    top_k_terms = st.slider(
        "Jumlah kata teratas yang ditampilkan",
        min_value=10,
        max_value=100,
        value=30,
        step=5,
    )

    run = st.button("🚀 Jalankan Perhitungan TF-IDF", use_container_width=True)

# ======================================================
# 👀 Preview teks sintaksis
# ======================================================
st.subheader("Preview Kolom text_sintaksis (setelah filter)")
st.dataframe(
    df[["incident_number", "text_sintaksis"]].head(10),
    use_container_width=True,
)

# ======================================================
# 🧮 Hitung / ambil TF-IDF (pakai session_state)
# ======================================================

# Kalau belum pernah dihitung dan tombol belum diklik → stop
if not run and st.session_state["tfidf_X"] is None:
    st.info("Atur parameter di sidebar lalu klik **Jalankan Perhitungan TF-IDF**.")
    st.stop()

if run or st.session_state["tfidf_X"] is None:
    # Hitung ulang TF-IDF dan simpan ke session_state
    with st.spinner("🔢 Menghitung matriks TF-IDF..."):
        max_features_param = max_features if max_features > 0 else None

        vectorizer = TfidfVectorizer(
            max_features=max_features_param,
            ngram_range=(1, ngram_max),
            min_df=min_df,
            max_df=max_df_ratio,
            use_idf=use_idf,
            norm=norm_choice,
            lowercase=False,   # teks sudah dilowercase di tahap preprocessing
        )

        X = vectorizer.fit_transform(df["text_sintaksis"].tolist())
        vocab = vectorizer.get_feature_names_out()
        n_docs, n_terms = X.shape

        st.session_state["tfidf_X"] = X
        st.session_state["tfidf_vocab"] = vocab
        st.session_state["tfidf_n_docs"] = n_docs
        st.session_state["tfidf_n_terms"] = n_terms

else:
    # Pakai hasil yang sudah ada di session_state (misal setelah klik tombol download)
    X = st.session_state["tfidf_X"]
    vocab = st.session_state["tfidf_vocab"]
    n_docs = st.session_state["tfidf_n_docs"]
    n_terms = st.session_state["tfidf_n_terms"]

st.success(f"Berhasil menghitung TF-IDF dengan ukuran matriks: {n_docs} dokumen × {n_terms} fitur.")

# ======================================================
# 📊 Statistik global kata (mean TF-IDF)
# ======================================================
st.subheader("📊 Kata dengan Rata-rata TF-IDF Tertinggi")

mean_scores = np.asarray(X.mean(axis=0)).ravel()
order = np.argsort(-mean_scores)  # descending
top_idx = order[:top_k_terms]

top_terms_df = pd.DataFrame({
    "term": vocab[top_idx],
    "mean_tfidf": mean_scores[top_idx],
})

st.dataframe(top_terms_df, use_container_width=True)

chart = alt.Chart(top_terms_df).mark_bar().encode(
    x=alt.X("mean_tfidf:Q", title="Rata-rata TF-IDF"),
    y=alt.Y("term:N", sort="-x", title="Term"),
    tooltip=["term", "mean_tfidf"]
).properties(
    height=max(300, 18 * len(top_terms_df))
)

st.altair_chart(chart, use_container_width=True)

# ======================================================
# 🔍 TF-IDF per dokumen (preview)
# ======================================================
st.subheader("🔍 Preview Matriks TF-IDF per Dokumen")

preview_rows = min(10, n_docs)
tfidf_preview = pd.DataFrame(
    X[:preview_rows].toarray(),
    columns=vocab,
)
tfidf_preview.insert(
    0,
    "incident_number",
    df["incident_number"].iloc[:preview_rows].values,
)

st.dataframe(tfidf_preview, use_container_width=True)

# ======================================================
# 📥 Download Matriks TF-IDF (versi aman memori)
# ======================================================
st.subheader("⬇️ Unduh Matriks TF-IDF")

approx_bytes = n_docs * n_terms * 8  # float64
approx_gib = approx_bytes / (1024 ** 3)

st.write(
    f"Matriks penuh berukuran **{n_docs} × {n_terms}** "
    f"(≈ {approx_gib:.2f} GiB jika disimpan sebagai dense float64)."
)
st.info(
    "Untuk menghindari kehabisan memori, kita hanya akan mengunduh **sample dokumen** "
    "dalam format dense. Untuk seluruh matriks, sebaiknya disimpan dalam format sparse "
    "menggunakan skrip Python offline."
)

# --- Download sample dokumen dalam bentuk dense CSV ---
max_sample = min(5_000, n_docs)
sample_n = st.slider(
    "Jumlah dokumen sample untuk diunduh (dense CSV)",
    min_value=100,
    max_value=max_sample,
    value=min(1_000, max_sample),
    step=100,
)

if st.button("⬇️ Download Sample TF-IDF (dense CSV)", use_container_width=True):
    with st.spinner("Menyiapkan sample matriks TF-IDF..."):
        # 🔁 Random sampling dokumen untuk sample
        rng = np.random.default_rng(seed=42)  # seed agar reproducible
        idx = rng.choice(n_docs, size=sample_n, replace=False)

        tfidf_sample_df = pd.DataFrame(
            X[idx].toarray(),
            columns=vocab,
        )
        tfidf_sample_df.insert(
            0,
            "incident_number",
            df["incident_number"].iloc[idx].values,
        )

        buf = io.BytesIO()
        tfidf_sample_df.to_csv(buf, index=False)
        buf.seek(0)

    st.download_button(
        "Download Sample TF-IDF (dense CSV)",
        data=buf.getvalue(),
        file_name=f"tfidf_text_sintaksis_sample_{sample_n}.csv",
        mime="text/csv",
        use_container_width=True,
    )
