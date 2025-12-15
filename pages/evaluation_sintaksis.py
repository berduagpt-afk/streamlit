# pages/evaluasi_sintaksis.py
# Evaluasi Pendekatan Sintaksis:
# - Cosine Similarity + Threshold
# - Silhouette Score
# - Davies-Bouldin Index
# - Evaluasi Temporal (opsional jika ada kolom tanggal)

import math
from datetime import datetime

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from sqlalchemy import create_engine, text as sa_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity

# ======================================================
# 🔐 Guard login
# ======================================================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login terlebih dahulu untuk mengakses halaman ini.")
    st.stop()

# ======================================================
# 🔌 Koneksi PostgreSQL
# ======================================================
@st.cache_resource(show_spinner=False)
def get_connection():
    cfg = st.secrets["connections"]["postgres"]
    url = (
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )
    return create_engine(url)

# ======================================================
# 📋 Helper: ambil daftar kolom dari tabel
# ======================================================
@st.cache_data(show_spinner=False)
def get_table_columns(table_fullname: str) -> list[str]:
    """
    Mengambil daftar nama kolom dari tabel.
    Kalau gagal (tabel tidak ada, dsb), kembalikan list kosong.
    """
    engine = get_connection()
    try:
        with engine.connect() as conn:
            q = f"SELECT * FROM {table_fullname} LIMIT 0"
            df = pd.read_sql(sa_text(q), conn)
        return list(df.columns)
    except Exception:
        return []

# ======================================================
# ⚙️ Helper: Load data dengan filter dinamis
# ======================================================
@st.cache_data(show_spinner=False)
def load_incident_data(
    table_fullname: str,
    text_col: str,
    date_col: str | None,
    modul_col: str | None,
    start_date: datetime | None,
    end_date: datetime | None,
    max_rows: int,
):
    engine = get_connection()

    # WHERE dinamis
    where_clauses = []
    params = {}

    if date_col:
        if start_date is not None:
            where_clauses.append(f"{date_col} >= :start_date")
            params["start_date"] = start_date
        if end_date is not None:
            where_clauses.append(f"{date_col} <= :end_date")
            params["end_date"] = end_date

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    # Kolom minimum yang diambil
    cols = ["incident_number", text_col]
    if date_col:
        cols.append(date_col)
    if modul_col:
        cols.append(modul_col)

    col_sql = ", ".join(cols)

    sql = f"""
        SELECT {col_sql}
        FROM {table_fullname}
        {where_sql}
        ORDER BY RANDOM()
        LIMIT :max_rows
    """
    params["max_rows"] = max_rows

    with engine.connect() as conn:
        df = pd.read_sql(sa_text(sql), conn, params=params)

    return df

# ======================================================
# 🧠 Helper: TF-IDF
# ======================================================
def compute_tfidf(corpus: list[str], max_features: int = 3000):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
    )
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

# ======================================================
# 🔍 Helper: Sampling pasangan Cosine
# ======================================================
def sample_cosine_pairs(X, n_pairs: int, random_state: int = 42):
    """
    Sampling pasangan dokumen dan menghitung cosine similarity.
    Hindari O(N^2) dengan cara:
    - pilih m dokumen acak
    - hitung full cosine matrix m x m
    - ambil bagian upper triangle sebagai sampel
    """
    rng = np.random.default_rng(random_state)
    n_docs = X.shape[0]

    if n_docs < 2:
        return np.array([]), np.array([]), np.array([])

    # Perkirakan m sehingga m(m-1)/2 ≈ n_pairs
    m_est = int(math.sqrt(2 * n_pairs)) + 1
    m = min(m_est, n_docs)

    idx = rng.choice(n_docs, size=m, replace=False)
    X_sub = X[idx]

    sim_matrix = cosine_similarity(X_sub, dense_output=False)
    sim_dense = sim_matrix.toarray()

    sims = []
    rows = []
    cols = []
    for i in range(m):
        for j in range(i + 1, m):
            sims.append(sim_dense[i, j])
            rows.append(idx[i])
            cols.append(idx[j])

    sims = np.array(sims)
    rows = np.array(rows)
    cols = np.array(cols)

    # Jika terlalu banyak, sampling lagi
    if len(sims) > n_pairs:
        chosen = rng.choice(len(sims), size=n_pairs, replace=False)
        sims = sims[chosen]
        rows = rows[chosen]
        cols = cols[chosen]

    return sims, rows, cols

# ======================================================
# 📊 Helper: Evaluasi KMeans (Silhouette + DBI)
# ======================================================
def evaluate_kmeans_range(X, k_min: int, k_max: int, random_state: int = 42):
    results = []

    for k in range(k_min, k_max + 1):
        if k <= 1 or k >= X.shape[0]:
            continue

        try:
            km = KMeans(
                n_clusters=k,
                random_state=random_state,
                n_init="auto",
                max_iter=300,
            )
            labels = km.fit_predict(X)

            if len(np.unique(labels)) < 2:
                continue

            sil = silhouette_score(X, labels)
            # DBI butuh array dense, hati-hati memori
            dbi = davies_bouldin_score(
                X.toarray() if hasattr(X, "toarray") else X, labels
            )
            results.append({"k": k, "silhouette": sil, "dbi": dbi})
        except Exception as e:
            print(f"Gagal evaluasi k={k}: {e}")
            continue

    if not results:
        return pd.DataFrame(columns=["k", "silhouette", "dbi"])

    return pd.DataFrame(results)

# ======================================================
# 🕒 Helper: Evaluasi Temporal
# ======================================================
def temporal_analysis(df, idx_i, idx_j, sims, date_col: str, threshold: float):
    """
    Menghitung distribusi selisih hari untuk pasangan dengan similarity >= threshold.
    df: data asli (sudah di-subset)
    """
    if not date_col or date_col not in df.columns:
        return pd.DataFrame(), {}

    mask = sims >= threshold
    if mask.sum() == 0:
        return pd.DataFrame(), {}

    dates = pd.to_datetime(df[date_col])

    sel_i = idx_i[mask]
    sel_j = idx_j[mask]
    sel_sims = sims[mask]

    delta_days = []
    recs = []

    for a, b, s in zip(sel_i, sel_j, sel_sims):
        d1 = dates.iloc[a]
        d2 = dates.iloc[b]
        if pd.isna(d1) or pd.isna(d2):
            continue
        d = abs((d1 - d2).days)
        delta_days.append(d)
        recs.append(
            {
                "idx_i": int(a),
                "idx_j": int(b),
                "similarity": float(s),
                "delta_days": int(d),
            }
        )

    if not delta_days:
        return pd.DataFrame(), {}

    df_temp = pd.DataFrame(recs)

    delta_arr = np.array(delta_days)
    summary = {
        "count_pairs": len(delta_arr),
        "median_days": float(np.median(delta_arr)),
        "mean_days": float(np.mean(delta_arr)),
        "p90_days": float(np.percentile(delta_arr, 90)),
        "pct_le_7": float((delta_arr <= 7).mean() * 100),
        "pct_le_14": float((delta_arr <= 14).mean() * 100),
        "pct_le_30": float((delta_arr <= 30).mean() * 100),
    }

    return df_temp, summary

# ======================================================
# 🧱 Layout UI
# ======================================================
st.title("📏 Evaluasi Pendekatan Sintaksis")
st.caption(
    "Tahapan evaluasi berbasis **TF-IDF + Cosine Similarity** dengan metrik "
    "**Cosine Threshold, Silhouette Score, Davies–Bouldin Index, dan Evaluasi Temporal**."
)

with st.sidebar:
    st.header("⚙️ Pengaturan Evaluasi")

    st.markdown("### Sumber Data")
    table_fullname = st.text_input(
        "Nama tabel (schema.tabel)",
        value="lasis_djp.incident_clean",
        help="Contoh: `lasis_djp.incident_clean` atau `lasis_djp.incident_raw`",
    )

    # Ambil daftar kolom (kalau tabel valid)
    cols_available = []
    if table_fullname.strip():
        cols_available = get_table_columns(table_fullname.strip())

    if cols_available:
        st.markdown(
            f"<small>Ditemukan <b>{len(cols_available)}</b> kolom di tabel ini.</small>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<small><i>Tidak bisa membaca kolom tabel (cek nama schema.tabel atau koneksi DB).</i></small>",
            unsafe_allow_html=True,
        )

    # Pilih kolom teks
    if cols_available:
        default_text_idx = 0
        for i, c in enumerate(cols_available):
            if c.lower() in ["text_sintaksis", "incident_clean", "incident_text"]:
                default_text_idx = i
                break

        text_col = st.selectbox(
            "Kolom teks sintaksis",
            options=cols_available,
            index=default_text_idx,
            help="Kolom hasil preprocessing sintaksis (TF-IDF akan dihitung dari sini).",
        )
    else:
        text_col = st.text_input(
            "Kolom teks sintaksis",
            value="text_sintaksis",
        )

    # Pilih kolom tanggal (opsional)
    if cols_available:
        date_options = ["(Tidak digunakan)"] + cols_available
        date_default_idx = 0
        for i, c in enumerate(cols_available, start=1):
            if c.lower() in ["tgl_submit", "tanggal", "tgl_tiket", "date"]:
                date_default_idx = i
                break

        selected_date = st.selectbox(
            "Kolom tanggal tiket (opsional)",
            options=date_options,
            index=date_default_idx,
            help="Pilih '(Tidak digunakan)' jika tabel tidak punya kolom tanggal.",
        )
        date_col = "" if selected_date == "(Tidak digunakan)" else selected_date
    else:
        date_col = st.text_input(
            "Kolom tanggal tiket (opsional)",
            value="",
            help="Boleh dikosongkan jika tidak ada.",
        )

    # Pilih kolom modul (opsional, untuk analisis tambahan)
    if cols_available:
        modul_options = ["(Tidak digunakan)"] + cols_available
        modul_default_idx = 0
        for i, c in enumerate(cols_available, start=1):
            if c.lower() in ["modul", "module"]:
                modul_default_idx = i
                break

        selected_modul = st.selectbox(
            "Kolom modul (opsional)",
            options=modul_options,
            index=modul_default_idx,
            help="Tidak wajib. Berguna jika ingin analisis per modul.",
        )
        modul_col = "" if selected_modul == "(Tidak digunakan)" else selected_modul
    else:
        modul_col = st.text_input(
            "Kolom modul (opsional)",
            value="",
            help="Boleh dikosongkan.",
        )

    st.markdown("---")
    st.markdown("### Filter Waktu (aktif hanya jika kolom tanggal diisi)")

    today = datetime.today().date()
    default_start = datetime(today.year, 1, 1).date()

    if date_col:
        start_date = st.date_input(
            "Tanggal mulai",
            value=default_start,
        )
        end_date = st.date_input(
            "Tanggal selesai",
            value=today,
        )
    else:
        start_date = None
        end_date = None
        st.markdown(
            "<small>Filter tanggal nonaktif karena kolom tanggal tidak dipilih.</small>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### Parameter Evaluasi")

    max_rows = st.slider(
        "Maksimum tiket untuk evaluasi",
        min_value=500,
        max_value=5000,
        value=2000,
        step=500,
        help="Semakin besar nilai ini, semakin berat perhitungan.",
    )

    max_features = st.slider(
        "Maksimum fitur TF-IDF",
        min_value=1000,
        max_value=8000,
        value=3000,
        step=500,
    )

    n_pairs = st.slider(
        "Jumlah pasangan acak untuk evaluasi Cosine",
        min_value=1000,
        max_value=50000,
        value=10000,
        step=1000,
    )

    cos_threshold = st.slider(
        "Cosine similarity threshold (untuk analisis & temporal)",
        min_value=0.5,
        max_value=0.99,
        value=0.8,
        step=0.01,
    )

    k_min, k_max = st.slider(
        "Rentang jumlah cluster KMeans (untuk Silhouette & DBI)",
        min_value=2,
        max_value=15,
        value=(3, 8),
    )

    random_state = st.number_input(
        "Random seed",
        min_value=0,
        max_value=9999,
        value=42,
        step=1,
    )

    run_btn = st.button("🚀 Jalankan Evaluasi", type="primary")

# ======================================================
# 🚀 Proses utama
# ======================================================
if run_btn:
    if not table_fullname.strip():
        st.error("Nama tabel belum diisi.")
        st.stop()

    # 1) Load data
    with st.spinner("Mengambil data dari database..."):
        try:
            df = load_incident_data(
                table_fullname=table_fullname.strip(),
                text_col=text_col,
                date_col=date_col.strip() or None,
                modul_col=modul_col.strip() or None,
                start_date=start_date,
                end_date=end_date,
                max_rows=max_rows,
            )
        except Exception as e:
            st.error(f"Gagal mengambil data dari database: {e}")
            st.stop()

    if df.empty:
        st.warning("Data kosong dengan kombinasi pengaturan saat ini.")
        st.stop()

    st.success(f"Berhasil memuat {len(df):,} baris data untuk evaluasi.")

    with st.expander("🔎 Pratinjau Data"):
        st.write(df.head())

    # 2) TF-IDF
    with st.spinner("Menghitung TF-IDF (pendekatan sintaksis)..."):
        texts = df[text_col].fillna("").astype(str).tolist()
        X, vectorizer = compute_tfidf(texts, max_features=max_features)

    st.info(
        f"TF-IDF selesai dihitung untuk **{X.shape[0]:,} tiket** "
        f"dengan **{X.shape[1]:,} fitur**."
    )

    # 3) Evaluasi Cosine + Threshold
    st.subheader("1️⃣ Distribusi Cosine Similarity & Threshold")

    with st.spinner("Sampling pasangan dokumen dan menghitung Cosine Similarity..."):
        sims, idx_i, idx_j = sample_cosine_pairs(X, n_pairs=n_pairs, random_state=random_state)

    if sims.size == 0:
        st.warning("Tidak cukup data untuk menghitung pasangan Cosine Similarity.")
    else:
        df_sims = pd.DataFrame({"similarity": sims})

        stats = {
            "count_pairs": len(sims),
            "mean": float(np.mean(sims)),
            "median": float(np.median(sims)),
            "p90": float(np.percentile(sims, 90)),
            "pct_ge_threshold": float((sims >= cos_threshold).mean() * 100),
        }

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Jumlah pasangan", f"{stats['count_pairs']:,}")
        c2.metric("Rata-rata", f"{stats['mean']:.3f}")
        c3.metric("Median", f"{stats['median']:.3f}")
        c4.metric("P90", f"{stats['p90']:.3f}")
        c5.metric(f"% ≥ {cos_threshold:.2f}", f"{stats['pct_ge_threshold']:.2f} %")

        hist_chart = (
            alt.Chart(df_sims)
            .mark_bar()
            .encode(
                x=alt.X("similarity:Q", bin=alt.Bin(maxbins=40), title="Cosine similarity"),
                y=alt.Y("count()", title="Jumlah pasangan"),
                tooltip=["count()"],
            )
            .properties(
                height=300,
                width="container",
                title="Distribusi Cosine Similarity (sampling pasangan dokumen)",
            )
        )
        st.altair_chart(hist_chart, use_container_width=True)

        st.caption(
            "Distribusi ini dapat digunakan untuk menentukan **ambang batas (threshold)** "
            "kemiripan sintaksis yang akan dipakai sebagai dasar pembentukan pasangan/klaster "
            "insiden serupa dalam pendekatan sintaksis."
        )

    st.markdown("---")

    # 4) Evaluasi KMeans: Silhouette & DBI
    st.subheader("2️⃣ Evaluasi Klaster KMeans (Silhouette Score & Davies–Bouldin Index)")

    with st.spinner("Melakukan klasterisasi KMeans untuk berbagai nilai k..."):
        df_k = evaluate_kmeans_range(X, k_min=k_min, k_max=k_max, random_state=random_state)

    if df_k.empty:
        st.warning("Tidak berhasil menghitung metrik klaster (silhouette/DBI) untuk rentang k ini.")
    else:
        st.dataframe(df_k, use_container_width=True)

        best_sil = df_k.loc[df_k["silhouette"].idxmax()]
        best_dbi = df_k.loc[df_k["dbi"].idxmin()]

        c1, c2 = st.columns(2)
        with c1:
            st.metric(
                "k terbaik (Silhouette Score)",
                int(best_sil["k"]),
                f"{best_sil['silhouette']:.3f}",
            )
        with c2:
            st.metric(
                "k terbaik (Davies-Bouldin Index)",
                int(best_dbi["k"]),
                f"{best_dbi['dbi']:.3f}",
            )

        sil_chart = (
            alt.Chart(df_k)
            .mark_line(point=True)
            .encode(
                x=alt.X("k:Q", title="Jumlah cluster (k)"),
                y=alt.Y("silhouette:Q", title="Silhouette Score"),
                tooltip=["k", "silhouette"],
            )
            .properties(
                height=300,
                title="Silhouette Score per nilai k",
            )
        )

        dbi_chart = (
            alt.Chart(df_k)
            .mark_line(point=True)
            .encode(
                x=alt.X("k:Q", title="Jumlah cluster (k)"),
                y=alt.Y("dbi:Q", title="Davies–Bouldin Index"),
                tooltip=["k", "dbi"],
            )
            .properties(
                height=300,
                title="Davies–Bouldin Index per nilai k (semakin kecil semakin baik)",
            )
        )

        st.altair_chart(sil_chart, use_container_width=True)
        st.altair_chart(dbi_chart, use_container_width=True)

        st.caption(
            "Metrik **Silhouette Score** dan **Davies–Bouldin Index** digunakan untuk "
            "menilai kualitas pemisahan klaster hasil pendekatan sintaksis."
        )

    st.markdown("---")

    # 5) Evaluasi Temporal
    st.subheader("3️⃣ Evaluasi Temporal untuk Pasangan Mirip (Cosine ≥ Threshold)")

    if not date_col:
        st.info("Evaluasi temporal dinonaktifkan karena kolom tanggal tidak dipilih.")
    elif sims.size == 0:
        st.warning("Temporal tidak bisa dihitung karena tidak ada pasangan similarity.")
    else:
        df_temp, temp_summary = temporal_analysis(
            df=df,
            idx_i=idx_i,
            idx_j=idx_j,
            sims=sims,
            date_col=date_col,
            threshold=cos_threshold,
        )

        if df_temp.empty:
            st.warning(
                "Tidak ditemukan pasangan tiket dengan informasi tanggal "
                f"dan cosine similarity ≥ {cos_threshold:.2f}."
            )
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Jumlah pasangan valid", f"{temp_summary['count_pairs']:,}")
            c2.metric("Median selisih hari", f"{temp_summary['median_days']:.1f}")
            c3.metric("Rata-rata selisih hari", f"{temp_summary['mean_days']:.1f}")
            c4.metric("P90 selisih hari", f"{temp_summary['p90_days']:.1f}")

            c5, c6, c7 = st.columns(3)
            c5.metric("≤ 7 hari", f"{temp_summary['pct_le_7']:.1f} %")
            c6.metric("≤ 14 hari", f"{temp_summary['pct_le_14']:.1f} %")
            c7.metric("≤ 30 hari", f"{temp_summary['pct_le_30']:.1f} %")

            hist_temp = (
                alt.Chart(df_temp)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "delta_days:Q",
                        bin=alt.Bin(maxbins=40),
                        title="Selisih hari antar tiket mirip (Cosine ≥ threshold)",
                    ),
                    y=alt.Y("count()", title="Jumlah pasangan"),
                    tooltip=["count()"],
                )
                .properties(
                    height=300,
                    width="container",
                    title="Distribusi selisih hari untuk pasangan tiket mirip",
                )
            )
            st.altair_chart(hist_temp, use_container_width=True)

            with st.expander("📋 Contoh pasangan tiket mirip (top 20 berdasarkan similarity)"):
                df_top_pairs = df_temp.sort_values("similarity", ascending=False).head(20)
                st.dataframe(df_top_pairs, use_container_width=True)

            st.caption(
                "Evaluasi temporal ini menunjukkan apakah tiket yang **mirip secara sintaksis** "
                f"(Cosine ≥ {cos_threshold:.2f}) juga muncul berdekatan dalam dimensi waktu."
            )

    st.success("✅ Tahapan evaluasi pendekatan sintaksis selesai dijalankan.")
else:
    st.info("Atur parameter di sidebar, lalu klik **🚀 Jalankan Evaluasi** untuk memulai.")
