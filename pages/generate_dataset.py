# pages/generate_random_incident.py
# Halaman untuk generate data random dari tabel lasis_djp.incident_raw

import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

# ============================================================
# 1️⃣ Guard: hanya untuk user yang login
# ============================================================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login dulu untuk mengakses halaman ini.")
    st.stop()

# ============================================================
# 2️⃣ Styling sederhana
# ============================================================
st.markdown(
    """
    <style>
    .page-title {
        color:#0B3A82;
        font-weight:800;
        font-size:1.8rem;
        margin-bottom:0.25rem;
    }
    .page-subtitle {
        color:#64748B;
        font-size:0.95rem;
        margin-bottom:1.5rem;
    }
    .info-box {
        padding:0.75rem 1rem;
        border-radius:0.75rem;
        border:1px solid #E5E7EB;
        background-color:#F9FAFB;
        margin-bottom:0.75rem;
        font-size:0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="page-title">🎲 Generate Data Random Incident</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">Menambah dataset dengan data random yang diambil dari tabel '
    '<b>lasis_djp.incident_raw</b>.</div>',
    unsafe_allow_html=True,
)

# ============================================================
# 3️⃣ Fungsi koneksi database (via st.connection)
#    ⚠️ PASTIKAN nama koneksi ("postgres_lasis") sama
#    dengan yang kamu gunakan di file lain.
#    Di secrets.toml biasanya ada:
#
#    [connections.postgres_lasis]
#    dialect = "postgresql"
#    host = "localhost"
#    port = 5432
#    database = "postgres"
#    username = "postgres"
#    password = "xxxxx"
# ============================================================

@st.cache_resource
def get_engine():
    try:
        # 🔧 GANTI "postgres_lasis" jika di project kamu namanya berbeda
        conn = st.connection("postgres_lasis", type="sql")
        return conn.engine
    except Exception as e:
        st.error(
            "Gagal membuat koneksi database dari secrets.\n\n"
            "Pastikan nama koneksi di st.connection(...) sama dengan yang ada di .streamlit/secrets.toml.\n"
            f"Detail error: {e}"
        )
        return None


# ============================================================
# 4️⃣ Load data incident_raw sebagai sumber
#    - Prioritas 1: st.session_state['incident_raw']
#    - Prioritas 2: SELECT * FROM lasis_djp.incident_raw
# ============================================================

def load_incident_raw():
    # 1) Coba dari session_state
    if "incident_raw" in st.session_state:
        st.info("Menggunakan dataframe dari st.session_state['incident_raw'].")
        return st.session_state["incident_raw"].copy()

    # 2) Coba dari database
    engine = get_engine()
    if engine is None:
        st.error(
            "Tidak bisa membuat koneksi database, dan session_state['incident_raw'] juga belum ada.\n"
            "Silakan cek kembali pengaturan secrets / nama koneksi DB."
        )
        return None

    try:
        # Jika schema berbeda, sesuaikan nama schema di sini
        query = "SELECT * FROM lasis_djp.incident_raw"
        df = pd.read_sql(query, con=engine)
        st.success(f"Berhasil membaca tabel lasis_djp.incident_raw dari database: {df.shape[0]:,} baris.")
        return df
    except Exception as e:
        st.error(f"Gagal membaca tabel lasis_djp.incident_raw dari database: {e}")
        return None


st.markdown("### 1. Sumber Data `incident_raw`")

df_src = load_incident_raw()
if df_src is None or df_src.empty:
    st.stop()

with st.expander("Lihat 5 baris pertama incident_raw"):
    st.dataframe(df_src.head())

st.markdown(
    f"<div class='info-box'>Dataset incident_raw memiliki <b>{df_src.shape[0]:,}</b> baris "
    f"dan <b>{df_src.shape[1]}</b> kolom.</div>",
    unsafe_allow_html=True,
)

# ============================================================
# 5️⃣ Input jumlah data random yang ingin ditambahkan
# ============================================================

st.markdown("### 2. Konfigurasi Jumlah Data Random")

col_num, col_mode = st.columns([1, 1])

with col_num:
    n_new = st.number_input(
        "Jumlah data baru yang ingin digenerate",
        min_value=1,
        max_value=100_000,
        value=100,
        step=10,
        help="Isi berapa banyak baris baru yang ingin ditambahkan ke dataset.",
    )

with col_mode:
    mode = st.selectbox(
        "Metode random",
        [
            "Sample baris (duplikasi acak dari incident_raw)",
            "Shuffle per kolom (kombinasi acak antar baris)",
        ],
        help=(
            "Sample baris: setiap data baru adalah salinan baris incident_raw yang dipilih acak.\n"
            "Shuffle per kolom: isi tiap kolom dipilih acak secara independen dari kolom tersebut."
        ),
    )

st.markdown("### 3. Opsi Output")

save_to_session = st.checkbox(
    "Simpan hasil ke `st.session_state['incident_random']`",
    value=True,
)

save_to_db = st.checkbox(
    "Simpan ke database (tabel `lasis_djp.incident_random`)",
    value=False,
    help="Jika dicentang, data random akan di-INSERT ke tabel lasis_djp.incident_random di database.",
)

# ============================================================
# 6️⃣ Fungsi generate data random
# ============================================================

def generate_random_rows(df: pd.DataFrame, n: int, mode: str) -> pd.DataFrame:
    df = df.copy()

    if mode.startswith("Sample baris"):
        # Oversampling baris: setiap baris baru adalah baris existing yang di-sample dengan replacement
        df_new = df.sample(n=n, replace=True, ignore_index=True)
    else:
        # Shuffle per kolom: ambil nilai acak per kolom (independen per kolom)
        data = {}
        for col in df.columns:
            series = df[col].dropna()
            if series.empty:
                data[col] = [None] * n
            else:
                sampled = series.sample(n=n, replace=True).reset_index(drop=True)
                data[col] = sampled
        df_new = pd.DataFrame(data)

    return df_new


# ============================================================
# 7️⃣ Tombol proses generate + simpan
# ============================================================

st.markdown("### 4. Generate Data Random dari incident_raw")

if st.button("🚀 Generate Data Random", type="primary"):
    with st.spinner(f"Mengenerate {n_new} baris data random ..."):
        df_random = generate_random_rows(df_src, n_new, mode)

        # Tambahkan kolom penanda
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        df_random["is_random"] = True
        df_random["random_batch_id"] = batch_id

    st.success(f"Berhasil mengenerate {n_new} baris data random dari incident_raw.")

    # Simpan ke session_state
    if save_to_session:
        st.session_state["incident_random"] = df_random
        st.info("Hasil disimpan ke st.session_state['incident_random'].")

    # Simpan ke database (tabel baru lasis_djp.incident_random)
    if save_to_db:
        engine = get_engine()
        if engine is None:
            st.error("Gagal konek ke database, data random tidak bisa disimpan ke tabel lasis_djp.incident_random.")
        else:
            try:
                df_random.to_sql(
                    "incident_random",
                    con=engine,
                    schema="lasis_djp",   # ⚠️ sesuaikan kalau schema kamu beda
                    if_exists="append",   # append ke tabel kalau sudah ada
                    index=False,
                )
                st.success(
                    f"✅ {len(df_random):,} baris data random berhasil disimpan ke tabel `lasis_djp.incident_random`."
                )
            except Exception as e:
                st.error(f"Gagal menyimpan ke tabel lasis_djp.incident_random: {e}")

    # Tampilkan preview
    st.markdown("#### Preview Data Random (5 baris pertama)")
    st.dataframe(df_random.head())

    # Tombol download CSV
    csv_bytes = df_random.to_csv(index=False).encode("utf-8")
    st.download_button(
        "💾 Download Data Random sebagai CSV",
        data=csv_bytes,
        file_name=f"incident_random_{n_new}_rows_{batch_id}.csv",
        mime="text/csv",
    )

else:
    st.info("Atur jumlah data dan metode random, lalu klik **Generate Data Random**.")
