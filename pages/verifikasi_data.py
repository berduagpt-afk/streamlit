# pages/verifikasi_data.py
# Verifikasi Kualitas Data Insiden — dari tabel lasis_djp.incident_raw

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

# ======================================================
# 🔐 GUARD LOGIN
# ======================================================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login dulu untuk mengakses halaman ini.")
    st.stop()

# Nama tabel output hasil pembersihan
OUTPUT_SCHEMA = "lasis_djp"
OUTPUT_TABLE_NAME = "incident_kelayakan"

# Default threshold minimum jumlah tiket per modul
MIN_TICKETS_PER_MODUL = 1154  # bisa diubah lewat slider di UI


# ======================================================
# 🔐 KONEKSI DATABASE POSTGRESQL
# ======================================================
def get_connection():
    """Bangun koneksi SQLAlchemy ke PostgreSQL dari .streamlit/secrets.toml"""
    cfg = st.secrets["connections"]["postgres"]
    url = (
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )
    return create_engine(url, pool_pre_ping=True)


@st.cache_data(show_spinner=False)
def load_from_db(schema="lasis_djp", table="incident_raw") -> pd.DataFrame:
    """
    Ambil data dari tabel incident_raw.
    SELECT eksplisit (7 kolom) supaya konsisten dengan statistik_dataset.py.
    """
    eng = get_connection()
    try:
        q = text(
            f'SELECT "tgl_submit","incident_number","site","assignee",'
            f'"modul","sub_modul","detailed_decription" '
            f'FROM "{schema}"."{table}"'
        )
        df = pd.read_sql(q, con=eng)
    finally:
        eng.dispose()
    return df


def safe_parse_datetime(series: pd.Series) -> pd.Series:
    """
    Parser 2-tahap:
    1) format eksplisit %Y-%m-%d %H:%M:%S.%f  (cocok '2024-05-27 14:43:59.000')
    2) fallback infer_datetime_format untuk format lain.
    """
    s = (
        series.astype(str)
        .str.strip()
        .replace({"": None, "None": None, "nan": None, "NaN": None})
    )
    parsed = pd.to_datetime(s, errors="coerce", format="%Y-%m-%d %H:%M:%S.%f")
    mask_fail = parsed.isna()
    if mask_fail.any():
        parsed.loc[mask_fail] = pd.to_datetime(
            s[mask_fail], errors="coerce", infer_datetime_format=True
        )
    return parsed


def load_data() -> pd.DataFrame:
    """
    Loader utama: panggil load_from_db lalu normalisasi kolom
    agar logika NA/Empty konsisten dengan statistik_dataset.py.
    """
    df = load_from_db()

    # alias jika ada salah nama deskripsi
    if "detailed_description" in df.columns and "detailed_decription" not in df.columns:
        df.rename(columns={"detailed_description": "detailed_decription"}, inplace=True)

    # parse tanggal robust
    if "tgl_submit" in df.columns:
        df["tgl_submit"] = safe_parse_datetime(df["tgl_submit"])

    # normalisasi kategori & teks kunci
    for c in ["site", "assignee", "modul", "sub_modul", "incident_number", "detailed_decription"]:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .replace({"": None, "None": None, "nan": None, "NaN": None})
            )
    return df


def save_dataframe(df: pd.DataFrame, table_name: str, schema: str = "lasis_djp", if_exists: str = "replace"):
    """Simpan DataFrame ke tabel PostgreSQL (dengan chunksize)"""
    engine = get_connection()
    try:
        df.to_sql(
            table_name,
            engine,
            schema=schema,
            if_exists=if_exists,
            index=False,
            chunksize=10_000,
        )
    finally:
        engine.dispose()


# ======================================================
# 🔧 FUNGSI KUALITAS DATA
# ======================================================
def column_quality(d: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung kualitas data per kolom:
    - NA   : nilai benar-benar NaN/None
    - EmptyString : nilai string kosong setelah strip()
    """
    na_cnt = d.isna().sum().rename("NA")
    empty_cnt = d.apply(lambda s: s.astype(str).str.strip().eq("").sum()).rename("EmptyString")
    total = len(d)
    out = pd.concat([na_cnt, empty_cnt], axis=1)
    out["%NA"] = (out["NA"] / total * 100).round(2)
    out["%Empty"] = (out["EmptyString"] / total * 100).round(2)
    return out


# ======================================================
# 🧷 SETUP HALAMAN
# ======================================================
st.title("🔎 Analisis & Pembersihan Kelayakan Data Insiden (incident_raw)")
st.caption(
    "Analisis kualitas data dari tabel **lasis_djp.incident_raw**, "
    "lalu menghapus baris dengan nilai NA/Empty di kolom kunci, "
    "deskripsi tiket yang duplikat, dan modul dengan jumlah tiket di bawah "
    "ambang batas kelayakan, kemudian menyimpan hasilnya ke tabel baru."
)

# ======================================================
# 📦 AMBIL & SIAPKAN DATA
# ======================================================
with st.spinner("📦 Mengambil data dari database (lasis_djp.incident_raw)..."):
    try:
        df = load_data()
        st.success(f"✅ Berhasil memuat {len(df):,} baris dari lasis_djp.incident_raw")
    except Exception as e:
        st.error(f"Gagal mengambil data dari database: {e}")
        st.stop()

if df.empty:
    st.warning("Dataset kosong. Pastikan tabel lasis_djp.incident_raw berisi data hasil upload.")
    st.stop()

# Kolom kunci dan kolom deskripsi global
key_cols = [
    c
    for c in [
        "tgl_submit",
        "incident_number",
        "site",
        "assignee",
        "modul",
        "sub_modul",
        "detailed_decription",
    ]
    if c in df.columns
]

candidate_desc_cols = ["detailed_decription", "isi_permasalahan", "deskripsi"]
DESC_COL = next((c for c in candidate_desc_cols if c in df.columns), None)

# ======================================================
# 📋 RINGKASAN UMUM
# ======================================================
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Jumlah baris", f"{df.shape[0]:,}")
with c2:
    st.metric("Jumlah kolom", f"{df.shape[1]:,}")
with c3:
    st.metric("Kolom bertipe object", f"{df.select_dtypes(include=['object']).shape[1]:,}")

st.subheader("Preview Data")
st.dataframe(df.head(10), use_container_width=True)

# ======================================================
# 🧮 TAB ANALISIS
# ======================================================
tab_null, tab_dupe, tab_clean = st.tabs(
    [
        "📊 Kualitas Data (NA & Empty)",
        "🔁 Duplikasi Deskripsi Tiket",
        "🧹 Pembersihan & Simpan",
    ]
)

# ------------------------------------------------------
# 📊 TAB 1 — KUALITAS DATA (NA & EMPTY)
# ------------------------------------------------------
with tab_null:
    st.markdown("### 📊 Kualitas Data per Kolom (NA & EmptyString)")

    if key_cols:
        qual = column_quality(df[key_cols])
    else:
        qual = column_quality(df)

    mask_problem = (qual["NA"] > 0) | (qual["EmptyString"] > 0)
    cols_with_issue = qual[mask_problem]
    cols_clean = qual[~mask_problem]

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Kolom dengan NA/Empty", f"{cols_with_issue.shape[0]:,}")
    with c2:
        st.metric("Kolom tanpa NA/Empty", f"{cols_clean.shape[0]:,}")

    qual_sorted = qual.sort_values(["NA", "EmptyString"], ascending=False)
    st.markdown("#### Tabel Kualitas Data (diurutkan dari NA/Empty terbesar)")
    st.dataframe(qual_sorted, use_container_width=True)

    st.markdown("#### Grafik Kolom dengan NA/Empty Terbanyak")
    top_n = min(20, qual_sorted.shape[0])
    chart_data = (
        qual_sorted.head(top_n)[["NA", "EmptyString"]]
        .reset_index()
        .rename(columns={"index": "kolom"})
        .set_index("kolom")
    )
    st.bar_chart(chart_data)

# ------------------------------------------------------
# 🔁 TAB 2 — DUPLIKAT DESKRIPSI TIKET
# ------------------------------------------------------
with tab_dupe:
    st.markdown("### 🔁 Analisis Deskripsi Tiket Duplikat")

    if DESC_COL is None:
        st.error(
            "Tidak menemukan kolom deskripsi tiket.\n"
            f"Kolom yang dicari: {candidate_desc_cols}\n"
            "Periksa kembali struktur tabel incident_raw."
        )
    else:
        st.info(f"Kolom deskripsi yang digunakan: **{DESC_COL}**")

        df_desc = df[df[DESC_COL].notna()].copy()
        df_desc[DESC_COL] = df_desc[DESC_COL].astype(str).str.strip()
        df_desc = df_desc[df_desc[DESC_COL].ne("")]

        if df_desc.empty:
            st.warning("Tidak ada deskripsi tiket yang dapat dianalisis.")
        else:
            # Tentukan kolom ID tiket
            id_col = "Incident_Number"
            if id_col not in df_desc.columns and "incident_number" in df_desc.columns:
                id_col = "incident_number"
            if id_col not in df_desc.columns:
                id_col = None

            if id_col:
                dup_group = (
                    df_desc.groupby(DESC_COL)
                    .agg(
                        jumlah_tiket=(id_col, "nunique"),
                        contoh_incident=(id_col, lambda s: ", ".join(map(str, s.head(5)))),
                    )
                    .reset_index()
                )
            else:
                dup_group = (
                    df_desc.groupby(DESC_COL)
                    .size()
                    .reset_index(name="jumlah_tiket")
                )
                dup_group["contoh_incident"] = "(Kolom ID tiket tidak tersedia)"

            dup_group = dup_group[dup_group["jumlah_tiket"] > 1].sort_values(
                "jumlah_tiket", ascending=False
            )

            if dup_group.empty:
                st.success("Tidak ditemukan deskripsi tiket duplikat (berdasarkan kesamaan teks persis).")
            else:
                total_groups = dup_group.shape[0]
                total_tickets = int(dup_group["jumlah_tiket"].sum())

                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Kelompok deskripsi duplikat", f"{total_groups:,}")
                with c2:
                    st.metric("Total tiket dalam kelompok duplikat", f"{total_tickets:,}")

                st.markdown("#### Ringkasan Deskripsi Duplikat")
                st.dataframe(
                    dup_group[[DESC_COL, "jumlah_tiket", "contoh_incident"]],
                    use_container_width=True,
                )

                st.markdown("#### Detail Tiket yang Termasuk Duplikat")
                dup_rows = df_desc[df_desc[DESC_COL].isin(dup_group[DESC_COL])].copy()

                cols_to_show = []
                for c in [id_col, DESC_COL, "modul", "sub_modul", "site", "assignee"]:
                    if c and c in dup_rows.columns:
                        cols_to_show.append(c)
                if DESC_COL not in cols_to_show:
                    cols_to_show.append(DESC_COL)

                st.dataframe(
                    dup_rows[cols_to_show].sort_values(by=[DESC_COL]),
                    use_container_width=True,
                    height=400,
                )

# ------------------------------------------------------
# 🧹 TAB 3 — PEMBERSIHAN & SIMPAN KE TABEL BARU
# ------------------------------------------------------
with tab_clean:
    st.markdown("### 🧹 Pembersihan Data & Simpan ke Tabel Baru")

    min_tickets_threshold = st.number_input(
        "Ambang minimum jumlah tiket per modul (threshold kelayakan)",
        min_value=100,
        max_value=10_000,
        value=MIN_TICKETS_PER_MODUL,
        step=50,
        help="Modul dengan jumlah tiket di bawah angka ini akan dikeluarkan dari dataset kelayakan.",
    )

    st.markdown(
        f"""
        Urutan pembersihan yang akan dilakukan:
        1. **Menghapus baris yang memiliki NA atau Empty** pada kolom kunci:
           `{', '.join(key_cols) or '(tidak ada kolom kunci)'}`.
        2. **Menghapus tiket dengan deskripsi duplikat** (berdasarkan kolom deskripsi yang sama, menyisakan satu tiket pertama).
        3. **Dari hasil langkah 1–2**, menghitung jumlah tiket per modul dan
           **menghapus tiket pada modul yang jumlah tiketnya di bawah {min_tickets_threshold} tiket**.
        4. Menyimpan hasilnya ke tabel **`{OUTPUT_SCHEMA}.{OUTPUT_TABLE_NAME}`** di PostgreSQL (`if_exists="replace"`).
        """
    )

    if DESC_COL:
        st.write("**Kolom deskripsi untuk cek duplikat:**", DESC_COL)
    else:
        st.write("**Kolom deskripsi:** tidak tersedia")

    run_clean = st.button("🚀 Jalankan Pembersihan & Simpan ke Database", use_container_width=True)

    if run_clean:
        with st.spinner("🧹 Membersihkan data dan menyimpan ke tabel baru..."):

            df_work = df.copy()

            # ========== LANGKAH 1: DROP NA/EMPTY DI KOLOM KUNCI ==========
            if key_cols:
                mask_null_any = df_work[key_cols].isna().any(axis=1)
                mask_empty_any = df_work[key_cols].apply(
                    lambda s: s.astype(str).str.strip().eq("")
                ).any(axis=1)
                mask_bad_null = mask_null_any | mask_empty_any
            else:
                mask_bad_null = pd.Series(False, index=df_work.index)

            df_after_null = df_work[~mask_bad_null].copy()

            # ========== LANGKAH 2: DROP DUPLIKAT DESKRIPSI ==========
            if DESC_COL:
                desc_series_after_null = (
                    df_after_null[DESC_COL].fillna("").astype(str).str.strip()
                )
                mask_dupe_desc_after_null = desc_series_after_null.duplicated(keep="first")

                mask_dupe_desc = pd.Series(False, index=df_work.index)
                mask_dupe_desc.loc[df_after_null.index] = mask_dupe_desc_after_null
            else:
                mask_dupe_desc = pd.Series(False, index=df_work.index)

            df_after_null_dupe = df_work[~(mask_bad_null | mask_dupe_desc)].copy()

            # ========== LANGKAH 3: DROP MODUL DENGAN JUMLAH TIKET < THRESHOLD ==========
            if "modul" in df_after_null_dupe.columns:
                counts_modul_after = df_after_null_dupe["modul"].value_counts(dropna=False)
                moduls_low = counts_modul_after[counts_modul_after < min_tickets_threshold].index

                mask_low_modul = pd.Series(False, index=df_work.index)
                mask_low_modul.loc[df_after_null_dupe.index] = df_after_null_dupe["modul"].isin(moduls_low)
            else:
                mask_low_modul = pd.Series(False, index=df_work.index)

            # ========== GABUNG & ALOKASI ALASAN DROP ==========
            mask_drop = mask_bad_null | mask_dupe_desc | mask_low_modul

            # Alasan eksklusif agar tidak double-count
            reason_null      = mask_bad_null
            reason_dupe      = ~reason_null & mask_dupe_desc
            reason_low_modul = ~reason_null & ~reason_dupe & mask_low_modul

            df_clean = df_work[~mask_drop].copy()

            # Statistik ringkas
            n_total          = len(df_work)
            n_drop_null      = int(reason_null.sum())
            n_drop_dupe      = int(reason_dupe.sum())
            n_drop_low_modul = int(reason_low_modul.sum())
            n_final          = len(df_clean)

            # ========== RINGKASAN MODUL SEBELUM & SESUDAH ==========
            if "modul" in df_work.columns:
                counts_raw      = df_work["modul"].value_counts(dropna=False)
                counts_after_12 = df_after_null_dupe["modul"].value_counts(dropna=False)
                counts_final    = df_clean["modul"].value_counts(dropna=False)

                modul_summary = (
                    pd.concat(
                        [
                            counts_raw.rename("jumlah_awal"),
                            counts_after_12.rename("setelah_langkah_1_2"),
                            counts_final.rename("setelah_threshold"),
                        ],
                        axis=1,
                    )
                    .fillna(0)
                    .astype(int)
                )

                modul_summary["status_kelayakan"] = modul_summary["setelah_threshold"].apply(
                    lambda x: "Layak" if x >= min_tickets_threshold and x > 0 else "Tidak Layak"
                )

                n_modul_awal  = modul_summary.shape[0]
                n_modul_layak = (modul_summary["status_kelayakan"] == "Layak").sum()
                coverage_pct  = (n_final / n_total * 100) if n_total > 0 else 0.0
            else:
                modul_summary = None
                n_modul_awal  = 0
                n_modul_layak = 0
                coverage_pct  = (n_final / n_total * 100) if n_total > 0 else 0.0

            # ========== SIMPAN KE DATABASE ==========
            try:
                save_dataframe(
                    df_clean,
                    table_name=OUTPUT_TABLE_NAME,
                    schema=OUTPUT_SCHEMA,
                    if_exists="replace",
                )
                st.success(
                    f"✅ Pembersihan selesai dan tersimpan ke "
                    f"`{OUTPUT_SCHEMA}.{OUTPUT_TABLE_NAME}`."
                )
                st.write(f"- Total baris awal                         : **{n_total:,}**")
                st.write(f"- Dihapus karena NA/Empty                  : **{n_drop_null:,}**")
                st.write(f"- Dihapus karena duplikat deskripsi        : **{n_drop_dupe:,}**")
                st.write(
                    f"- Dihapus karena modul < {min_tickets_threshold} tiket : "
                    f"**{n_drop_low_modul:,}**"
                )
                st.write(f"- Total baris setelah pembersihan          : **{n_final:,}**")

                if modul_summary is not None:
                    st.markdown("#### Ringkasan Kelayakan Modul (sebelum & sesudah threshold)")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Jumlah modul awal", f"{n_modul_awal:,}")
                    with c2:
                        st.metric("Modul layak (≥ threshold)", f"{n_modul_layak:,}")
                    with c3:
                        st.metric("Coverage tiket setelah pembersihan", f"{coverage_pct:,.2f} %")

                    st.dataframe(
                        modul_summary.sort_values("jumlah_awal", ascending=False)
                        .reset_index()
                        .rename(columns={"index": "modul"}),
                        use_container_width=True,
                    )

                st.markdown("#### Preview Data Hasil Pembersihan")
                st.dataframe(df_clean.head(10), use_container_width=True)

            except Exception as e:
                st.error(f"Gagal menyimpan ke database: {e}")
