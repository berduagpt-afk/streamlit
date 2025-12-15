# pages/statistik_dataset.py — Statistik Populasi 7 Kolom (tanpa filter)
# Fitur: KPI, Tren bulanan, Top-N (site/modul), Kualitas data,
#        Statistik jumlah kata (min/max/mean), Word frequency,
#        WordCloud, Top Bigram (2-kata), Treemap modul→sub_modul.

# (Opsional) Aktifkan hanya jika TIDAK diset di app.py:
# import streamlit as st
# st.set_page_config(page_title="Statistik Dataset", layout="wide")

import io
import re
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from collections import Counter
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# -------------------------------------------------------------------
# Guard login
# -------------------------------------------------------------------
if not st.session_state.get("logged_in", False):
    st.error("Silakan login dulu untuk mengakses halaman ini.")
    st.stop()

# -------------------------------------------------------------------
# Koneksi Postgres & Loader
# -------------------------------------------------------------------
def pg_engine():
    cfg = st.secrets["connections"]["postgres"]
    dialect = cfg.get("dialect", "postgresql")
    if "+" not in dialect:
        dialect = f"{dialect}+psycopg2"
    url = URL.create(
        drivername=dialect,
        username=cfg["username"],
        password=cfg["password"],
        host=cfg["host"],
        port=int(cfg["port"]),
        database=cfg["database"],
    )
    return create_engine(url, pool_pre_ping=True)

@st.cache_data(show_spinner=False)
def load_from_db(schema="lasis_djp", table="incident_raw") -> pd.DataFrame:
    eng = pg_engine()
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
    s = series.astype(str).str.strip().replace({"": None, "None": None, "nan": None, "NaN": None})
    parsed = pd.to_datetime(s, errors="coerce", format="%Y-%m-%d %H:%M:%S.%f")
    mask_fail = parsed.isna()
    if mask_fail.any():
        parsed.loc[mask_fail] = pd.to_datetime(s[mask_fail], errors="coerce", infer_datetime_format=True)
    return parsed

def load_data() -> pd.DataFrame:
    if st.session_state.get("df_7cols") is not None:
        df = st.session_state["df_7cols"].copy()
    else:
        df = load_from_db()
    # alias jika ada salah nama
    if "detailed_description" in df.columns and "detailed_decription" not in df.columns:
        df.rename(columns={"detailed_description": "detailed_decription"}, inplace=True)
    # parse tanggal robust
    df["tgl_submit"] = safe_parse_datetime(df["tgl_submit"])
    # normalisasi kategori & teks kunci
    for c in ["site", "assignee", "modul", "sub_modul", "incident_number", "detailed_decription"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().replace({"None": None, "nan": None, "NaN": None})
    return df

df = load_data()
if df.empty:
    st.warning("Dataset kosong.")
    st.stop()

st.title("📊 Statistik Populasi Dataset")

# -------------------------------------------------------------------
# 1) KPI Populasi
# -------------------------------------------------------------------
total_tiket = len(df)
tgl_null = df["tgl_submit"].isna().sum()
modul_null = df["modul"].isna().sum() + (df["modul"].astype(str).str.strip() == "").sum()
desc_null = df["detailed_decription"].isna().sum() + (df["detailed_decription"].astype(str).str.strip() == "").sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Tiket", f"{total_tiket:,}")
c2.metric("tgl_submit Null", f"{tgl_null:,}")
c3.metric("modul Null/Empty", f"{modul_null:,}")
c4.metric("detailed_decription Null/Empty", f"{desc_null:,}")
st.markdown("---")

# -------------------------------------------------------------------
# 2) Tren jumlah tiket per bulan
# -------------------------------------------------------------------
st.subheader("📈 Tren Jumlah Tiket per Bulan")
if df["tgl_submit"].notna().any():
    monthly = (
        df.dropna(subset=["tgl_submit"])
          .set_index("tgl_submit")
          .resample("MS")
          .size()
          .reset_index(name="count")
    )
    chart = alt.Chart(monthly).mark_line(point=True).encode(
        x=alt.X("tgl_submit:T", title="Bulan"),
        y=alt.Y("count:Q", title="Jumlah Tiket"),
        tooltip=[alt.Tooltip("yearmonth(tgl_submit):T", title="Bulan"), "count:Q"]
    ).properties(height=320)
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("Semua nilai `tgl_submit` kosong/invalid, tidak dapat menampilkan tren bulanan.")
st.markdown("---")

# -------------------------------------------------------------------
# 3) Distribusi Kategori (Top N) — site & modul
# -------------------------------------------------------------------
st.subheader("🏷️ Distribusi Kategori (Top N)")

def topn_bar(series: pd.Series, title: str, n=15):
    vc = (
        series.dropna().astype(str).str.strip()
        .replace("", np.nan).dropna()
        .value_counts().head(n).reset_index()
    )
    vc.columns = ["label", "count"]
    if vc.empty:
        st.info(f"Tidak ada data {title.lower()}."); return
    ch = alt.Chart(vc).mark_bar().encode(
        x=alt.X("count:Q", title="#Tiket"),
        y=alt.Y("label:N", sort="-x", title=title),
        tooltip=["label","count"]
    ).properties(height=max(200, min(500, 20*len(vc))))
    st.altair_chart(ch, use_container_width=True)

cc1, cc2 = st.columns(2)
with cc1: topn_bar(df["site"], "Site", n=15)
with cc2: topn_bar(df["modul"], "Modul", n=15)

st.markdown("---")

# -------------------------------------------------------------------
# 4) Kualitas Data (NA & Empty String) untuk setiap kolom
# -------------------------------------------------------------------
st.subheader("🧪 Kualitas Data per Kolom (NA & Empty)")

def column_quality(d: pd.DataFrame) -> pd.DataFrame:
    na_cnt = d.isna().sum().rename("NA")
    empty_cnt = d.apply(lambda s: s.astype(str).str.strip().eq("")).sum().rename("EmptyString")
    total = len(d)
    out = pd.concat([na_cnt, empty_cnt], axis=1)
    out["%NA"] = (out["NA"] / total * 100).round(2)
    out["%Empty"] = (out["EmptyString"] / total * 100).round(2)
    return out

qual = column_quality(df[["tgl_submit","incident_number","site","assignee","modul","sub_modul","detailed_decription"]])
st.dataframe(qual, use_container_width=True)
st.markdown("---")

# -------------------------------------------------------------------
# 5) Statistik detailed_decription — jumlah kata (min, max, mean)
# -------------------------------------------------------------------
st.subheader("📝 Statistik detailed_decription (jumlah kata)")

desc_series = df["detailed_decription"].fillna("").astype(str).str.strip()

def word_count(text: str) -> int:
    # hitung token alfanumerik sebagai kata
    return len(re.findall(r"\w+", text))

wc = desc_series.apply(word_count)
min_wc = int(wc.min()) if not wc.empty else 0
max_wc = int(wc.max()) if not wc.empty else 0
mean_wc = float(wc.mean()) if not wc.empty else 0.0

d1, d2, d3 = st.columns(3)
d1.metric("Kata Tersedikit", f"{min_wc:,}")
d2.metric("Kata Terbanyak", f"{max_wc:,}")
d3.metric("Rata-rata Kata/Tiket", f"{mean_wc:.2f}")

# mini-histogram distribusi jumlah kata
bins = wc.value_counts(bins=12, sort=False).reset_index()
bins.columns = ["range", "count"]
if not bins.empty:
    ch_wc = alt.Chart(bins).mark_bar().encode(
        x=alt.X("range:O", title="Jumlah Kata"),
        y=alt.Y("count:Q", title="#Tiket"),
        tooltip=["range","count"]
    ).properties(height=260)
    st.altair_chart(ch_wc, use_container_width=True)
st.markdown("---")

# -------------------------------------------------------------------
# 6) Word frequency & WordCloud
# -------------------------------------------------------------------
st.subheader("🔡 Word Frequency & WordCloud (detailed_decription)")

BASIC_STOPWORDS_ID = {
    "yang","dan","atau","di","ke","dari","dengan","untuk","pada","dalam","ada","tidak","jadi",
    "ini","itu","karena","sudah","hingga","agar","oleh","saat","seperti","mohon","tersebut",
    "kami","saya","bapak","ibu","dapat","akan","jika","sebagai","juga","atas","kpd","terima","kasih",
    "wp","npwp","djp"
}

def tokenize(text: str):
    t = re.sub(r"[^0-9a-zA-Záéíóúàèìòùâêîôûäëïöüñç\s]", " ", str(text).lower())
    toks = [w for w in t.split() if w and w not in BASIC_STOPWORDS_ID and not w.isdigit()]
    return toks

tokens = []
for s in desc_series:
    tokens.extend(tokenize(s))

# Top-N word frequency
N_TOP = 40
freq = Counter(tokens).most_common(N_TOP)
freq_df = pd.DataFrame(freq, columns=["word","count"])
if not freq_df.empty:
    ch_freq = alt.Chart(freq_df).mark_bar().encode(
        x=alt.X("count:Q", title="Frekuensi"),
        y=alt.Y("word:N", sort="-x", title="Kata"),
        tooltip=["word","count"]
    ).properties(height=max(300, min(600, 18*len(freq_df))))
    st.altair_chart(ch_freq, use_container_width=True)
else:
    st.info("Tidak ada token yang bisa ditampilkan (cek stopwords/cleaning).")

# WordCloud (opsional; butuh paket 'wordcloud')
try:
    from wordcloud import WordCloud
    joined = " ".join(tokens[:1_000_000])
    wc_img = WordCloud(width=1000, height=400, background_color="white").generate(joined)
    st.image(wc_img.to_array(), caption="WordCloud (detailed_decription)", use_column_width=True)
except Exception as e:
    st.info("Modul `wordcloud` belum terpasang. Install: `pip install wordcloud`")
    st.caption(f"(Detail: {e})")

# -------------------------------------------------------------------
# 6b) Top Bigram (2-kata paling sering)
# -------------------------------------------------------------------
st.subheader("🧩 Top Bigram (2-kata)")

def make_bigrams(tok_list, min_len=2):
    bigs = []
    for i in range(len(tok_list) - 1):
        a, b = tok_list[i], tok_list[i+1]
        if len(a) >= min_len and len(b) >= min_len:
            bigs.append(f"{a} {b}")
    return bigs

bigrams = make_bigrams(tokens)
big_freq = Counter(bigrams).most_common(30)
big_df = pd.DataFrame(big_freq, columns=["bigram","count"])
if not big_df.empty:
    ch_big = alt.Chart(big_df).mark_bar().encode(
        x=alt.X("count:Q", title="Frekuensi"),
        y=alt.Y("bigram:N", sort="-x", title="Bigram"),
        tooltip=["bigram","count"]
    ).properties(height=max(300, min(700, 18*len(big_df))))
    st.altair_chart(ch_big, use_container_width=True)
else:
    st.info("Belum ada bigram yang memenuhi kriteria.")

st.markdown("---")

# -------------------------------------------------------------------
# 7) Treemap modul → sub_modul
# -------------------------------------------------------------------
st.subheader("🧭 Treemap: Modul → Sub ModuI")

tree_df = (
    df[["modul","sub_modul"]]
    .fillna({"modul":"(Unknown modul)","sub_modul":"(Unknown sub_modul)"})
    .assign(count=1)
    .groupby(["modul","sub_modul"], as_index=False)["count"].sum()
)

if tree_df["count"].sum() == 0:
    st.info("Data modul/sub_modul kosong.")
else:
    try:
        import plotly.express as px
        fig = px.treemap(
            tree_df,
            path=["modul","sub_modul"],
            values="count",
            title=None
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        # Fallback Altair: grouped bar (sub_modul per modul)
        st.caption("Plotly tidak tersedia, menampilkan fallback Altair (grouped bar).")
        alt_df = tree_df.copy()
        ch_tree = alt.Chart(alt_df).mark_bar().encode(
            x=alt.X("count:Q", title="#Tiket"),
            y=alt.Y("sub_modul:N", sort="-x", title="Sub ModuI"),
            color=alt.Color("modul:N", title="Modul"),
            tooltip=["modul","sub_modul","count"]
        ).properties(height=500)
        st.altair_chart(ch_tree, use_container_width=True)

# -------------------------------------------------------------------
# Unduh populasi (tanpa filter)
# -------------------------------------------------------------------
st.markdown("---")
buf = io.BytesIO()
df.to_csv(buf, index=False)
st.download_button(
    "⬇️ Unduh CSV (populasi 7 kolom)",
    data=buf.getvalue(),
    file_name="dataset_populasi_7kolom.csv",
    mime="text/csv",
)
