# pages/data_preparation/text_processing/sintaksis_preprocessing.py
# Pemrosesan Text Sintaksis — Data Insiden DJP — Data Preparation (lanjutan)
# Source : lasis_djp.incident_normalized (kolom: text_norm, tgl_normalized)
# Output : lasis_djp.incident_sintaksis (kolom tambah: text_sintaksis, tokens_sintaksis_json, n_tokens_sintaksis, tgl_sintaksis)
#
# Fokus: tokenization + stopword removal (CUSTOM ONLY, tanpa Sastrawi)
# - Stopword & stop-phrase: pages/ref_sintaksis.py
# - Preview top tokens (sebelum & sesudah stopword) + sampling untuk performa
# - Progress bar batch
# - Output table di-REPLACE tiap run

from __future__ import annotations

import re
import json
from collections import Counter
from datetime import datetime, timezone, timedelta

import pandas as pd
import streamlit as st
import altair as alt
from sqlalchemy import create_engine

# ======================================================
# 🔐 Guard login
# ======================================================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login terlebih dahulu.")
    st.stop()

# ======================================================
# ⚙️ Konstanta DB
# ======================================================
SCHEMA = "lasis_djp"
SRC_TABLE = "incident_normalized"
OUT_TABLE = "incident_sintaksis"

TEXT_IN_COL = "text_norm"
TS_IN_COL = "tgl_normalized"

TEXT_OUT_COL = "text_sintaksis"
TOK_OUT_COL = "tokens_sintaksis_json"     # ✅ token disimpan sebagai JSON string
NTOK_OUT_COL = "n_tokens_sintaksis"       # ✅ jumlah token
TS_OUT_COL = "tgl_sintaksis"

# ======================================================
# 🔌 DB ENGINE (SQLAlchemy 2.x safe)
# ======================================================
@st.cache_resource(show_spinner=False)
def get_engine():
    cfg = st.secrets["connections"]["postgres"]
    url = (
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )
    return create_engine(url, pool_pre_ping=True, future=True)


@st.cache_data(show_spinner=False)
def load_source_from_db(schema: str = SCHEMA, table: str = SRC_TABLE) -> pd.DataFrame:
    eng = get_engine()
    with eng.connect() as conn:
        return pd.read_sql_table(table, con=conn, schema=schema)


def save_replace(df: pd.DataFrame, schema: str = SCHEMA, table: str = OUT_TABLE):
    eng = get_engine()
    df.to_sql(
        table,
        eng,
        schema=schema,
        if_exists="replace",
        index=False,
        chunksize=10_000,
        method="multi",
    )

# ======================================================
# 🧠 Custom Stopwords (ref_sintaksis.py) — WAJIB
# ======================================================
try:
    from pages.ref_sintaksis import (
        get_custom_stopwords,
        remove_custom_stopwords_tokens,
        KEEP_TERMS,
        remove_stop_phrases,
    )
    CUSTOM_STOPWORDS_SET = get_custom_stopwords()
    _CUSTOM_SW_OK = True
except Exception:
    _CUSTOM_SW_OK = False
    CUSTOM_STOPWORDS_SET = set()
    KEEP_TERMS = set()

    def remove_stop_phrases(text: str) -> str:
        return "" if text is None else str(text)

    def remove_custom_stopwords_tokens(tokens, *, stopwords=None, keep_terms=None):
        return tokens

# ======================================================
# 🔧 Tokenization
# ======================================================
_TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)

def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def _is_placeholder(t: str) -> bool:
    return t.startswith("__PH_") and t.endswith("__")


def process_sintaksis(
    text_norm: str,
    *,
    do_stopword: bool = True,
    min_token_len: int = 2,
    keep_placeholders: bool = True,
    use_stop_phrases: bool = True,
) -> tuple[str, list[str]]:
    """
    Return:
      - text_sintaksis (string siap TF-IDF)
      - tokens_after (token final setelah filtering/stopword)
    """
    if text_norm is None:
        text_norm = ""
    s = str(text_norm).strip()
    if not s:
        return "", []

    # hapus frasa template (opsional)
    if use_stop_phrases:
        s = remove_stop_phrases(s)

    toks = tokenize(s)

    # filter panjang token
    if min_token_len and min_token_len > 1:
        toks = [t for t in toks if len(t) >= min_token_len]

    # stopword removal custom
    if do_stopword:
        if keep_placeholders:
            ph = [t for t in toks if _is_placeholder(t)]
            non_ph = [t for t in toks if not _is_placeholder(t)]
            non_ph = remove_custom_stopwords_tokens(
                non_ph,
                stopwords=CUSTOM_STOPWORDS_SET,
                keep_terms=KEEP_TERMS,
            )
            toks = ph + non_ph
        else:
            toks = remove_custom_stopwords_tokens(
                toks,
                stopwords=CUSTOM_STOPWORDS_SET,
                keep_terms=KEEP_TERMS,
            )

    toks = [t for t in toks if t]
    return " ".join(toks).strip(), toks

# ======================================================
# 📌 Preview Top Token Helper
# ======================================================
def top_tokens_from_texts(
    texts: list[str],
    *,
    sample_n: int = 50_000,
    random_state: int = 42,
    min_token_len: int = 2,
    do_stopword: bool = True,
    keep_placeholders: bool = True,
    use_stop_phrases: bool = True,
    top_k: int = 30,
) -> pd.DataFrame:
    n = len(texts)
    if n == 0:
        return pd.DataFrame(columns=["token", "freq"])

    if sample_n and n > sample_n:
        s = pd.Series(texts).sample(int(sample_n), random_state=int(random_state))
        texts_use = s.tolist()
    else:
        texts_use = texts

    c = Counter()
    for t in texts_use:
        _, toks = process_sintaksis(
            t,
            do_stopword=do_stopword,
            min_token_len=min_token_len,
            keep_placeholders=keep_placeholders,
            use_stop_phrases=use_stop_phrases,
        )
        c.update([x.lower() for x in toks if x])

    top = c.most_common(int(top_k))
    return pd.DataFrame(top, columns=["token", "freq"])


def show_top_tokens(df_top: pd.DataFrame, title: str):
    st.markdown(f"#### {title}")
    if df_top.empty:
        st.info("Tidak ada token untuk ditampilkan.")
        return

    st.dataframe(df_top, use_container_width=True)

    chart = (
        alt.Chart(df_top)
        .mark_bar()
        .encode(
            x=alt.X("freq:Q", title="Frekuensi"),
            y=alt.Y("token:N", sort="-x", title="Token"),
            tooltip=["token", "freq"],
        )
    )
    st.altair_chart(chart, use_container_width=True)

# ======================================================
# 🧾 UI HEADER
# ======================================================
st.title("🧾 Sintaksis Preprocessing (TF-IDF) — Custom Stopwords")
st.caption(
    f"Sumber: `{SCHEMA}.{SRC_TABLE}` (kolom `{TEXT_IN_COL}`) → "
    f"Output: `{SCHEMA}.{OUT_TABLE}` "
    f"(kolom `{TEXT_OUT_COL}`, `{TOK_OUT_COL}`, `{NTOK_OUT_COL}`, `{TS_OUT_COL}`; replace tiap run)."
)

if not _CUSTOM_SW_OK:
    st.warning(
        "ref_sintaksis.py tidak ditemukan / gagal diimport. Stopword removal akan NO-OP."
    )

# ======================================================
# 📦 LOAD DATA
# ======================================================
with st.spinner(f"📦 Memuat data dari `{SCHEMA}.{SRC_TABLE}` ..."):
    try:
        df_src = load_source_from_db()
        st.success(f"Berhasil memuat {len(df_src):,} baris.")
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        st.stop()

if df_src.empty:
    st.warning("Dataset kosong.")
    st.stop()

if "incident_number" not in df_src.columns:
    st.error("Kolom wajib `incident_number` tidak ditemukan pada sumber.")
    st.stop()

if TEXT_IN_COL not in df_src.columns:
    st.error(f"Kolom wajib `{TEXT_IN_COL}` tidak ditemukan pada sumber.")
    st.stop()

# ======================================================
# 🧭 SIDEBAR
# ======================================================
with st.sidebar:
    st.header("⚙️ Pengaturan Sintaksis (Custom)")

    with st.form("syn_form"):
        st.markdown("### Tokenization & Filtering")
        min_token_len = st.number_input("Panjang minimal token", min_value=1, max_value=10, value=2, step=1)
        keep_placeholders = st.checkbox("Pertahankan placeholder __PH_*__", value=True)

        st.markdown("---")
        st.markdown("### Stopword Removal")
        do_stopword = st.checkbox("Aktifkan stopword removal (custom)", value=True, disabled=not _CUSTOM_SW_OK)
        use_stop_phrases = st.checkbox("Aktifkan stop-phrase (frasa template)", value=True, disabled=not _CUSTOM_SW_OK)

        st.markdown("---")
        st.markdown("### Preview Top Tokens")
        enable_preview = st.checkbox("Aktifkan preview top token", value=True)
        top_k = st.number_input("Top-K token", min_value=10, max_value=200, value=30, step=10)
        sample_n = st.number_input("Sample max baris (untuk preview)", min_value=5_000, max_value=200_000, value=50_000, step=5_000)
        preview_random_state = st.number_input("Random state sample", min_value=0, max_value=9999, value=42, step=1)

        st.markdown("---")
        run = st.form_submit_button("🚀 Jalankan Sintaksis Preprocessing", use_container_width=True)

# ======================================================
# 👀 PREVIEW BEFORE
# ======================================================
st.subheader("Preview Data (Sebelum)")
st.dataframe(df_src[["incident_number", TEXT_IN_COL, TS_IN_COL]].head(10), use_container_width=True)

texts_all = df_src[TEXT_IN_COL].fillna("").astype(str).tolist()

# ======================================================
# 🔎 PREVIEW TOP TOKENS (BEFORE RUN)
# ======================================================
if enable_preview:
    st.subheader("🔎 Preview Top Tokens (Sample)")
    st.caption(
        "Preview menggunakan sampling agar cepat. "
        "Sebelum stopword = tanpa stopword/stop-phrase. "
        "Sesudah stopword = mengikuti opsi stopword custom + stop-phrase."
    )

    with st.spinner("Menghitung top tokens (sebelum stopword) ..."):
        df_top_before = top_tokens_from_texts(
            texts_all,
            sample_n=int(sample_n),
            random_state=int(preview_random_state),
            min_token_len=int(min_token_len),
            do_stopword=False,
            keep_placeholders=keep_placeholders,
            use_stop_phrases=False,
            top_k=int(top_k),
        )
    show_top_tokens(df_top_before, "Top Tokens — Sebelum Stopword (OFF)")

    with st.spinner("Menghitung top tokens (sesudah stopword custom) ..."):
        df_top_after = top_tokens_from_texts(
            texts_all,
            sample_n=int(sample_n),
            random_state=int(preview_random_state),
            min_token_len=int(min_token_len),
            do_stopword=bool(do_stopword) and _CUSTOM_SW_OK,
            keep_placeholders=keep_placeholders,
            use_stop_phrases=bool(use_stop_phrases) and _CUSTOM_SW_OK,
            top_k=int(top_k),
        )
    show_top_tokens(df_top_after, "Top Tokens — Sesudah Stopword (CUSTOM)")

if not run:
    st.info("Atur parameter di sidebar lalu klik **Jalankan Sintaksis Preprocessing**.")
    st.stop()

# ======================================================
# 🧮 PROCESS WITH PROGRESS BAR
# ======================================================
WIB = timezone(timedelta(hours=7))
run_dt = datetime.now(WIB).replace(tzinfo=None)

n = len(texts_all)
progress = st.progress(0, text="Menyiapkan proses sintaksis...")
status = st.empty()
status.info("⚙️ Memproses tokenization & stopword removal (custom)...")

out_text = [""] * n
out_tokens_json = ["[]"] * n
out_ntokens = [0] * n

update_every = max(200, n // 200)

for i, t in enumerate(texts_all):
    txt, toks = process_sintaksis(
        t,
        do_stopword=bool(do_stopword) and _CUSTOM_SW_OK,
        min_token_len=int(min_token_len),
        keep_placeholders=keep_placeholders,
        use_stop_phrases=bool(use_stop_phrases) and _CUSTOM_SW_OK,
    )
    out_text[i] = txt
    out_tokens_json[i] = json.dumps(toks, ensure_ascii=False)
    out_ntokens[i] = int(len(toks))

    if (i + 1) % update_every == 0 or (i + 1) == n:
        p = int(((i + 1) / n) * 100)
        progress.progress((i + 1) / n, text=f"Pemrosesan berjalan... {p}% ({i+1:,}/{n:,})")

status.success("✅ Sintaksis preprocessing selesai.")

# ======================================================
# 🧩 BUILD OUTPUT TABLE
# ======================================================
df_out = df_src.copy()
df_out[TEXT_OUT_COL] = out_text
df_out[TOK_OUT_COL] = out_tokens_json
df_out[NTOK_OUT_COL] = out_ntokens
df_out[TS_OUT_COL] = pd.Timestamp(run_dt)

# ======================================================
# 💾 SAVE OUTPUT (REPLACE)
# ======================================================
try:
    with st.spinner(f"💾 Menyimpan ke `{SCHEMA}.{OUT_TABLE}` (replace) ..."):
        save_replace(df_out, schema=SCHEMA, table=OUT_TABLE)
    st.success(f"Berhasil menyimpan ke `{SCHEMA}.{OUT_TABLE}`.")
except Exception as e:
    st.error(f"Gagal menyimpan ke database: {e}")
    st.stop()

# ======================================================
# 📊 PREVIEW AFTER + STATS
# ======================================================
st.subheader("Preview Hasil (Sesudah)")
show_cols = ["incident_number", TEXT_IN_COL, TEXT_OUT_COL, NTOK_OUT_COL, TS_OUT_COL]
st.dataframe(df_out[show_cols].head(10), use_container_width=True)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Jumlah baris", f"{df_out.shape[0]:,}")
c2.metric("Stopword custom", "ON" if (do_stopword and _CUSTOM_SW_OK) else "OFF")
c3.metric("Stop-phrase", "ON" if (use_stop_phrases and _CUSTOM_SW_OK) else "OFF")
c4.metric("Placeholder", "KEEP" if keep_placeholders else "DROP")
c5.metric("Rata-rata token", f"{(df_out[NTOK_OUT_COL].mean() if len(df_out) else 0):.2f}")

st.markdown("### Statistik Panjang Teks Sintaksis (karakter)")
length_series = df_out[TEXT_OUT_COL].astype(str).str.len()
st.write(length_series.describe()[["min", "mean", "max"]])

st.markdown("### Statistik Jumlah Token per Tiket")
tok_series = df_out[NTOK_OUT_COL]
st.write(tok_series.describe()[["min", "mean", "max"]])

st.markdown("#### Histogram Distribusi Jumlah Token")
tok_df = pd.DataFrame({"n_tokens": tok_series})
hist_tok = alt.Chart(tok_df).mark_bar().encode(
    x=alt.X("n_tokens:Q", bin=alt.Bin(maxbins=60), title="Jumlah Token"),
    y=alt.Y("count()", title="Jumlah Tiket"),
)
st.altair_chart(hist_tok, use_container_width=True)

# ======================================================
# 📥 DOWNLOAD CSV (ringkas)
# ======================================================
csv_bytes = df_out[["incident_number", TEXT_OUT_COL, TOK_OUT_COL, NTOK_OUT_COL, TS_OUT_COL]].to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Output Sintaksis (CSV)",
    data=csv_bytes,
    file_name="incident_sintaksis.csv",
    mime="text/csv",
)
