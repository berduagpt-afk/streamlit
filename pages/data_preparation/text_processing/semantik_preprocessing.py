# pages/data_preparation/text_processing/semantik_preprocessing.py
# ======================================================
# Pemrosesan Text Semantik — Data Insiden DJP — Data Preparation
# Source  : lasis_djp.incident_normalized (kolom: text_norm, tgl_normalized)
# Output  : lasis_djp.incident_semantik
#
# Fokus:
# - Context-preserving preprocessing (untuk IndoBERT embedding + HDBSCAN)
# - Normalisasi ringan + (opsional) hapus stop-phrase template
# - Placeholder dibuat BERT-friendly (mis: __PH_LONGNUM__ -> ph_longnum)
# - Truncation strategy untuk mengendalikan panjang teks (head / tail / head_tail)
# - Token disimpan opsional untuk quality check (bukan untuk TF-IDF)
#
# Catatan:
# - Stopword removal default OFF (karena transformer butuh konteks)
# - Output bisa replace (default) atau append (untuk reproducibility tesis)

from __future__ import annotations

import re
import json
import uuid
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
OUT_TABLE = "incident_semantik"

TEXT_IN_COL = "text_norm"
TS_IN_COL = "tgl_normalized"

RUN_ID_COL = "semantik_run_id"
TEXT_OUT_COL = "text_semantic"
TOK_OUT_COL = "tokens_semantic_json"       # opsional (QC)
NTOK_OUT_COL = "n_tokens_semantic"
NCHAR_OUT_COL = "n_chars_semantic"
TS_OUT_COL = "tgl_semantik"


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


def save_df(df: pd.DataFrame, *, schema: str = SCHEMA, table: str = OUT_TABLE, mode: str = "replace"):
    """
    mode:
      - "replace": drop & recreate table
      - "append" : append rows (disarankan bila ingin histori run di tesis)
    """
    eng = get_engine()
    if_exists = "replace" if mode == "replace" else "append"
    df.to_sql(
        table,
        eng,
        schema=schema,
        if_exists=if_exists,
        index=False,
        chunksize=10_000,
        method="multi",
    )


# ======================================================
# 🧠 Optional Stop-phrases (gunakan bila ada)
# ======================================================
# Untuk semantik, stop-phrase hanya untuk template pembuka/penutup, bukan konten masalah.
# Jika Anda sudah punya fungsi di ref_sintaksis, boleh dipakai ulang.
try:
    from pages.ref_sintaksis import remove_stop_phrases  # type: ignore
    _STOPPH_OK = True
except Exception:
    _STOPPH_OK = False

    def remove_stop_phrases(text: str) -> str:
        return "" if text is None else str(text)


# ======================================================
# 🔧 Helper: placeholder conversion + cleanup ringan + truncation
# ======================================================

# placeholder lama: __PH_LONGNUM__  -> ph_longnum
_PH_RE = re.compile(r"__PH_([A-Z0-9_]+)__")

# tokenization ringan untuk QC (bukan untuk representasi)
_TOKEN_RE = re.compile(r"\b[\w]+\b", flags=re.UNICODE)


def _to_ph_bert_friendly(text: str) -> str:
    def _repl(m: re.Match) -> str:
        key = m.group(1).lower()
        key = re.sub(r"_+", "_", key).strip("_")
        return f"ph_{key}" if key else "ph"
    return _PH_RE.sub(_repl, text)


def _cleanup_whitespace(text: str) -> str:
    # jaga konteks, hanya rapikan spasi
    s = text.replace("\u00a0", " ")
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _truncate_text(text: str, *, max_chars: int, mode: str) -> str:
    """
    Truncation berbasis karakter (lebih stabil di UI; embedding nanti tetap dapat limit token).
    mode:
      - "none"
      - "head"
      - "tail"
      - "head_tail"
    """
    if max_chars is None or max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text

    if mode == "head":
        return text[:max_chars].rstrip()
    if mode == "tail":
        return text[-max_chars:].lstrip()
    if mode == "head_tail":
        # ambil awal dan akhir, sisipkan separator
        sep = " … "
        keep = max_chars - len(sep)
        if keep <= 10:
            return text[:max_chars].rstrip()
        head_len = keep // 2
        tail_len = keep - head_len
        return (text[:head_len].rstrip() + sep + text[-tail_len:].lstrip()).strip()
    return text  # "none" atau fallback


def tokenize_qc(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def process_semantik(
    text_norm: str,
    *,
    use_stop_phrases: bool = True,
    convert_placeholders: bool = True,
    trunc_mode: str = "head_tail",
    max_chars: int = 2000,
    keep_tokens_json: bool = True,
) -> tuple[str, list[str]]:
    """
    Return:
      - text_semantic: string yang tetap natural (context-preserving)
      - tokens_qc: token ringan untuk quality check (opsional disimpan)
    """
    if text_norm is None:
        text_norm = ""
    s = str(text_norm)

    # optional: buang frasa template (aman)
    if use_stop_phrases and _STOPPH_OK:
        s = remove_stop_phrases(s)

    # placeholder jadi bert-friendly
    if convert_placeholders:
        s = _to_ph_bert_friendly(s)

    # rapikan whitespace
    s = _cleanup_whitespace(s)

    # truncation (agar stabil untuk embedding batching)
    if trunc_mode and trunc_mode != "none":
        s = _truncate_text(s, max_chars=int(max_chars), mode=str(trunc_mode))

    # token QC
    toks = tokenize_qc(s) if keep_tokens_json else []
    return s, toks


# ======================================================
# 📌 Preview Top Terms (QC)
# ======================================================
def top_terms_qc(
    texts: list[str],
    *,
    sample_n: int = 50_000,
    random_state: int = 42,
    use_stop_phrases: bool = True,
    convert_placeholders: bool = True,
    trunc_mode: str = "head_tail",
    max_chars: int = 2000,
    top_k: int = 30,
) -> pd.DataFrame:
    n = len(texts)
    if n == 0:
        return pd.DataFrame(columns=["term", "freq"])

    if sample_n and n > sample_n:
        s = pd.Series(texts).sample(int(sample_n), random_state=int(random_state))
        texts_use = s.tolist()
    else:
        texts_use = texts

    c = Counter()
    for t in texts_use:
        s2, toks = process_semantik(
            t,
            use_stop_phrases=use_stop_phrases,
            convert_placeholders=convert_placeholders,
            trunc_mode=trunc_mode,
            max_chars=int(max_chars),
            keep_tokens_json=True,
        )
        # buang token super-pendek hanya untuk statistik QC
        c.update([x for x in toks if x and len(x) >= 2])

    top = c.most_common(int(top_k))
    return pd.DataFrame(top, columns=["term", "freq"])


def show_top_terms(df_top: pd.DataFrame, title: str):
    st.markdown(f"#### {title}")
    if df_top.empty:
        st.info("Tidak ada term untuk ditampilkan.")
        return

    st.dataframe(df_top, use_container_width=True)

    chart = (
        alt.Chart(df_top)
        .mark_bar()
        .encode(
            x=alt.X("freq:Q", title="Frekuensi"),
            y=alt.Y("term:N", sort="-x", title="Term"),
            tooltip=["term", "freq"],
        )
    )
    st.altair_chart(chart, use_container_width=True)


# ======================================================
# 🧾 UI HEADER
# ======================================================
st.title("🧾 Semantik Preprocessing (IndoBERT) — Context Preserving")
st.caption(
    f"Sumber: `{SCHEMA}.{SRC_TABLE}` (kolom `{TEXT_IN_COL}`) → "
    f"Output: `{SCHEMA}.{OUT_TABLE}` "
    f"(kolom `{RUN_ID_COL}`, `{TEXT_OUT_COL}`, `{TOK_OUT_COL}`, `{NTOK_OUT_COL}`, `{NCHAR_OUT_COL}`, `{TS_OUT_COL}`)."
)

if not _STOPPH_OK:
    st.info("Opsional: `remove_stop_phrases` tidak ditemukan. Fitur stop-phrase akan nonaktif.")


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

for col in ["incident_number", TEXT_IN_COL, TS_IN_COL]:
    if col not in df_src.columns:
        st.error(f"Kolom wajib `{col}` tidak ditemukan pada sumber.")
        st.stop()


# ======================================================
# 🧭 SIDEBAR
# ======================================================
with st.sidebar:
    st.header("⚙️ Pengaturan Semantik")

    with st.form("sem_form"):
        st.markdown("### Normalisasi")
        convert_placeholders = st.checkbox("Ubah placeholder menjadi BERT-friendly (ph_*)", value=True)
        use_stop_phrases = st.checkbox(
            "Hapus stop-phrase template (jika tersedia)",
            value=True,
            disabled=not _STOPPH_OK,
        )

        st.markdown("---")
        st.markdown("### Truncation (untuk stabilitas embedding)")
        trunc_mode = st.selectbox(
            "Mode truncation",
            options=["none", "head", "tail", "head_tail"],
            index=3,
            help="Gunakan head_tail agar konteks awal & akhir tetap terbawa.",
        )
        max_chars = st.number_input(
            "Max karakter text_semantic",
            min_value=200,
            max_value=20_000,
            value=2000,
            step=100,
            help="Batas berbasis karakter. Embedding IndoBERT tetap punya batas token; ini untuk kontrol awal.",
        )

        st.markdown("---")
        st.markdown("### Output & Quality Check")
        keep_tokens_json = st.checkbox("Simpan tokens_semantic_json (untuk QC)", value=True)
        enable_preview = st.checkbox("Aktifkan preview top terms (QC)", value=True)
        top_k = st.number_input("Top-K terms", min_value=10, max_value=200, value=30, step=10)
        sample_n = st.number_input("Sample max baris (preview)", min_value=5_000, max_value=200_000, value=50_000, step=5_000)
        preview_random_state = st.number_input("Random state sample", min_value=0, max_value=9999, value=42, step=1)

        st.markdown("---")
        st.markdown("### Mode Penyimpanan")
        save_mode = st.selectbox(
            "Simpan ke tabel output",
            options=["replace", "append"],
            index=0,
            help="replace: overwrite tabel (default). append: simpan histori run (disarankan untuk tesis).",
        )

        st.markdown("---")
        run = st.form_submit_button("🚀 Jalankan Semantik Preprocessing", use_container_width=True)


# ======================================================
# 👀 PREVIEW BEFORE
# ======================================================
st.subheader("Preview Data (Sebelum)")
st.dataframe(df_src[["incident_number", TEXT_IN_COL, TS_IN_COL]].head(10), use_container_width=True)

texts_all = df_src[TEXT_IN_COL].fillna("").astype(str).tolist()


# ======================================================
# 🔎 PREVIEW TOP TERMS (QC)
# ======================================================
if enable_preview:
    st.subheader("🔎 Preview Top Terms (Quality Check — Sample)")
    st.caption(
        "Preview untuk mengecek kualitas normalisasi (bukan untuk TF-IDF). "
        "Menggunakan sampling agar cepat."
    )

    with st.spinner("Menghitung top terms (QC) ..."):
        df_top = top_terms_qc(
            texts_all,
            sample_n=int(sample_n),
            random_state=int(preview_random_state),
            use_stop_phrases=bool(use_stop_phrases) and _STOPPH_OK,
            convert_placeholders=bool(convert_placeholders),
            trunc_mode=str(trunc_mode),
            max_chars=int(max_chars),
            top_k=int(top_k),
        )
    show_top_terms(df_top, "Top Terms — Setelah Preprocessing (QC)")

if not run:
    st.info("Atur parameter di sidebar lalu klik **Jalankan Semantik Preprocessing**.")
    st.stop()


# ======================================================
# 🧮 PROCESS WITH PROGRESS BAR
# ======================================================
WIB = timezone(timedelta(hours=7))
run_dt = datetime.now(WIB).replace(tzinfo=None)
run_id = str(uuid.uuid4())

n = len(texts_all)
progress = st.progress(0, text="Menyiapkan proses semantik...")
status = st.empty()
status.info("⚙️ Memproses normalisasi semantik (context-preserving)...")

out_run_id = [run_id] * n
out_text = [""] * n
out_tokens_json = ["[]"] * n
out_ntokens = [0] * n
out_nchars = [0] * n

update_every = max(200, n // 200)

for i, t in enumerate(texts_all):
    txt, toks = process_semantik(
        t,
        use_stop_phrases=bool(use_stop_phrases) and _STOPPH_OK,
        convert_placeholders=bool(convert_placeholders),
        trunc_mode=str(trunc_mode),
        max_chars=int(max_chars),
        keep_tokens_json=bool(keep_tokens_json),
    )
    out_text[i] = txt
    out_nchars[i] = int(len(txt))
    out_ntokens[i] = int(len(toks)) if keep_tokens_json else int(len(tokenize_qc(txt)))

    if keep_tokens_json:
        out_tokens_json[i] = json.dumps(toks, ensure_ascii=False)
    else:
        out_tokens_json[i] = "[]"

    if (i + 1) % update_every == 0 or (i + 1) == n:
        p = int(((i + 1) / n) * 100)
        progress.progress((i + 1) / n, text=f"Pemrosesan berjalan... {p}% ({i+1:,}/{n:,})")

status.success(f"✅ Semantik preprocessing selesai. run_id={run_id}")


# ======================================================
# 🧩 BUILD OUTPUT TABLE
# ======================================================
df_out = df_src.copy()
df_out[RUN_ID_COL] = out_run_id
df_out[TEXT_OUT_COL] = out_text
df_out[TOK_OUT_COL] = out_tokens_json
df_out[NTOK_OUT_COL] = out_ntokens
df_out[NCHAR_OUT_COL] = out_nchars
df_out[TS_OUT_COL] = pd.Timestamp(run_dt)


# ======================================================
# 💾 SAVE OUTPUT
# ======================================================
try:
    with st.spinner(f"💾 Menyimpan ke `{SCHEMA}.{OUT_TABLE}` ({save_mode}) ..."):
        save_df(df_out, schema=SCHEMA, table=OUT_TABLE, mode=str(save_mode))
    st.success(f"Berhasil menyimpan ke `{SCHEMA}.{OUT_TABLE}` (mode={save_mode}).")
except Exception as e:
    st.error(f"Gagal menyimpan ke database: {e}")
    st.stop()


# ======================================================
# 📊 PREVIEW AFTER + STATS
# ======================================================
st.subheader("Preview Hasil (Sesudah)")
show_cols = ["incident_number", TEXT_IN_COL, TEXT_OUT_COL, NCHAR_OUT_COL, NTOK_OUT_COL, TS_OUT_COL, RUN_ID_COL]
st.dataframe(df_out[show_cols].head(10), use_container_width=True)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Jumlah baris", f"{df_out.shape[0]:,}")
c2.metric("Stop-phrase", "ON" if (use_stop_phrases and _STOPPH_OK) else "OFF")
c3.metric("Placeholder", "BERT-friendly" if convert_placeholders else "AS-IS")
c4.metric("Truncation", trunc_mode.upper())
c5.metric("Rata-rata token", f"{(df_out[NTOK_OUT_COL].mean() if len(df_out) else 0):.2f}")

st.markdown("### Statistik Panjang Teks Semantik (karakter)")
length_series = df_out[TEXT_OUT_COL].astype(str).str.len()
st.write(length_series.describe()[["min", "mean", "max"]])

st.markdown("### Statistik Jumlah Token per Tiket (QC)")
tok_series = df_out[NTOK_OUT_COL]
st.write(tok_series.describe()[["min", "mean", "max"]])

st.markdown("#### Histogram Distribusi Jumlah Token (QC)")
tok_df = pd.DataFrame({"n_tokens": tok_series})
hist_tok = alt.Chart(tok_df).mark_bar().encode(
    x=alt.X("n_tokens:Q", bin=alt.Bin(maxbins=60), title="Jumlah Token (QC)"),
    y=alt.Y("count()", title="Jumlah Tiket"),
)
st.altair_chart(hist_tok, use_container_width=True)

st.markdown("#### Histogram Distribusi Panjang Karakter")
len_df = pd.DataFrame({"n_chars": df_out[NCHAR_OUT_COL]})
hist_len = alt.Chart(len_df).mark_bar().encode(
    x=alt.X("n_chars:Q", bin=alt.Bin(maxbins=60), title="Panjang Karakter"),
    y=alt.Y("count()", title="Jumlah Tiket"),
)
st.altair_chart(hist_len, use_container_width=True)


# ======================================================
# 📥 DOWNLOAD CSV (ringkas)
# ======================================================
csv_cols = ["incident_number", RUN_ID_COL, TEXT_OUT_COL, NTOK_OUT_COL, NCHAR_OUT_COL, TS_OUT_COL]
csv_bytes = df_out[csv_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Output Semantik (CSV Ringkas)",
    data=csv_bytes,
    file_name="incident_semantik.csv",
    mime="text/csv",
)
