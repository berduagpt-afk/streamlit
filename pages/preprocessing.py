# preprocessing.py
# Halaman Preprocessing end-to-end untuk prototipe insiden berulang

import re
import unicodedata
from datetime import datetime

import pandas as pd
import streamlit as st
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ===================== Optional deps =====================
# ftfy: memperbaiki mojibake umum ('Ã©' -> 'é'), aman jika tidak ada
try:
    from ftfy import fix_text as _ftfy_fix
    _FTFY_AVAILABLE = True
except Exception:
    _FTFY_AVAILABLE = False

    def _ftfy_fix(x):
        return x

# stanza: lemmatizer Bahasa Indonesia (butuh stanza.download('id') saat setup)
try:
    import stanza

    _STANZA_OK = True

    @st.cache_resource(show_spinner=False)
    def get_stanza_id():
        # NOTE: jalankan stanza.download('id') di environment sebelum pakai pertama kali
        return stanza.Pipeline(
            lang="id",
            processors="tokenize,pos,lemma",
            tokenize_no_ssplit=True,
            use_gpu=False,
        )

    def lemmatize_with_stanza(tokens):
        nlp = get_stanza_id()
        doc = nlp(" ".join(tokens))
        return [w.lemma for s in doc.sentences for w in s.words if w.lemma]
except Exception:
    _STANZA_OK = False

    def lemmatize_with_stanza(tokens):
        return tokens  # fallback no-op

# redaction_config: pola regex (URL, NPWP, DOCNO, NAMA_WP, dll) di file terpisah
try:
    from redaction_config import (
        redact_text,
        get_default_enabled,
        list_pattern_names,
    )

    _REDACTION_AVAILABLE = True
except Exception:
    _REDACTION_AVAILABLE = False

    def redact_text(text, policy="placeholder", enabled=None):
        # fallback: tidak melakukan redaksi apa pun
        return text, []

    def get_default_enabled(include_special=True):
        return {}

    def list_pattern_names(include_special=True):
        return []


st.title("🛠️ Preprocessing Data")
st.caption(
    "Bersihkan deskripsi tiket: perbaiki mojibake → redaksi non-informatif → "
    "cleaning → stopwords/stemming → tokenisasi/lemmatization → (opsional) ekstraksi timestamp."
)

# ===================== Ambil dataset dari session =====================
df_input = st.session_state.get("df_raw", None)
if df_input is None:
    df_input = st.session_state.get("dataset", None)  # kompatibel dgn versi lama

if df_input is None:
    st.warning("⚠️ Dataset belum tersedia. Silakan upload dataset di menu **Upload Dataset**.")
    st.stop()
if not isinstance(df_input, pd.DataFrame) or df_input.empty:
    st.error("Dataset tidak valid atau kosong.")
    st.stop()

df = df_input.copy()

if "isi" not in df.columns:
    st.error("Kolom `isi` (deskripsi teks) tidak ditemukan di dataset.")
    st.stop()

# ===================== Resources (cache) =====================
@st.cache_resource(show_spinner=False)
def get_nlp_tools():
    stop_factory = StopWordRemoverFactory()
    stem_factory = StemmerFactory()
    return stop_factory.create_stop_word_remover(), stem_factory.create_stemmer()


stop_remover, stemmer = get_nlp_tools()


@st.cache_resource(show_spinner=False)
def get_stopword_set():
    return set(StopWordRemoverFactory().get_stop_words())


STOPWORDS_SET = get_stopword_set()

# ===================== Mojibake / weird-char cleaning =====================
def strip_weird_chars(
    s: str,
    force_ascii: bool = False,
    whitelist_pattern: str = r"[^A-Za-z0-9\s\.,;:\-_/@#%&\(\)\[\]\{\}\+=\"'’“”‘?!]",
) -> str:
    """
    - Normalisasi unicode (NFKC)
    - Hilangkan zero-width & karakter kontrol
    - Jika force_ascii=True -> ASCII only (hapus non-ASCII)
    - Jika False -> whitelist: karakter di luar pattern diganti spasi
    """
    if not isinstance(s, str):
        return ""
    t = unicodedata.normalize("NFKC", s)
    # zero-width & BOM
    t = re.sub(r"[\u200B-\u200D\uFEFF]", "", t)
    # kontrol C0/C1
    t = re.sub(r"[\x00-\x1F\x7F-\x9F]", " ", t)
    if force_ascii:
        t = t.encode("ascii", "ignore").decode("ascii", "ignore")
    else:
        t = re.sub(whitelist_pattern, " ", t)
    # rapikan spasi
    t = re.sub(r"\s+", " ", t).strip()
    return t


def fix_then_strip(s: str, use_ftfy: bool, force_ascii: bool) -> str:
    if not isinstance(s, str):
        return ""
    t = _ftfy_fix(s) if (use_ftfy and _FTFY_AVAILABLE) else s
    return strip_weird_chars(t, force_ascii=force_ascii)


# ===================== Cleaning dasar =====================
def clean_text_basic(
    txt: str,
    to_lower: bool = True,
    remove_punct: bool = True,
    remove_digits: bool = False,
    strip_ws: bool = True,
) -> str:
    if not isinstance(txt, str):
        return ""
    s = re.sub(r"<[^>]+>", " ", txt)  # HTML
    if to_lower:
        s = s.lower()
    if remove_punct:
        s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    if remove_digits:
        s = re.sub(r"\d+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip() if strip_ws else s


# ===================== Preprocess series (text-level) =====================
@st.cache_data(show_spinner=False)
def preprocess_series(
    texts: pd.Series,
    use_ftfy: bool,
    force_ascii: bool,
    use_stopwords_text: bool,
    use_stemming_text: bool,
    to_lower: bool,
    remove_punct: bool,
    remove_digits: bool,
    redact: bool,
    redact_policy: str,
    redact_enabled: dict | None,
) -> pd.Series:
    # 0) perbaiki mojibake & karakter aneh
    cleaned = texts.fillna("").astype(str).apply(
        lambda x: fix_then_strip(x, use_ftfy=use_ftfy, force_ascii=force_ascii)
    )

    # 1) redaksi non-informatif (sebelum punctuation removal)
    if redact:
        def _apply_redact(x):
            x2, _hits = redact_text(x, policy=redact_policy, enabled=redact_enabled)
            return x2

        cleaned = cleaned.apply(_apply_redact)

    # 2) basic cleaning
    cleaned = cleaned.apply(
        lambda x: clean_text_basic(
            x,
            to_lower=to_lower,
            remove_punct=remove_punct,
            remove_digits=remove_digits,
            strip_ws=True,
        )
    )

    # 3) stopword removal (text-level)
    if use_stopwords_text:
        cleaned = cleaned.apply(lambda x: stop_remover.remove(x))

    # 4) stemming (text-level)
    if use_stemming_text:
        cleaned = cleaned.apply(lambda x: stemmer.stem(x))

    return cleaned


# ===================== Tokenisasi & Lemmatization =====================
def tokenize_text(s: str, method: str = "regex"):
    if not isinstance(s, str) or not s:
        return []
    if method == "whitespace":
        return [t for t in s.split() if t]
    # default: regex (aman untuk Indo)
    return re.findall(r"\b\w+\b", s, flags=re.UNICODE)


# ===================== Timestamp extraction =====================
def extract_timestamp_from_id_bugtrack(series: pd.Series) -> pd.Series:
    """
    id_bugtrack contoh: 20230102074234Mon -> 2023-01-02 07:42:34 (ambil 14 digit pertama).
    """
    return pd.to_datetime(
        series.astype(str).str.slice(0, 14),
        format="%Y%m%d%H%M%S",
        errors="coerce",
    )


# ===================== SIDEBAR (Form) =====================
with st.sidebar:
    st.header("⚙️ Pengaturan Preprocessing")

    with st.form("prep_form", clear_on_submit=False):
        st.markdown("**Normalisasi & Cleaning**")
        to_lower = st.checkbox("Lowercase", value=True)
        remove_punct = st.checkbox("Hapus tanda baca", value=True)
        remove_digits = st.checkbox("Hapus angka", value=False)

        st.markdown("**Bahasa (Sastrawi, text-level)**")
        use_stopwords_text = st.checkbox("Stopword removal (text-level)", value=True)
        use_stemming_text = st.checkbox("Stemming (text-level, Sastrawi)", value=False)

        st.markdown("**Mojibake & ASCII**")
        use_ftfy = st.checkbox(
            "Perbaiki mojibake (ftfy)", value=_FTFY_AVAILABLE, help="Aktif jika ftfy terpasang."
        )
        force_ascii = st.checkbox("Paksa ASCII only", value=False)

        st.markdown("**Redaksi non-informatif**")
        if _REDACTION_AVAILABLE:
            redact = st.checkbox("Aktifkan redaksi", value=True)
            redact_policy = st.selectbox(
                "Kebijakan redaksi", ["placeholder", "remove"], index=0
            )
            names = list_pattern_names(include_special=True)
            default_enabled = get_default_enabled(include_special=True)
            selected = st.multiselect(
                "Aktifkan pola redaksi",
                options=names,
                default=[n for n in names if default_enabled.get(n, True)],
                help="Contoh: URL, EMAIL, NPWP, DOCNO, NAMA_WP, dll.",
            )
            redact_enabled = {n: (n in selected) for n in names}
        else:
            redact = False
            redact_policy = "placeholder"
            redact_enabled = {}
            st.info(
                "File `redaction_config.py` tidak ditemukan. Redaksi dinonaktifkan (no-op)."
            )

        st.markdown("**Tokenisasi & Lanjutan**")
        tok_method = st.selectbox("Metode tokenisasi", ["regex", "whitespace"], index=0)
        token_stop = st.checkbox("Stopword removal (token-level)", value=True)
        lemma_choice = st.selectbox(
            "Lemmatization",
            ["Tidak ada", "Stemming (Sastrawi)", "Lemmatization (Stanza)"],
            index=1,
            help="Untuk BERT, biasanya pilih 'Tidak ada'. Stanza butuh model 'id'.",
        )

        st.markdown("**Timestamp (opsional)**")
        has_id_bt = "id_bugtrack" in df.columns
        want_ts = st.checkbox(
            "Buat kolom `timestamp` dari `id_bugtrack`",
            value=("timestamp" not in df.columns) if has_id_bt else False,
            disabled=(not has_id_bt),
        )

        run = st.form_submit_button(
            "🚀 Jalankan Preprocessing", use_container_width=True
        )

# ===================== MAIN: Preview sebelum =====================
st.subheader("Preview Data (sebelum)")
st.dataframe(df.head(10), use_container_width=True)

if not run:
    st.info("Atur opsi di sidebar lalu klik **Jalankan Preprocessing**.")
    st.stop()

# ===================== Jalankan preprocessing =====================
with st.spinner("Memproses teks..."):
    # 1) Text-level pipeline
    df["Deskripsi_Bersih"] = preprocess_series(
        df["isi"],
        use_ftfy=use_ftfy,
        force_ascii=force_ascii,
        use_stopwords_text=use_stopwords_text,
        use_stemming_text=use_stemming_text,
        to_lower=to_lower,
        remove_punct=remove_punct,
        remove_digits=remove_digits,
        redact=redact,
        redact_policy=redact_policy,
        redact_enabled=redact_enabled,
    )

    # 2) Timestamp (opsional)
    if want_ts:
        df["timestamp"] = extract_timestamp_from_id_bugtrack(df["id_bugtrack"])
        n_na = int(df["timestamp"].isna().sum())
        if n_na > 0:
            st.warning(
                f"{n_na} baris gagal diparse menjadi timestamp dari `id_bugtrack`."
            )

    # 3) Tokenization & downstream (token-level)
    tokens_series = df["Deskripsi_Bersih"].apply(
        lambda s: tokenize_text(s, method=tok_method)
    )

    # stopword (token-level)
    if token_stop:
        tokens_series = tokens_series.apply(
            lambda toks: [t for t in toks if t not in STOPWORDS_SET]
        )

    # stemming/lemmatization pada token
    if lemma_choice == "Stemming (Sastrawi)":
        tokens_series = tokens_series.apply(lambda toks: [stemmer.stem(t) for t in toks])
    elif lemma_choice == "Lemmatization (Stanza)" and _STANZA_OK:
        tokens_series = tokens_series.apply(lemmatize_with_stanza)

    df["tokens"] = tokens_series
    df["tokens_str"] = df["tokens"].apply(lambda toks: " ".join(toks))

# ===================== Output =====================
st.subheader("Preview Data (sesudah)")
cols_to_show = ["isi", "Deskripsi_Bersih", "tokens"]
if "timestamp" in df.columns:
    cols_to_show.append("timestamp")
st.dataframe(df[cols_to_show].head(10), use_container_width=True)

m1, m2 = st.columns(2)
m1.metric("Jumlah baris", f"{df.shape[0]:,}")
m2.metric("Jumlah kolom", f"{df.shape[1]:,}")

csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Download Dataset Preprocessed (CSV)",
    data=csv_bytes,
    file_name="dataset_preprocessed.csv",
    mime="text/csv",
)

# Simpan untuk halaman lain
st.session_state["df_clean"] = df
st.session_state["tokens"] = df["tokens"]
st.session_state["tokens_str"] = df["tokens_str"]

st.success("Preprocessing selesai dan disimpan ke sesi ✅")
