# pages/ref_sintaksis.py
# PATCH: tambahkan stopword frasa (regex) sebelum tokenization

from __future__ import annotations
from typing import Iterable, Set
import re

# ======================================================
# 1) STOPWORD CUSTOM (token-level)
# ======================================================
STOPWORD_CUSTOM_SET: Set[str] = {
    "selamat", "pagi", "siang", "sore", "malam",
    "assalamualaikum", "waalaikumsalam", "salam",

    "mohon", "minta", "tolong", "bantu", "dibantu", "bantuannya",
    "izin", "ijin", "permisi", "harap", "kiranya", "sekiranya",
    "terima", "kasih", "terimakasih", "makasih",
    "thanks", "thank", "thx",

    "dengan", "hormat",
    "yang", "kami", "saya", "anda", "bapak", "ibu", "pak", "bu", "sdh", "sudah",
    "apakah", "dimana", "kapan", "bagaimana", "kenapa", "mengapa",
    "sehubungan", "terkait", "perihal", "mengenai", "tentang",
    "berikut", "sebagai", "adalah", "yaitu", "yakni",
    "agar",

    "mau", "ingin", "akan", "bisa", "tidak",

    "nomor_pokok_wajib_pajak", "nomor_induk_kependudukan",
}
STOPWORD_CUSTOM_SET.discard("tidak")  # jangan buang negasi

# ======================================================
# 2) KEEP TERMS (whitelist)
# ======================================================
KEEP_TERMS: Set[str] = {
    "error", "gagal", "login", "lupa", "reset", "otp", "email", "nomor",
    "sertifikat", "password", "verifikasi",

    "npwp", "nik", "nop",
    "spt", "sse", "ssp", "sp2dk",
    "pph", "ppn", "ppnbm",
    "e_faktur", "e_billing", "e_filing", "djp_online",
}

# ======================================================
# 2b) STOPWORD PHRASE (regex-level) — PATCH BARU
# ======================================================
# Dipakai untuk menghapus frasa template sebelum tokenization.
# Semua pattern dibuat longgar untuk variasi spasi/tanda baca.
STOPPHRASE_PATTERNS = [
    r"\bterima\s*kasih\b",
    r"\bterima\s*ksh\b",
    r"\btrimakasih\b",
    r"\bterimakasih\b",
    r"\bmohon\s*bantu(annya)?\b",
    r"\bdengan\s*hormat\b",
    r"\bselamat\s*pagi\b",
    r"\bselamat\s*siang\b",
    r"\bselamat\s*sore\b",
    r"\bselamat\s*malam\b",
    r"\bass?alamualaik(um)?\b",
    r"\bwaalaikumsalam\b",
]

_RE_STOPPHRASES = re.compile("|".join(f"(?:{p})" for p in STOPPHRASE_PATTERNS), flags=re.IGNORECASE)

def remove_stop_phrases(text: str) -> str:
    """
    Hapus frasa template sebelum tokenization.
    Dipanggil setelah text_norm (biasanya sudah lowercase/punct cleaned),
    tapi kita tetap IGNORECASE agar aman.
    """
    if text is None:
        return ""
    s = str(text)
    s = _RE_STOPPHRASES.sub(" ", s)
    return re.sub(r"\s{2,}", " ", s).strip()

# ======================================================
# 3) Helper functions (tetap)
# ======================================================
def get_custom_stopwords(
    *,
    extra_add: Iterable[str] | None = None,
    extra_remove: Iterable[str] | None = None,
) -> Set[str]:
    sw = set(STOPWORD_CUSTOM_SET)
    if extra_add:
        sw.update({str(x).strip().lower() for x in extra_add if str(x).strip()})
    if extra_remove:
        sw.difference_update({str(x).strip().lower() for x in extra_remove if str(x).strip()})
    return sw


def remove_custom_stopwords_tokens(
    tokens: list[str],
    *,
    stopwords: Set[str] | None = None,
    keep_terms: Set[str] | None = None,
) -> list[str]:
    sw = stopwords if stopwords is not None else STOPWORD_CUSTOM_SET
    keep = keep_terms if keep_terms is not None else KEEP_TERMS

    out: list[str] = []
    for t in tokens:
        tl = t.lower()
        if tl in keep:
            out.append(t)
            continue
        if tl in sw:
            continue
        out.append(t)
    return out
