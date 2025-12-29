# ref_normalization.py
# Kamus Word Normalization & Lexical Normalization untuk domain perpajakan
#
# Catatan:
# - WORD_NORMALIZATION_MAP: kamu edit sendiri (sesuai request)
# - Placeholder replacement: diubah agar SESUAI JENIS (__PH_NPWP__, __PH_EMAIL__, dst)
# - Disediakan helper pipeline "stable_preprocess_order()" untuk urutan pemrosesan paling stabil
#   (boleh kamu pakai/abaikan, tapi urutannya sudah diset yang paling aman untuk kasus tiket)

import re
import unicodedata
from collections import Counter
from typing import Dict, Any, Tuple, Optional, Set

# ---------------------------------------------------------
# 1) WORD-LEVEL NORMALIZATION
# Bentuk variasi → bentuk baku (direct replace, token-level)
# ---------------------------------------------------------
WORD_NORMALIZATION_MAP = {
    # Identitas
    "npwpd": "npwp",
    "npw": "npwp",
    "npwp15": "npwp",
    "npwp 15": "npwp",
    "npwp16": "npwp",
    "npwp 16": "npwp",

    "nik.": "nik",
    "no nik": "nik",

    "nop.": "nop",
    "no nop": "nop",

    # Jenis Pajak
    "pph21": "pph 21",
    "pph22": "pph 22",
    "pph23": "pph 23",
    "pph25": "pph 25",
    "pph26": "pph 26",
    "pph4(2)": "pph 4 ayat 2",
    "pph final": "pph 4 ayat 2",

    "ppn11": "ppn",
    "ppnbm.": "ppnbm",

    # Dokumen
    "sptmasa": "spt_masa",
    "sptmasapajak": "spt_masa",
    "spttahunan": "spt_tahunan",

    "skpkb.": "skpkb",
    "skpkn": "skpn",
    "skplb.": "skplb",

    # Surat tagihan
    "stp.": "surat_tagihan_pajak",

    # Aplikasi DJP
    "efiling": "e_filing",
    "e filing": "e_filing",
    "efaktur": "e_faktur",
    "e faktur": "e_faktur",
    "ebilling": "e_billing",
    "e billing": "e_billing",
    "ebupot": "e_bupot",
    "e bupot": "e_bupot",

    "djponline": "djp_online",
    "djpon": "djp_online",

    # Layanan
    "pbb-p2": "pbb",
    "bea materai": "bea meterai",

    # Misc
    "ar.": "ar",
    "acct rep": "ar",
}

# ---------------------------------------------------------
# 2) LEXICAL NORMALIZATION_MAP
# Pola regex (fleksibel) → istilah baku (kanonik)
# ---------------------------------------------------------
LEXICAL_NORMALIZATION_MAP = {
    # --- Aktor Perpajakan ---
    r"\bwp\b": "wajib_pajak",
    r"\bwpop\b": "wajib_pajak_orang_pribadi",
    r"\bwp op\b": "wajib_pajak_orang_pribadi",
    r"\bwp_badan\b": "wajib_pajak_badan",
    r"\bwp badan\b": "wajib_pajak_badan",

    r"\bbut\b": "bentuk_usaha_tetap",
    r"\bar\b": "account_representative",
    r"\bkpp\b": "kantor_pelayanan_pajak",
    r"\bkpp\s?pratama\b": "kantor_pelayanan_pajak_pratama",
    r"\bkpp\s?khusus\b": "kantor_pelayanan_pajak_khusus",
    r"\bkanwil\b": "kantor_wilayah",
    r"\bkanwil\s?djp\b": "kantor_wilayah_direktorat_jenderal_pajak",
    r"\bkp2kp\b": "kantor_pelayanan_penyuluhan_dan_konsultasi_perpajakan",

    r"\bdjp\b": "direktorat_jenderal_pajak",
    r"\bditjen pajak\b": "direktorat_jenderal_pajak",
    r"\bdirjen pajak\b": "direktur_jenderal_pajak",

    # --- Identitas Pajak ---
    r"\bnpwp\b": "nomor_pokok_wajib_pajak",
    r"\bnik\b": "nomor_induk_kependudukan",
    r"\bnop\b": "nomor_objek_pajak",

    # --- Jenis Pajak ---
    r"\bpph\b": "pajak_penghasilan",
    r"\bpph\s?21\b": "pajak_penghasilan_pasal_21",
    r"\bpph\s?22\b": "pajak_penghasilan_pasal_22",
    r"\bpph\s?23\b": "pajak_penghasilan_pasal_23",
    r"\bpph\s?25\b": "pajak_penghasilan_pasal_25",
    r"\bpph\s?26\b": "pajak_penghasilan_pasal_26",
    r"\bpph\s*4\s*(?:ayat\s*)?\(?\s*2\s*\)?\b": "pajak_penghasilan_pasal_4_ayat_2",

    r"\bppn\b": "pajak_pertambahan_nilai",
    r"\bppnbm\b": "pajak_penjualan_atas_barang_mewah",

    r"\bbkp\b": "barang_kena_pajak",
    r"\bjkp\b": "jasa_kena_pajak",

    # --- Dokumen Penting ---
    r"\bspt\b": "surat_pemberitahuan",
    r"\bspt masa\b": "surat_pemberitahuan_masa",
    r"\bspt tahunan\b": "surat_pemberitahuan_tahunan",
    r"\bskp\b": "surat_ketetapan_pajak",
    r"\bskpkb\b": "surat_ketetapan_pajak_kurang_bayar",
    r"\bskpkbt\b": "surat_ketetapan_pajak_kurang_bayar_tambahan",
    r"\bskpn\b": "surat_ketetapan_pajak_nihil",
    r"\bskplb\b": "surat_ketetapan_pajak_lebih_bayar",
    r"\bstp\b": "surat_tagihan_pajak",

    r"\bsp2dk\b": "surat_permintaan_penjelasan_atas_data_dan_atau_keterangan",
    r"\bsse\b": "surat_setoran_elektronik",
    r"\bssp\b": "surat_setoran_pajak",

    r"\bnsfp\b": "nomor_seri_faktur_pajak",
    r"\bfaktur pajak\b": "faktur_pajak",

    # --- Hak & Prosedur ---
    r"\bpembetulan spt\b": "pembetulan_surat_pemberitahuan",
    r"\bpembetulan sk\b": "pembetulan_surat_ketetapan_pajak",

    r"\bkeberatan\b": "keberatan_pajak",
    r"\bkb\b": "keberatan_pajak",
    r"\bbanding\b": "upaya_hukum_banding",
    r"\bpk\b": "peninjauan_kembali",

    # --- Aplikasi DJP ---
    r"\bdjp online\b": "djp_online",
    r"\be-filing\b": "e_filing",
    r"\be-faktur\b": "e_faktur",
    r"\be-billing\b": "e_billing",
    r"\be-bupot\b": "e_bupot",

    # --- Istilah teknis UU ---
    r"\bsubjek pajak\b": "subjek_pajak",
    r"\bobjek pajak\b": "objek_pajak",
    r"\bmasa pajak\b": "masa_pajak",
    r"\btahun pajak\b": "tahun_pajak",
    r"\bdaerah pabean\b": "daerah_pabean",

    # --- Istilah umum helpdesk ---
    r"\blh\b": "lebih_bayar",
    r"\blb\b": "lebih_bayar",
    r"\bpkp\b": "pengusaha_kena_pajak",
    r"\bde\b": "status_dihapus",
    r"\bba\b": "berita_acara",
}

# =========================================================
# 3) PLACEHOLDER HANDLING (TYPE-AWARE TOKENS)
# =========================================================
# Bentuk placeholder input yang didukung:
# - Angle placeholder: <NPWP>, <NIK>, <EMAIL>, dst (case-insensitive)
# - URL / Email / IP nyata
# - Long number (>= 8 digit) untuk nomor dokumen/identitas "mentah"
#
# Output token placeholder:
# - __PH_NPWP__ , __PH_EMAIL__ , __PH_URL__ , __PH_IP__ , __PH_LONGNUM__ , dst
# - Jika jenis tidak dikenali → __PH_OTHER__

_PLACEHOLDER_KEYS = {
    "NPWP", "NIK", "NOP",
    "NAMA_WP", "NO_WP",
    "NO_DOKUMEN", "NO_SKP", "NO_CM", "NO_PC",
    "EMAIL", "TELP", "NO_HP",
    "URL", "IP", "IMEI", "IMSI",
    "ALAMAT", "KODE_POS",
    "NAMA", "USERNAME", "USER",
}

_RE_ANGLE_PLACEHOLDER = re.compile(r"<\s*([A-Z0-9_]+)\s*>", flags=re.IGNORECASE)
_RE_URL = re.compile(r"\bhttps?://\S+|\bwww\.\S+", flags=re.IGNORECASE)
_RE_EMAIL = re.compile(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", flags=re.IGNORECASE)
_RE_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_RE_LONG_NUMBER = re.compile(r"\b\d{8,}\b")


def count_placeholders(text: Optional[str]) -> Dict[str, Any]:
    """
    Menghitung placeholder angle (<NPWP>, <EMAIL>, dst).
    Untuk URL/Email/IP/longnumber, hitungannya disediakan di replace_and_count_placeholders().
    """
    if text is None:
        return {"total": 0, "by_type": {}, "raw_matches": []}

    s = str(text)
    matches = [m.group(1) for m in _RE_ANGLE_PLACEHOLDER.finditer(s)]
    matches_norm = [k.strip().upper() for k in matches]
    c = Counter(matches_norm)
    return {"total": int(sum(c.values())), "by_type": dict(c), "raw_matches": matches_norm}


def _ph_token(kind: str) -> str:
    """Bentuk token placeholder final."""
    return f" _PH_{kind}_ "


def replace_non_informative(
    text: Optional[str],
    *,
    replace_angle_placeholders: bool = True,
    replace_url: bool = True,
    replace_email: bool = True,
    replace_ip: bool = True,
    replace_long_numbers: bool = True,
) -> str:
    """
    Replace non-informatif menjadi placeholder bertipe.

    Output token:
    - <NPWP> -> __PH_NPWP__
    - URL -> __PH_URL__
    - email -> __PH_EMAIL__
    - IP -> __PH_IP__
    - long number (>=8 digit) -> __PH_LONGNUM__
    - <UNKNOWN> -> __PH_OTHER__ (jika tidak ada di _PLACEHOLDER_KEYS)
    """
    if text is None:
        return ""

    s = str(text)

    if replace_angle_placeholders:
        def _repl_angle(m: re.Match) -> str:
            raw = (m.group(1) or "").strip().upper()
            kind = raw if raw in _PLACEHOLDER_KEYS else "OTHER"
            return _ph_token(kind)

        s = _RE_ANGLE_PLACEHOLDER.sub(_repl_angle, s)

    if replace_url:
        s = _RE_URL.sub(_ph_token("URL"), s)

    if replace_email:
        s = _RE_EMAIL.sub(_ph_token("EMAIL"), s)

    if replace_ip:
        s = _RE_IP.sub(_ph_token("IP"), s)

    if replace_long_numbers:
        s = _RE_LONG_NUMBER.sub(_ph_token("LONGNUM"), s)

    return re.sub(r"\s{2,}", " ", s).strip()


def replace_and_count_placeholders(
    text: Optional[str],
    *,
    replace_angle_placeholders: bool = True,
    replace_url: bool = True,
    replace_email: bool = True,
    replace_ip: bool = True,
    replace_long_numbers: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """
    Return (cleaned_text, stats).

    stats mencakup:
    - angle placeholders per jenis
    - url/email/ip/longnum count (terpisah)
    """
    if text is None:
        return "", {
            "total": 0,
            "by_type": {},
            "raw_matches": [],
            "by_regex": {"URL": 0, "EMAIL": 0, "IP": 0, "LONGNUM": 0},
        }

    s = str(text)

    angle_stats = count_placeholders(s)
    by_regex = {
        "URL": len(_RE_URL.findall(s)) if replace_url else 0,
        "EMAIL": len(_RE_EMAIL.findall(s)) if replace_email else 0,
        "IP": len(_RE_IP.findall(s)) if replace_ip else 0,
        "LONGNUM": len(_RE_LONG_NUMBER.findall(s)) if replace_long_numbers else 0,
    }

    cleaned = replace_non_informative(
        s,
        replace_angle_placeholders=replace_angle_placeholders,
        replace_url=replace_url,
        replace_email=replace_email,
        replace_ip=replace_ip,
        replace_long_numbers=replace_long_numbers,
    )

    # gabungkan total: angle + regex
    total = int(angle_stats.get("total", 0) + sum(by_regex.values()))
    stats = dict(angle_stats)
    stats["by_regex"] = by_regex
    stats["total"] = total
    return cleaned, stats


# =========================================================
# 4) OPTIONAL: REMOVE SINGLE-CHAR TOKENS
# =========================================================
def remove_single_char_tokens(
    text: Optional[str],
    *,
    keep: Optional[Set[str]] = None,
    also_remove_digits: bool = False,
) -> str:
    """
    Hapus token 1 karakter yang berdiri sendiri (mis: 'a', 'n').
    Disarankan dipanggil setelah punctuation cleaning.

    - keep: token 1-char yang dipertahankan (contoh: {'e'})
    - also_remove_digits: True untuk menghapus token digit 1-char juga.
    """
    if text is None:
        return ""

    keep = keep or set()
    toks = re.findall(r"\b\w+\b", str(text), flags=re.UNICODE)

    out = []
    for t in toks:
        if len(t) == 1 and t not in keep:
            if (not also_remove_digits) and t.isdigit():
                out.append(t)
            # else: drop token 1-char
        else:
            out.append(t)

    return " ".join(out).strip()


# =========================================================
# 5) STABLE PROCESSING ORDER (HELPER OPTIONAL)
# =========================================================
_RE_HTML = re.compile(r"<[^>]+>")

def strip_weird_chars(text_in: Optional[str], *, force_ascii: bool = False) -> str:
    """Utility yang konsisten dengan halaman normalization."""
    if text_in is None:
        return ""
    t = str(text_in)
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"[\u200B-\u200D\uFEFF]", "", t)
    t = re.sub(r"[\x00-\x1F\x7F-\x9F]", " ", t)
    if force_ascii:
        t = t.encode("ascii", "ignore").decode("ascii", "ignore")
    return re.sub(r"\s{2,}", " ", t).strip()


def apply_word_normalization(text_in: str) -> str:
    """Token-level-ish replace (kamu boleh edit sendiri WORD_NORMALIZATION_MAP)."""
    t = text_in
    for wrong, canon in WORD_NORMALIZATION_MAP.items():
        pattern = r"\b" + re.escape(wrong) + r"\b"
        t = re.sub(pattern, canon, t, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", t).strip()


def apply_lexical_normalization(text_in: str) -> str:
    """Regex/phrase-level replace."""
    t = text_in
    for pattern, repl in LEXICAL_NORMALIZATION_MAP.items():
        t = re.sub(pattern, repl, t, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", t).strip()


def stable_preprocess_order(
    text: Optional[str],
    *,
    use_ftfy: bool = False,
    ftfy_fix_func=None,
    force_ascii: bool = False,
    use_redaction: bool = True,
    to_lower: bool = True,
    use_word_norm: bool = True,
    use_lexical_norm: bool = True,
    remove_punct: bool = True,
    remove_digits: bool = False,
    remove_single_chars: bool = False,
) -> str:
    """
    Urutan pemrosesan paling stabil untuk tiket helpdesk (disarankan):

    1) ftfy (opsional, jika tersedia)               -> perbaiki mojibake
    2) strip_weird_chars                           -> NFKC, control chars
    3) remove HTML                                 -> stabilkan teks
    4) placeholder replacement type-aware           -> __PH_NPWP__/__PH_EMAIL__/...
    5) lowercase
    6) word normalization (direct map)
    7) lexical normalization (regex phrase)
    8) remove punct / digits                        (opsional; hati-hati bila butuh pola angka)
    9) remove single-char tokens (opsional; paling akhir)
    10) normalize whitespace
    """
    s = "" if text is None else str(text)

    # 1) ftfy
    if use_ftfy and ftfy_fix_func is not None:
        try:
            s = ftfy_fix_func(s)
        except Exception:
            pass

    # 3) HTML
    s = _RE_HTML.sub(" ", s)

    # 4) placeholder replacement (type-aware)
    if use_redaction:
        s = replace_non_informative(s)

    # 5) lower
    if to_lower:
        s = s.lower()

    # 6) word norm
    if use_word_norm:
        s = apply_word_normalization(s)

    # 7) lexical norm
    if use_lexical_norm:
        s = apply_lexical_normalization(s)

    # 8) punct/digits
    if remove_punct:
        s = re.sub(r"[^\w\s]", " ", s)
    if remove_digits:
        s = re.sub(r"\d+", " ", s)

    # 9) single-char tokens
    if remove_single_chars:
        s = remove_single_char_tokens(s, keep={"e"})

    # 10) whitespace
    return re.sub(r"\s+", " ", s).strip()
