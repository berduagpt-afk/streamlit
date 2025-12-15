# redaction_config.py
# Konfigurasi regex + token pengganti untuk "Penghilangan elemen non-informatif"
# di dataset tiket insiden DJP (NPWP, NIK, NOP, nama WP, email, dsb).

import re
from collections import defaultdict

# -----------------------------------------------------------------------------
# 1. POLA-POLA REGEX
# -----------------------------------------------------------------------------

# Placeholder tag yang mungkin sudah ada di teks (misal hasil masking sebelumnya)
PLACEHOLDER_TAGS = re.compile(
    r"<\s*(NPWP|NAMA_WP|NO_SKP|NO_DOKUMEN|NO_CM|NO_PC|NO_NIK|NIK|NOP)\s*>",
    flags=re.IGNORECASE,
)

# NPWP: 99.999.999.9-999.999
NPWP_SEPARATED = re.compile(
    r"\b\d{2}\.\d{3}\.\d{3}\.\d-\d{3}\.\d{3}\b"
)

# NPWP compact: 15 digit
NPWP_COMPACT = re.compile(
    r"\b\d{15}\b"
)

# NIK: 16 digit
NIK_PATTERN = re.compile(
    r"\b\d{16}\b"
)

# NOP PBB
NOP_PATTERN = re.compile(
    r"\bNOP\s*[:\-]?\s*[0-9\.\-]{10,30}\b",
    flags=re.IGNORECASE,
)

# Nomor dokumen / SK / kasus / CM / PC umum
DOC_NUMBER_GENERIC = re.compile(
    r"\b(no\.?|nomor)\s+"
    r"(skp|sk|dokumen|dok|kasus|cm|pc)\s*[:\-]?\s*"
    r"[A-Za-z0-9\/\.\-]+",
    flags=re.IGNORECASE,
)

# Kode PC khusus (misal jodo / PCxxx)
PC_CODE_PATTERN = re.compile(
    r"\b(?:PC|jodo)\s*[A-Za-z0-9_\-\.]{2,30}\b",
    flags=re.IGNORECASE,
)

# Nama Wajib Pajak (heuristik)
NAMA_WP_PATTERN = re.compile(
    r"(atas\s+wajib\s+pajak|nama\s+wp)\s+"
    r"[A-Z0-9][A-Za-z0-9\.\s'\-]{2,60}",
    flags=re.IGNORECASE,
)

# Email
EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)

# URL
URL_PATTERN = re.compile(
    r"https?://\S+"
)

# Nomor telepon (Indonesia style, sangat generik)
PHONE_PATTERN = re.compile(
    r"\b(?:\+62|62|0)\d{8,13}\b"
)

# Angka panjang (opsional, kalau mau dibersihkan juga)
LONG_NUMBER_PATTERN = re.compile(
    r"\b\d{8,}\b"
)

# -----------------------------------------------------------------------------
# 2. DAFTAR RULE: pola + token pengganti
#    Urutan penting: yang spesifik dikerjakan dulu,
#    baru yang generik (misal LONG_NUMBER).
# -----------------------------------------------------------------------------
REDACTION_RULES = [
    # Jika sudah ada placeholder <NPWP> dsb, samakan saja formatnya
    {
        "label": "placeholder_tags",
        "pattern": PLACEHOLDER_TAGS,
        "token": "<PLACEHOLDER>",
    },
    {
        "label": "npwp",
        "pattern": NPWP_SEPARATED,
        "token": "<NPWP>",
    },
    {
        "label": "npwp",
        "pattern": NPWP_COMPACT,
        "token": "<NPWP>",
    },
    {
        "label": "nik",
        "pattern": NIK_PATTERN,
        "token": "<NIK>",
    },
    {
        "label": "nop",
        "pattern": NOP_PATTERN,
        "token": "<NOP>",
    },
    {
        "label": "doc_number",
        "pattern": DOC_NUMBER_GENERIC,
        "token": "<NO_DOKUMEN>",
    },
    {
        "label": "pc_code",
        "pattern": PC_CODE_PATTERN,
        "token": "<NO_PC>",
    },
    {
        "label": "nama_wp",
        "pattern": NAMA_WP_PATTERN,
        "token": "<NAMA_WP>",
    },
    {
        "label": "email",
        "pattern": EMAIL_PATTERN,
        "token": "<EMAIL>",
    },
    {
        "label": "url",
        "pattern": URL_PATTERN,
        "token": "<URL>",
    },
    {
        "label": "phone",
        "pattern": PHONE_PATTERN,
        "token": "<NO_TELP>",
    },
    # OPSIONAL: nyalakan kalau mau semua angka panjang dijadikan token umum
    # Hati-hati kalau banyak nominal rupiah
    # {
    #     "label": "long_number",
    #     "pattern": LONG_NUMBER_PATTERN,
    #     "token": "<LONG_NUMBER>",
    # },
]

# -----------------------------------------------------------------------------
# 3. FUNGSI UTAMA: Ganti elemen non-informatif dengan token
# -----------------------------------------------------------------------------
def replace_non_informative(text: str, return_counts: bool = False):
    """
    Mengganti elemen non-informatif (NPWP, NIK, NOP, NAMA_WP, dll)
    menjadi token umum seperti <NPWP>, <NAMA_WP>, dst.

    Param:
        text          : teks asli
        return_counts : jika True, juga mengembalikan dict jumlah token

    Return:
        - jika return_counts=False -> teks_tertokenisasi
        - jika return_counts=True  -> (teks_tertokenisasi, dict_count)
    """
    counts = defaultdict(int)
    new_text = text

    for rule in REDACTION_RULES:
        pattern = rule["pattern"]
        token = rule["token"]

        def _sub(match):
            counts[token] += 1
            return token

        new_text = pattern.sub(_sub, new_text)

    if return_counts:
        return new_text, dict(counts)
    return new_text


# -----------------------------------------------------------------------------
# 4. FUNGSI BANTUAN: Hitung placeholder pada teks yang sudah diproses
# -----------------------------------------------------------------------------
PLACEHOLDER_FINDER = re.compile(r"<[A-Z_]+>")

def count_placeholders(text: str):
    """
    Menghitung frekuensi token placeholder (mis. <NPWP>, <NOP>, <NAMA_WP>)
    pada teks yang sudah diproses.

    Return:
        dict: {token: jumlah}
    """
    counts = defaultdict(int)
    for m in PLACEHOLDER_FINDER.finditer(text):
        counts[m.group(0)] += 1
    return dict(counts)
