"""
Konfigurasi & utilitas redaksi teks (URL, NPWP, No Dokumen, Nama WP, dsb.)

Cara pakai (di preprocessing.py):
    from redaction_config import redact_text, get_default_enabled, list_pattern_names

    default_enabled = get_default_enabled()
    names = list_pattern_names()
    # ... buat checkbox untuk setiap 'name' ...
    text_redacted, hits = redact_text(text, policy="placeholder", enabled=enabled_dict)
"""

from __future__ import annotations
import re
from functools import lru_cache
from typing import Dict, List, Tuple

# --------- Pola spesial yang butuh penggantian custom ----------
# Pertahankan label "Nama WP" tapi redaksi nilainya
# group(1) = prefix (label + separator), group(2) = nama
NAMA_WP_PATTERN = re.compile(
    r"(?i)\b(nama\s*(?:wajib\s*pajak|wp)\s*[:\-]?\s*)([A-Z][A-Za-z .'\-]{2,})"
)

# --------- Pola umum (bisa ditambah/ubah tanpa sentuh kode utama) ----------
# Setiap item: (NAME, PATTERN, FLAGS_STR)
# NAME dipakai sebagai placeholder __NAME__
REDACTION_SPECS: List[Tuple[str, str, str]] = [
    ("URL",     r"\b(?:https?://|www\.)[^\s]+", ""),
    ("EMAIL",   r"\b[\w\.-]+@[\w\.-]+\.\w+\b", ""),
    ("IP",      r"\b(?:(?:\d{1,3}\.){3}\d{1,3})\b", ""),
    ("MAC",     r"\b(?:[0-9A-Fa-f]{2}[:\-]){5}[0-9A-Fa-f]{2}\b", ""),
    ("PHONE",   r"\b(?:\+62|62|0)8\d{8,12}\b", ""),
    # NPWP: 99.999.999.9-999.999 atau 15 digit, dengan/ tanpa kata 'npwp'
    ("NPWP",    r"(?i)\b(?:npwp[:\-\s]*)?(?:\d{2}\.?\d{3}\.?\d{3}\.?\d{1}-?\d{3}\.?\d{3}|\d{15})\b", ""),
    # No dokumen/surat umum
    ("DOCNO",   r"(?i)\b(?:no\.?|nomor|doc(?:ument)?|surat)\s*[:\-]?\s*[A-Z0-9/\-\.]{4,}\b", ""),
    ("DOCPAT",  r"\b[0-9]{1,4}/[A-Z0-9.\-]{2,}/[0-9]{2,4}\b", ""),
    # ID tiket umum
    ("TICKET",  r"\b(?:INC|REQ|PRB|CHG)[-_]?\d{4,}\b", ""),
    # UUID & commit hash
    ("UUID",    r"\b[0-9a-fA-F]{8}\-[0-9a-fA-F]{4}\-[1-5][0-9a-fA-F]{3}\-[89abAB][0-9a-fA-F]{3}\-[0-9a-fA-F]{12}\b", ""),
    ("GITHASH", r"\b[0-9a-f]{7,40}\b", ""),
    # Path sistem
    ("WINPATH", r"\b[A-Za-z]:\\[^\s]+", ""),
    ("UNIXPATH", r"(?<!\w)/(?:[A-Za-z0-9._-]+/)+[A-Za-z0-9._-]+", ""),
]

# --------- Utilitas ---------
def list_pattern_names(include_special: bool = True) -> List[str]:
    names = [name for name, _, _ in REDACTION_SPECS]
    if include_special:
        names = ["NAMA_WP"] + names
    return names

def get_default_enabled(include_special: bool = True) -> Dict[str, bool]:
    return {name: True for name in list_pattern_names(include_special=include_special)}

def _compile_flags(flags_str: str) -> int:
    flags = 0
    if "i" in flags_str.lower():
        flags |= re.IGNORECASE
    return flags

@lru_cache(maxsize=1)
def compile_redactors():
    """
    Return:
        specials: list of ("NAMA_WP", compiled_pattern, replacer_func)
        general : list of (NAME, compiled_regex)
    """
    # specials
    def _nama_wp_replacer(match: re.Match) -> str:
        # ganti nilai nama dengan placeholder, prefix tetap
        return match.group(1) + " __NAME__ "

    specials = [("NAMA_WP", NAMA_WP_PATTERN, _nama_wp_replacer)]

    # general
    general = []
    for name, pat, flags_str in REDACTION_SPECS:
        rx = re.compile(pat, _compile_flags(flags_str))
        general.append((name, rx))
    return specials, general

def _placeholder(tag: str) -> str:
    return f" __{tag}__ "

def redact_text(
    text: str,
    policy: str = "placeholder",   # "placeholder" | "remove"
    enabled: Dict[str, bool] | None = None,
) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Redaksi teks berdasarkan konfigurasi di atas.
    Returns:
        text_redacted, hits  (hits = [(NAME, matched_text), ...])
    """
    if not isinstance(text, str):
        return "", []
    specials, general = compile_redactors()
    hits: List[Tuple[str, str]] = []

    def repl(tag: str) -> str:
        return _placeholder(tag) if policy == "placeholder" else " "

    # 1) specials (respect 'enabled' toggle)
    for tag, rx, fn in specials:
        if enabled is not None and not enabled.get(tag, True):
            continue
        def _r(m, _t=tag):
            hits.append((_t, m.group(0)))
            return fn(m).replace("__NAME__", _t if policy == "placeholder" else "").strip() if tag == "NAMA_WP" else repl(_t)
        text = rx.sub(_r, text)

    # 2) general
    for name, rx in general:
        if enabled is not None and not enabled.get(name, True):
            continue
        def _r(m, _n=name):
            hits.append((_n, m.group(0)))
            return repl(_n)
        text = rx.sub(_r, text)

    # rapikan spasi
    text = re.sub(r"\s+", " ", text).strip()
    return text, hits

__all__ = [
    "REDACTION_SPECS",
    "NAMA_WP_PATTERN",
    "list_pattern_names",
    "get_default_enabled",
    "compile_redactors",
    "redact_text",
]
