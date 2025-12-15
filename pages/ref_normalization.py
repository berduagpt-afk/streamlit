# ref_normalization.py 
# Kamus Word Normalization & Lexical Normalization untuk domain perpajakan

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
    "sptmasa": "spt masa",
    "sptmasapajak": "spt masa",
    "spttahunan": "spt tahunan",

    "skpkb.": "skpkb",
    "skpkn": "skpn",
    "skplb.": "skplb",

    # Surat tagihan
    "stp.": "stp",

    # Aplikasi DJP
    "efiling": "e-filing",
    "e filing": "e-filing",
    "efaktur": "e-faktur",
    "e faktur": "e-faktur",
    "ebilling": "e-billing",
    "e billing": "e-billing",
    "ebupot": "e-bupot",
    "e bupot": "e-bupot",

    "djponline": "djp online",
    "djpon": "djp online",

    # Layanan
    "pbb-p2": "pbb",
    "bea materai": "bea meterai",

    # Misc
    "ar.": "ar",
    "acct rep": "ar",
}


# Pola regex → frasa kanonik (lexical, konteks pajak)
# ---------------------------------------------------------
# 2) LEXICAL NORMALIZATION_MAP
# Pola regex (fleksibel) → istilah baku (kanonik)
# ---------------------------------------------------------
LEXICAL_NORMALIZATION_MAP = {

    # --- Aktor Perpajakan ---
    r"\bwp\b": "wajib pajak",
    r"\bwpop\b": "wajib pajak orang pribadi",
    r"\bwp op\b": "wajib pajak orang pribadi",
    r"\bwp_badan\b": "wajib pajak badan",
    r"\bwp badan\b": "wajib pajak badan",

    r"\bbut\b": "bentuk usaha tetap",
    r"\bar\b": "account representative",
    r"\bkpp\b": "kantor pelayanan pajak",
    r"\bkp2kp\b": "kantor pelayanan penyuluhan dan konsultasi perpajakan",

    r"\bdjp\b": "direktorat jenderal pajak",
    r"\bdirjen pajak\b": "direktorat jenderal pajak",

    # --- Identitas Pajak ---
    r"\bnpwp\b": "nomor pokok wajib pajak",
    r"\bnik\b": "nomor induk kependudukan",
    r"\bnop\b": "nomor objek pajak",

    # --- Jenis Pajak ---
    r"\bpph\b": "pajak penghasilan",

    r"\bpph\s?21\b": "pajak penghasilan pasal 21",
    r"\bpph\s?22\b": "pajak penghasilan pasal 22",
    r"\bpph\s?23\b": "pajak penghasilan pasal 23",
    r"\bpph\s?25\b": "pajak penghasilan pasal 25",
    r"\bpph\s?26\b": "pajak penghasilan pasal 26",
    r"\bpph\s?4(ayat)?\s?2\b": "pajak penghasilan pasal 4 ayat 2",

    r"\bppn\b": "pajak pertambahan nilai",
    r"\bppnbm\b": "pajak penjualan atas barang mewah",

    r"\bbkp\b": "barang kena pajak",
    r"\bjkp\b": "jasa kena pajak",

    # --- Dokumen Penting ---
    r"\bspt\b": "surat pemberitahuan",
    r"\bspt masa\b": "surat pemberitahuan masa",
    r"\bspt tahunan\b": "surat pemberitahuan tahunan",

    r"\bskp\b": "surat ketetapan pajak",
    r"\bskpkb\b": "surat ketetapan pajak kurang bayar",
    r"\bskpkbt\b": "surat ketetapan pajak kurang bayar tambahan",
    r"\bskpn\b": "surat ketetapan pajak nihil",
    r"\bskplb\b": "surat ketetapan pajak lebih bayar",

    r"\bstp\b": "surat tagihan pajak",

    r"\bsp2dk\b": "surat permintaan penjelasan atas data dan/atau keterangan",
    r"\bsse\b": "surat setoran elektronik",
    r"\bssp\b": "surat setoran pajak",

    r"\bnsfp\b": "nomor seri faktur pajak",
    r"\bfaktur pajak\b": "faktur pajak",

    # --- Hak & Prosedur ---
    r"\brestitusi\b": "pengembalian kelebihan pembayaran pajak",
    r"\bkompensasi lb\b": "kompensasi kelebihan pembayaran pajak",
    r"\bpengurangan sanksi\b": "pengurangan atau penghapusan sanksi administrasi",
    r"\bpenghapusan sanksi\b": "pengurangan atau penghapusan sanksi administrasi",

    r"\bpembetulan spt\b": "pembetulan surat pemberitahuan",
    r"\bpembetulan sk\b": "pembetulan surat ketetapan pajak",

    r"\bkeberatan\b": "keberatan pajak",
    r"\bbanding\b": "upaya hukum banding",
    r"\bpk\b": "peninjauan kembali",

    # --- Aplikasi DJP ---
    r"\bdjp online\b": "djp online",
    r"\be-filing\b": "e-filing",
    r"\be-faktur\b": "e-faktur",
    r"\be-billing\b": "e-billing",
    r"\be-bupot\b": "e-bupot",

    # --- Istilah teknis UU ---
    r"\bsubjek pajak\b": "subjek pajak",
    r"\bobjek pajak\b": "objek pajak",
    r"\bmasa pajak\b": "masa pajak",
    r"\btahun pajak\b": "tahun pajak",
    r"\bdaerah pabean\b": "daerah pabean",

    # --- Istilah umum helpdesk ---
    r"\blh\b": "lebih bayar",
    r"\bpkp\b": "pengusaha kena pajak",
}

