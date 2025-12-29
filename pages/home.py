# pages/home.py — DJP Themed Home Page (PATCHED)
import streamlit as st
from datetime import datetime

# --- Guard: hanya untuk user yang login ---
if not st.session_state.get("logged_in", False):
    st.error("Silakan login dulu untuk mengakses halaman ini.")
    st.stop()

# --- Palet warna (tema DJP) ---
DJP_BLUE = "#0B3A82"
DJP_GOLD = "#FFC20E"
BG_PANEL = "#F6F7FB"
BORDER   = "#E5E7EB"
TEXT     = "#0F172A"
MUTED    = "#64748B"

def fmt_dt(x):
    """Format last_update agar rapi (mendukung datetime / string / lainnya)."""
    if x is None:
        return "—"
    if isinstance(x, datetime):
        return x.strftime("%Y-%m-%d %H:%M")
    return str(x)

# --- Styling Global ---
st.markdown(
    f"""
<style>
:root {{
  --djp-blue: {DJP_BLUE};
  --djp-gold: {DJP_GOLD};
  --surface: #FFFFFF;
  --panel: {BG_PANEL};
  --border: {BORDER};
  --text: {TEXT};
  --muted: {MUTED};
}}

.hero {{
  position: relative;
  padding: 2rem 1.5rem;
  border-radius: 18px;
  border: 1px solid var(--border);
  color: white;
  overflow: hidden;

  /* Fallback background (untuk browser yang tidak support color-mix) */
  background: var(--djp-blue);

  /* Enhanced background (modern browsers) */
  background:
    radial-gradient(1200px 500px at 0% -20%, color-mix(in oklab, var(--djp-gold) 25%, white) 40%, transparent 60%),
    linear-gradient(135deg, color-mix(in oklab, var(--djp-blue) 85%, #0a2e66) 0%, var(--djp-blue) 65%, color-mix(in oklab, var(--djp-gold) 80%, #fff) 120%);
}}

.hero h1 {{ margin: 0 0 .25rem 0; font-size: 2rem; line-height: 1.1; font-weight: 800; }}
.hero p  {{ margin: 0 0 .75rem 0; color: rgba(255,255,255,.92); font-size:1rem; }}

.hero-logo {{
  position: absolute;
  right: 16px;
  top: 16px;
  width: 64px;
  height: 64px;
  opacity: .95;
}}
.hero-logo svg {{
  width: 64px;
  height: 64px;
  display: block;
}}

.kpi-grid {{
  display: grid;
  grid-template-columns: repeat(4,1fr);
  gap: .75rem;
  margin-top: .75rem;
}}
.kpi-card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-top: 4px solid var(--djp-gold);
  border-radius: 16px;
  padding: 1rem;
}}
.kpi-card .label {{ color: var(--muted); font-size:.9rem; }}
.kpi-card .value {{ font-size: 1.4rem; font-weight: 700; color: var(--text); }}

@media (max-width: 1100px){{
  .kpi-grid{{ grid-template-columns: repeat(2,1fr); }}
}}
@media (max-width: 640px){{
  .kpi-grid{{ grid-template-columns: 1fr; }}
}}

.section {{ margin-top: 1.25rem; }}
.footer {{ color: var(--muted); font-size:0.9rem; }}
</style>
""",
    unsafe_allow_html=True,
)

# --- HERO SECTION ---
st.markdown('<div class="hero">', unsafe_allow_html=True)

# Logo DJP minimalis SVG di pojok kanan atas (PATCH: position via CSS)
st.markdown(
    f"""
<div class="hero-logo">
  <svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="djplogo" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="{DJP_BLUE}" />
        <stop offset="100%" stop-color="{DJP_GOLD}" />
      </linearGradient>
    </defs>
    <path d="M32 4 C46 4 60 10 60 24 v18 c0 14-12 24-28 30C16 66 4 56 4 42V24C4 10 18 4 32 4z"
          fill="url(#djplogo)" stroke="white" stroke-width="2"/>
    <path d="M16 26 l32 0 M16 34 l32 0 M16 42 l24 0"
          stroke="white" stroke-width="3" stroke-linecap="round"/>
  </svg>
</div>
""",
    unsafe_allow_html=True,
)

col1, col2 = st.columns([1.8, 1])
with col1:
    st.markdown("<h1>Incident Labeling • DJP Prototype</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p>Sistem pelabelan otomatis tiket insiden berulang menggunakan pendekatan Sintaksis (TF-IDF) dan Semantik (IndoBERT) di lingkungan DJP.</p>",
        unsafe_allow_html=True,
    )
    st.write("**Aksi Cepat**")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.page_link("pages/upload_dataset.py", label="Upload Dataset", icon=":material/upload_file:", help="Unggah file CSV/Excel untuk diproses")
    with col_b:
        st.page_link("pages/load_dataset.py", label="Dataset Siap Preprocessing", icon=":material/cleaning_services:", help="Normalisasi teks dan stemming otomatis")
    with col_c:
        st.page_link("pages/cluster_dashboard.py", label="Cluster Dashboard", icon=":material/analytics:", help="Visualisasi klaster dan pelabelan insiden")

with col2:
    st.markdown(
        f"""
<svg viewBox="0 0 260 160" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;">
  <defs>
    <linearGradient id="gb" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0%" stop-color="{DJP_BLUE}"/>
      <stop offset="100%" stop-color="{DJP_GOLD}"/>
    </linearGradient>
    <filter id="s" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="10" stdDeviation="10" flood-opacity="0.25"/>
    </filter>
  </defs>
  <rect x="14" y="16" width="232" height="120" rx="16" fill="url(#gb)" filter="url(#s)"/>
  <g transform="translate(34,36)" opacity="0.95">
    <rect x="0" y="0" width="120" height="16" rx="6" fill="#ffffff"/>
    <rect x="0" y="26" width="160" height="16" rx="6" fill="#ffffff"/>
    <rect x="0" y="52" width="140" height="16" rx="6" fill="#ffffff"/>
    <rect x="0" y="78" width="180" height="16" rx="6" fill="#ffffff"/>
  </g>
  <g transform="translate(190,28)">
    <path d="M20 0 C40 0 60 10 60 28 v22 c0 18-20 30-40 36c-20-6-40-18-40-36V28C-0 10 20 0 20 0z" fill="#ffffff" opacity="0.9"/>
    <rect x="10" y="16" width="20" height="6" rx="3" fill="{DJP_BLUE}"/>
    <rect x="10" y="28" width="28" height="6" rx="3" fill="{DJP_GOLD}"/>
    <rect x="10" y="40" width="24" height="6" rx="3" fill="{DJP_BLUE}"/>
  </g>
</svg>
""",
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)

# --- KPI Ringkas ---
df = st.session_state.get("df")
n_tickets = len(df) if df is not None else 0
n_clusters = st.session_state.get("n_clusters", 0)
n_outliers = st.session_state.get("n_outliers", 0)
last_update = st.session_state.get("last_update")

outlier_pct = (n_outliers / n_tickets * 100) if n_tickets else None

st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
for label, value in [
    ("Total Ticket", f"{n_tickets:,}"),
    ("Jumlah Cluster", n_clusters if n_clusters else "—"),
    ("Outlier (%)", f"{outlier_pct:.1f}%" if outlier_pct is not None else "—"),
    ("Terakhir diproses", fmt_dt(last_update)),
]:
    st.markdown(
        f'<div class="kpi-card">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )
st.markdown("</div>", unsafe_allow_html=True)

# --- Panduan Alur Kerja ---
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("Alur Kerja Sistem (CRISP-DM)")
st.markdown(
    """
1. :material/upload_file: **Upload Dataset** – unggah data insiden LASIS (CSV/Excel).  
2. :material/cleaning_services: **Preprocessing** – normalisasi teks, tokenisasi, stopword removal, stemming (Sastrawi).  
3. :material/analytics: **Clustering & Labeling** – TF-IDF/Cosine Similarity atau IndoBERT + HDBSCAN.  
4. :material/insights: **Evaluasi & Reporting** – validasi model dan tampilkan ringkasan manajerial.  
"""
)
st.markdown("</div>", unsafe_allow_html=True)

# --- Cuplikan Dataset ---
if df is None:
    st.info("Belum ada dataset yang dimuat. Mulai dengan **Upload Dataset** terlebih dahulu.")
else:
    with st.expander("Cuplikan 5 baris pertama dataset"):
        st.dataframe(df.head(), use_container_width=True)

# --- FOOTER ---
st.markdown("---")
st.caption(f"© {datetime.now().year} • Prototype Tesis Achmad Luthfi • Tema: Direktorat Jenderal Pajak (Biru–Emas)")
