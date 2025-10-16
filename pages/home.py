# pages/home.py — DJP Themed Dashboard with Custom Icon
import streamlit as st
from datetime import datetime

# --- Guard: hanya untuk user yang login ---
if not st.session_state.get("logged_in", False):
    st.error("Silakan login dulu untuk mengakses halaman ini.")
    st.stop()

# --- Palet warna DJP ---
DJP_BLUE = "#0B3A82"
DJP_GOLD = "#FFC20E"
BG_PANEL = "#F6F7FB"
BORDER   = "#E5E7EB"
TEXT     = "#0F172A"
MUTED    = "#64748B"

# --- Styling global ---
st.markdown(f"""
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
  background:
    radial-gradient(1200px 500px at 0% -20%, color-mix(in oklab, var(--djp-gold) 25%, white) 40%, transparent 60%),
    linear-gradient(135deg, color-mix(in oklab, var(--djp-blue) 85%, #0a2e66) 0%, var(--djp-blue) 65%, color-mix(in oklab, var(--djp-gold) 80%, #fff) 120%);
  border: 1px solid var(--border);
  color: white;
  overflow: hidden;
}}
.hero h1 {{ margin: 0 0 .25rem 0; font-size: 2rem; line-height: 1.1; font-weight: 800; }}
.hero p  {{ margin: 0 0 .75rem 0; color: rgba(255,255,255,.9); }}
.hero-logo {{
  position: absolute;
  top: 1rem; right: 1rem;
  width: 56px; height: 56px;
}}

.link-chips {{
  display: flex;
  flex-wrap: wrap;
  gap: .5rem;
  margin-top: .5rem;
}}
.link-chips button[kind="secondary"] {{
  border-radius: 999px !important;
  border: 1px solid color-mix(in oklab, var(--djp-blue) 70%, white 30%) !important;
  background: color-mix(in oklab, var(--djp-blue) 5%, white 95%) !important;
  color: var(--djp-blue) !important;
  font-weight: 500 !important;
}}
.link-chips button[kind="secondary"]:hover {{
  background: color-mix(in oklab, var(--djp-blue) 10%, white 90%) !important;
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

.section {{ margin-top: 1.25rem; }}
.footer {{ color: var(--muted); }}
</style>
""", unsafe_allow_html=True)

# --- HERO SECTION ---
st.markdown('<div class="hero">', unsafe_allow_html=True)

# Tambahkan logo DJP minimalis SVG di pojok kanan atas
st.markdown(f"""
<div class="hero-logo">
  <svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="djplogo" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="{DJP_BLUE}" />
        <stop offset="100%" stop-color="{DJP_GOLD}" />
      </linearGradient>
    </defs>
    <!-- bentuk perisai -->
    <path d="M32 4 C46 4 60 10 60 24 v18 c0 14-12 24-28 30C16 66 4 56 4 42V24C4 10 18 4 32 4z"
          fill="url(#djplogo)" stroke="white" stroke-width="2"/>
    <!-- garis diagonal -->
    <path d="M16 26 l32 0 M16 34 l32 0 M16 42 l24 0"
          stroke="white" stroke-width="3" stroke-linecap="round"/>
  </svg>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1.8, 1])
with col1:
    st.markdown("<h1>Incident Labeling • DJP Prototype</h1>", unsafe_allow_html=True)
    st.markdown("<p>Kelola dataset tiket, lakukan preprocessing, dan eksplorasi klaster untuk insight yang dapat ditindaklanjuti.</p>", unsafe_allow_html=True)
    st.write("**Aksi cepat:**")

    # Tombol aksi cepat bergaya chip
    st.markdown('<div class="link-chips">', unsafe_allow_html=True)
    st.link_button("Upload Dataset", "pages/upload_dataset.py", icon=":material/upload_file:", type="secondary")
    st.link_button("Preprocessing", "pages/preprocessing.py", icon=":material/cleaning_services:", type="secondary")
    st.link_button("Cluster Dashboard", "pages/cluster_dashboard.py", icon=":material/analytics:", type="secondary")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Ilustrasi SVG bergaya DJP
    st.markdown(f"""
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
      <!-- perisai kecil -->
      <g transform="translate(190,28)">
        <path d="M20 0 C40 0 60 10 60 28 v22 c0 18-20 30-40 36c-20-6-40-18-40-36V28C-0 10 20 0 20 0z" fill="#ffffff" opacity="0.9"/>
        <rect x="10" y="16" width="20" height="6" rx="3" fill="{DJP_BLUE}"/>
        <rect x="10" y="28" width="28" height="6" rx="3" fill="{DJP_GOLD}"/>
        <rect x="10" y="40" width="24" height="6" rx="3" fill="{DJP_BLUE}"/>
      </g>
    </svg>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- KPI SECTION ---
df = st.session_state.get("df")
n_tickets  = len(df) if df is not None else 0
n_clusters = st.session_state.get("n_clusters", 0)
n_outliers = st.session_state.get("n_outliers", 0)
last_update = st.session_state.get("last_update")

st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
for label, value in [
    ("Total Ticket", f"{n_tickets:,}"),
    ("Jumlah Cluster", n_clusters if n_clusters else "—"),
    ("Outlier (%)", f"{(n_outliers/n_tickets*100):.1f}%" if n_tickets and n_outliers else "—"),
    ("Terakhir diproses", last_update or "—"),
]:
    st.markdown(
        f'<div class="kpi-card">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- ALUR KERJA / GUIDE ---
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("Alur kerja singkat")
st.markdown("""
1. :material/upload_file: **Upload Dataset** di *Data Processing › Upload Dataset* (CSV/Excel).  
2. :material/cleaning_services: **Preprocessing** – normalisasi teks, stopword, stemming.  
3. :material/analytics: **Clustering & Labeling** – buka *Dashboard › Cluster Dashboard*.  
4. :material/insights: **Executive Summary** – ringkasan otomatis untuk manajemen.
""")
st.markdown('</div>', unsafe_allow_html=True)

# --- CUPLIKAN DATASET ---
if df is None:
    st.info("Belum ada dataset yang dimuat. Mulai dengan **Upload Dataset** terlebih dahulu.")
else:
    with st.expander("Cuplikan 5 baris pertama dataset"):
        st.dataframe(df.head())

# --- FOOTER ---
st.markdown("---")
st.caption(f"© {datetime.now().year} • Prototype internal • Tema: DJP (biru–emas)")
