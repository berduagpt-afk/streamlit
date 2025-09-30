# pages/dashboard.py
import streamlit as st
from datetime import datetime

# --- Guard: hanya untuk user yang login ---
if not st.session_state.get("logged_in", False):
    st.error("Silakan login dulu untuk mengakses halaman ini.")
    st.stop()

# --- Styling ringan ---
st.markdown("""
<style>
.hero {
  padding: 2rem 1.5rem; border-radius: 18px;
  background:
    radial-gradient(1200px 500px at 10% -10%, #fde68a66, transparent),
    radial-gradient(800px 400px at 90% 0%, #a7f3d066, transparent),
    linear-gradient(180deg, #ffffff, #f8fafc);
  border: 1px solid #e5e7eb;
}
.kpi-card{ padding:1rem; border:1px solid #e5e7eb; border-radius:16px; background:#fff; }
.muted { color:#6b7280; font-size:0.95rem; }
.big   { font-size:2rem; font-weight:800; line-height:1.1; margin-bottom:.25rem;}
.cta a { margin-right:.5rem; }
</style>
""", unsafe_allow_html=True)

# --- HERO ---
st.markdown('<div class="hero">', unsafe_allow_html=True)
col1, col2 = st.columns([2,1])
with col1:
    st.markdown('<div class="big">Incident Labeling Prototype</div>', unsafe_allow_html=True)
    st.markdown('<p class="muted">Kelola dataset tiket, lakukan preprocessing, dan eksplorasi klaster secara interaktif.</p>', unsafe_allow_html=True)
    st.write(":material/bolt: **Aksi cepat**")
    st.page_link("pages/upload_dataset.py", label="Upload Dataset", icon=":material/upload_file:")
    st.page_link("pages/preprocessing.py", label="Preprocessing Data", icon=":material/cleaning_services:")
    st.page_link("pages/cluster_dashboard.py", label="Lihat Cluster Dashboard", icon=":material/analytics:")
with col2:
    # Ilustrasi SVG inline (tidak perlu file terpisah)
    st.markdown("""
    <svg viewBox="0 0 240 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;">
      <defs>
        <linearGradient id="g1" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0%" stop-color="#22c55e" stop-opacity="0.9"/>
          <stop offset="100%" stop-color="#06b6d4" stop-opacity="0.9"/>
        </linearGradient>
        <linearGradient id="g2" x1="1" x2="0" y1="0" y2="1">
          <stop offset="0%" stop-color="#f97316" stop-opacity="0.9"/>
          <stop offset="100%" stop-color="#fde047" stop-opacity="0.9"/>
        </linearGradient>
        <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="8" stdDeviation="12" flood-opacity="0.2"/>
        </filter>
      </defs>
      <rect x="10" y="20" width="220" height="140" rx="16" fill="url(#g1)" filter="url(#shadow)"/>
      <g transform="translate(28,38)">
        <rect x="0" y="0"  width="184" height="20" rx="6" fill="#ffffff" opacity="0.9"/>
        <rect x="0" y="30" width="144" height="20" rx="6" fill="#ffffff" opacity="0.9"/>
        <rect x="0" y="60" width="164" height="20" rx="6" fill="#ffffff" opacity="0.9"/>
        <circle cx="170" cy="10" r="6" fill="url(#g2)"/>
        <circle cx="150" cy="40" r="6" fill="url(#g2)"/>
        <circle cx="180" cy="70" r="6" fill="url(#g2)"/>
      </g>
    </svg>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- KPI singkat (pakai data di session_state kalau ada) ---
df = st.session_state.get("df")
n_tickets  = len(df) if df is not None else 0
n_clusters = st.session_state.get("n_clusters", 0)
n_outliers = st.session_state.get("n_outliers", 0)
last_update = st.session_state.get("last_update")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric("Total Ticket", f"{n_tickets:,}")
    st.markdown('</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric("Jumlah Cluster", n_clusters if n_clusters else "—")
    st.markdown('</div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    pct_out = f"{(n_outliers/n_tickets*100):.1f}%" if n_tickets and n_outliers else "—"
    st.metric("Outlier (%)", pct_out)
    st.markdown('</div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric("Terakhir diproses", last_update or "—")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Quick guide ---
st.subheader("Mulai cepat")
st.markdown("""
1. :material/upload_file: **Upload Dataset** di menu *Data Processing › Upload Dataset* (CSV/Excel).
2. :material/cleaning_services: **Preprocessing** – normalisasi teks, stopword, stemming.
3. :material/analytics: **Clustering & Labeling** – buka *Dashboard › Cluster Dashboard*.
4. :material/insights: **Executive Summary** – ringkasan otomatis untuk manajemen.
""")

# Tampilkan cuplikan data bila sudah ada
if df is None:
    st.info("Belum ada dataset yang dimuat. Mulai dengan **Upload Dataset** terlebih dahulu.")
else:
    with st.expander("Cuplikan 5 baris pertama dataset"):
        st.dataframe(df.head())

# --- Footer kecil ---
st.markdown("---")
st.caption(f"© {datetime.now().year} • Incident Labeling Prototype • Build {st.__version__}")
