import streamlit as st
from packaging import version

# ======================================================
# 🧭 KONFIGURASI HALAMAN (WAJIB pertama)
# ======================================================
st.set_page_config(
    page_title="Incident Labeling Prototype • DJP",
    page_icon=":material/bug_report:",
    layout="wide",
)

# ======================================================
# 🧩 CEK VERSI STREAMLIT
# ======================================================
MIN_VER = "1.33.0"  # Dibutuhkan untuk API st.Page / st.navigation
if version.parse(st.__version__) < version.parse(MIN_VER):
    st.error(
        f"Versi Streamlit kamu {st.__version__}. "
        f"Fitur multipage (st.Page/st.navigation) butuh >= {MIN_VER}. "
        "Jalankan:  pip install -U streamlit"
    )
    st.stop()

# ======================================================
# 🔐 SESSION STATE DEFAULTS
# ======================================================
ss = st.session_state
ss.setdefault("logged_in", False)
ss.setdefault("login_attempts", 0)

# ======================================================
# 🧠 AUTHENTIKASI
# ======================================================
def check_credentials(username: str, password: str) -> bool:
    """
    Validasi user via .streamlit/secrets.toml
    Tambahkan di file:
    [auth]
    username="admin"
    password="123"
    """
    try:
        u = st.secrets["auth"]["username"]
        p = st.secrets["auth"]["password"]
    except Exception:
        # fallback default (untuk prototipe)
        u, p = "admin", "admin"
    return username == u and password == p


def do_login(username: str):
    ss.logged_in = True
    ss.auth_user = username
    # Bersihkan form login agar tidak nyangkut
    for k in ("login_username", "login_password"):
        ss.pop(k, None)
    st.rerun()


def do_logout():
    ss.logged_in = False
    for k in ("auth_user", "login_username", "login_password"):
        ss.pop(k, None)
    st.rerun()

# ======================================================
# 🧾 HALAMAN LOGIN & LOGOUT
# ======================================================
def login():
    st.title("🔐 Masuk ke Sistem")
    with st.form("login_form", clear_on_submit=False, border=True):
        username = st.text_input("Username", key="login_username", autocomplete="username")
        password = st.text_input("Password", type="password", key="login_password", autocomplete="current-password")

        c1, c2 = st.columns([1, 1])
        submitted = c1.form_submit_button("Log in", type="primary", use_container_width=True)
        c2.form_submit_button("Reset", use_container_width=True)

        if submitted:
            if check_credentials(username, password):
                ss.login_attempts = 0
                do_login(username)
            else:
                ss.login_attempts += 1
                st.error("Username atau password salah.")
                if ss.login_attempts >= 5:
                    st.warning("Terlalu banyak percobaan login. Coba lagi nanti.")
                    st.stop()


def logout():
    st.title("👤 Akun Pengguna")
    st.write(f"Anda masuk sebagai: `{ss.get('auth_user', 'user')}`")
    if st.button("Log out", type="primary", use_container_width=True):
        do_logout()

# ======================================================
# 📚 REGISTER HALAMAN
# ======================================================
# Autentikasi
login_page = st.Page(login, title="Log in", icon=":material/login:")
logout_page = st.Page(logout, title="Log out", icon=":material/logout:")

# Halaman utama dan fitur
home = st.Page("pages/home.py", title="Home", icon=":material/home:", default=True)
# data_understanding = st.Page("pages/data_understanding.py", title="Upload Dataset", icon=":material/upload_file:")
upload_dataset = st.Page("pages/upload_dataset.py", title="Upload Dataset", icon=":material/upload_file:")
statistik_dataset = st.Page("pages/statistik_dataset.py", title="Statistik Dataset", icon=":material/monitoring:")
verifikasi_data = st.Page("pages/verifikasi_data.py", title="Verifikasi Kualitas Data", icon=":material/task_alt:")
# generate_dataset = st.Page("pages/generate_dataset.py", title="Generate Dataset", icon=":material/upload_file:")
load_dataset = st.Page("pages/load_dataset.py", title="Load Dataset", icon=":material/database:")
preprocessing = st.Page("pages/preprocessing.py", title="Preprocessing Data", icon=":material/cleaning_services:")
tfidf_extraction = st.Page("pages/tfidf_extraction.py", title="TF-IDF Extraction", icon=":material/text_snippet:")
# Modelling
sintaksis_cosine = st.Page("pages/modeling_sintaksis_cosine.py", title="Sintaksis Cosine", icon=":material/hub:")
sintaksis_clustering = st.Page("pages/modeling_sintaksis_clustering.py", title="Sintaksis Clustering", icon=":material/hub:")
evaluasi_cosine_treshold = st.Page("pages/modeling_evaluasi_cosine_threshold.py", title="Evaluasi Cosine Threshold", icon=":material/hub:")
# Evaluasi
evaluation_sintaksis = st.Page("pages/evaluation_sintaksis.py", title="Evaluation Sintaksis", icon=":material/hub:")
# Analisis
analisis_sintaksis = st.Page("pages/analisis_sintaksis.py", title="Analisis Sintaksis", icon=":material/123:")
analisis_semantik = st.Page("pages/analisis_semantik.py", title="Analisis Semantik", icon=":material/psychology:")
cluster_dashboard = st.Page("pages/cluster_dashboard.py", title="Cluster Dashboard", icon=":material/bubble_chart:")
exec_summary = st.Page("pages/exec_summary.py", title="Executive Summary", icon=":material/insights:")
reporting_summary = st.Page("pages/reporting_summary.py", title="Reporting Summary", icon=":material/stacked_line_chart:")

# ======================================================
# 🧭 NAVIGASI BERDASARKAN STATUS LOGIN
# ======================================================
if ss.logged_in:
    pg = st.navigation(
        {
            "🏠 Home": [home],
            "📂 Data Understanding": [upload_dataset, statistik_dataset, verifikasi_data],
            # "📂 Generate Data": [generate_dataset],
            "📂 Data Processing": [load_dataset, preprocessing],
            "📂 Modeling": [sintaksis_cosine, sintaksis_clustering, evaluasi_cosine_treshold],
            "📂 Evaluation": [evaluation_sintaksis],
            "🧮 Analisis": [tfidf_extraction, analisis_sintaksis, analisis_semantik],
            "📊 Dashboard": [cluster_dashboard, exec_summary, reporting_summary],
            "👤 Account": [logout_page],
        }
    )
else:
    pg = st.navigation([login_page])

# ======================================================
# 🚀 JALANKAN NAVIGASI
# ======================================================
pg.run()
