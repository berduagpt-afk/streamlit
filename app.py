import streamlit as st
from packaging import version

# --- Konfigurasi Page (HANYA di app.py) ---
st.set_page_config(
    page_title="Incident Labeling Prototype",
    page_icon=":material/bug_report:",
    layout="wide",
)

# --- Cek versi Streamlit untuk API st.Page/st.navigation ---
MIN_VER = "1.33.0"  # aman untuk Page API; sesuaikan jika perlu
if version.parse(st.__version__) < version.parse(MIN_VER):
    st.error(
        f"Versi Streamlit kamu {st.__version__}. "
        f"Fitur multipage (st.Page/st.navigation) butuh >= {MIN_VER}. "
        "Jalankan:  pip install -U streamlit"
    )
    st.stop()

# --- Session State defaults ---
ss = st.session_state
ss.setdefault("logged_in", False)
ss.setdefault("login_attempts", 0)

# --- Auth Helpers ---
def check_credentials(username: str, password: str) -> bool:
    """
    Validasi sederhana via st.secrets.
    Tambahkan di .streamlit/secrets.toml:
    [auth]
    username="admin"
    password="123"
    """
    try:
        u = st.secrets["auth"]["username"]
        p = st.secrets["auth"]["password"]
    except Exception:
        # fallback prototipe (TIDAK untuk produksi)
        u, p = "admin", "admin"
    return username == u and password == p

def do_login(username: str):
    ss.logged_in = True
    ss.auth_user = username  # <- JANGAN tulis ke 'username' karena itu dipakai widget
    # opsional: bersihkan nilai widget supaya tidak nyangkut
    for k in ("login_username", "login_password"):
        if k in ss: del ss[k]
    st.rerun()

def do_logout():
    ss.logged_in = False
    for k in ("auth_user", "login_username", "login_password"):
        if k in ss: del ss[k]
    st.rerun()

# --- Pages (functions) ---
def login():
    st.title("Masuk")
    with st.form("login_form", clear_on_submit=False, border=True):

        username = st.text_input("Username", key="login_username", autocomplete="username")
        password = st.text_input("Password", type="password", key="login_password", autocomplete="current-password")

        col1, col2 = st.columns([1,1])
        with col1:
            submitted = st.form_submit_button("Log in", use_container_width=True, type="primary")
        with col2:
            st.form_submit_button("Reset", use_container_width=True)

        if submitted:
            if check_credentials(username, password):
                ss.login_attempts = 0
                do_login(username)
            else:
                ss.login_attempts += 1
                st.error("Username atau password salah.")
                if ss.login_attempts >= 5:
                    st.warning("Terlalu banyak percobaan. Coba lagi nanti.")
                    st.stop()

def logout():
    st.title("Akun")
    st.write(f"Anda masuk sebagai: `{ss.get('username','user')}`")
    if st.button("Log out", type="primary", use_container_width=True):
        do_logout()

# --- Page registry ---
login_page = st.Page(login, title="Log in", icon=":material/login:")
logout_page = st.Page(logout, title="Log out", icon=":material/logout:")

home = st.Page("pages/home.py", title="Home", icon=":material/dashboard:", default=True)
upload_dataset = st.Page("pages/upload_dataset.py", title="Upload Dataset", icon=":material/upload_file:")
preprocessing = st.Page("pages/preprocessing.py", title="Preprocessing Data", icon=":material/cleaning_services:")
analisis_sintaksis = st.Page("pages/analisis_sintaksis.py", title="Analisis Sintaksis", icon=":material/123:")
analisis_semantik = st.Page("pages/analisis_semantik.py", title="Analisis Semantik", icon=":material/psychology:")
cluster_dashboard = st.Page("pages/cluster_dashboard.py", title="Cluster Dashboard", icon=":material/analytics:")
exec_summary = st.Page("pages/exec_summary.py", title="Executive Summary", icon=":material/insights:")

# --- Navigation ---
if ss.logged_in:
    pg = st.navigation(
        {
            "Home": [home],
            "Data Processing": [upload_dataset, preprocessing],
            "Analisis": [analisis_sintaksis, analisis_semantik],
            "Dashboard": [cluster_dashboard, exec_summary],
            "Account": [logout_page],
        }
    )
else:
    pg = st.navigation([login_page])

pg.run()
