import streamlit as st

# --- Konfigurasi Page ---
st.set_page_config(page_title="Incident Labeling Prototype",
                   page_icon=":material/bug_report:",
                   layout="wide")

# --- Session State ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# --- Auth Handlers ---
def do_login():
    st.session_state.logged_in = True

def do_logout():
    st.session_state.logged_in = False

# --- Pages (functions) ---
def login():
    st.title("Masuk")
    with st.form("login_form", clear_on_submit=False):
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        submitted = st.form_submit_button("Log in", use_container_width=True)
        if submitted:
            # TODO: validasi kredensial (hash/ENV/dll) – untuk prototipe langsung masuk
            do_login()

def logout():
    st.title("Akun")
    st.write(f"Anda masuk sebagai: `{st.session_state.get('username','user')}`")
    if st.button("Log out", type="primary", use_container_width=True):
        do_logout()

# --- Page registry ---
login_page = st.Page(login, title="Log in", icon=":material/login:")
logout_page = st.Page(logout, title="Log out", icon=":material/logout:")

dashboard = st.Page(
    "pages/dashboard.py", title="Dashboard", icon=":material/dashboard:", default=True
)
upload_dataset = st.Page("pages/upload_dataset.py", title="Upload Dataset", icon=":material/upload_file:")
preprocessing = st.Page("pages/preprocessing.py", title="Preprocessing Data", icon=":material/cleaning_services:")

analisis_sintaksis = st.Page("pages/analisis_sintaksis.py", title="Analisis Sintaksis", icon=":material/123:")
analisis_semantik = st.Page("pages/analisis_semantik.py", title="Analisis Semantik", icon=":material/psychology:")

cluster_dashboard = st.Page("pages/cluster_dashboard.py", title="Cluster Dashboard", icon=":material/analytics:")
exec_summary = st.Page("pages/exec_summary.py", title="Executive Summary", icon=":material/insights:")

# --- Navigation ---
if st.session_state.logged_in:
    pg = st.navigation(
        {
            "Home": [dashboard],
            "Data Processing": [upload_dataset, preprocessing],
            "Analisis": [analisis_sintaksis, analisis_semantik],
            "Dashboard": [cluster_dashboard, exec_summary],
            "Account": [logout_page],
        }
    )
else:
    pg = st.navigation([login_page])

pg.run()
