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
# Data Preparation
load_dataset = st.Page("pages/load_dataset.py", title="Load Dataset", icon=":material/database:")
# Text Processing
data_normalization = st.Page("pages/data_preparation/text_processing/data_normalization.py", title="Data Normalization", icon=":material/cleaning_services:")
# Pendekatan Sintaksis
sintaksis_preprocessing = st.Page("pages/data_preparation/text_processing/sintaksis_preprocessing.py", title="Sintaksis Preprocessing", icon=":material/cleaning_services:")
semantik_preprocessing = st.Page("pages/data_preparation/text_processing/semantik_preprocessing.py", title="Semantik Preprocessing", icon=":material/cleaning_services:")
# Feature Extraction 
viewer_sintaksis_tfidf = st.Page("pages/data_preparation/feature_extraction/viewer_sintaksis_tfidf.py", title="Viewer TF-IDF", icon=":material/hub:")
viewer_semantik_bert = st.Page("pages/data_preparation/feature_extraction/viewer_semantik_bert.py", title="Viewer BERT", icon=":material/hub:")
# Pendekatan Semantik
tfidf_extraction = st.Page("pages/tfidf_extraction.py", title="TF-IDF Extraction", icon=":material/text_snippet:")
# Modeling
# Sintaksis
modeling_sintaksis_pra_threshold = st.Page("pages/modeling/modeling_sintaksis_pra_threshold.py", title="Cosine Similarity Pra Threshold", icon=":material/hub:")
modeling_sintaksis_viewer = st.Page("pages/modeling/modeling_sintaksis_viewer.py", title="Cosine Similarity Threshold", icon=":material/hub:")
# Semantik
# modeling_semantik_embedding = st.Page("pages/modeling/modeling_semantik_embedding.py", title="Modeling Semantik Embedding", icon=":material/hub:")
# modeling_semantik_hdbscan = st.Page("pages/modeling/modeling_semantik_hdbscan.py", title="Modeling Semantik HDBSCAN", icon=":material/hub:")
modeling_semantik_hdbscan_viewer = st.Page("pages/modeling/modeling_semantik_hdbscan_viewer.py", title="HBSCAN Viewer", icon=":material/hub:")
# Evaluasi Modeling Sintaksis dan Semantik
modeling_evaluasi_dbcv = st.Page("pages/modeling/modeling_evaluasi_dbcv.py", title="Evaluasi DBCV", icon=":material/hub:")
modeling_evaluasi_sc_dbi_sintaksis = st.Page("pages/modeling/modeling_evaluasi_sc_dbi_sintaksis.py", title="Evaluasi Sintaksis Silhouette Score & DBI", icon=":material/hub:")
modeling_evaluasi_sc_dbi_semantik = st.Page("pages/modeling/modeling_evaluasi_sc_dbi_semantik.py", title="Evaluasi Semantik Silhouette Score & DBI", icon=":material/hub:")
modeling_evaluasi = st.Page("pages/modeling/modeling_evaluasi_final.py", title="Evaluasi Modeling", icon=":material/hub:")
modeling_evaluasi_temporal = st.Page("pages/modeling/modeling_evaluasi_temporal.py", title="Evaluasi Temporal", icon=":material/hub:")
modeling_evaluasi_temporal_sintaksis = st.Page("pages/modeling/modeling_evaluasi_temporal_sintaksis.py", title="Evaluasi Temporal Sintaksis", icon=":material/hub:")
modeling_evaluasi_temporal_semantik = st.Page("pages/modeling/modeling_evaluasi_temporal_semantik.py", title="Evaluasi Temporal Semantik", icon=":material/hub:")
modeling_evaluasi_compare = st.Page("pages/modeling/modeling_evaluasi_compare.py", title="Perbandingan Hasil Evaluasi Temporal Sintaksis vs Semantik", icon=":material/hub:")
modeling_hasil_evaluasi = st.Page("pages/modeling/modeling_hasil_evaluasi.py", title="Hasil Evaluasi Modeling", icon=":material/hub:")
# Labeling
labeling_temporal = st.Page("pages/labeling/labeling_temporal_rule.py", title="Labeling Tiket Insiden", icon=":material/label:")
labeling_topic_cluster = st.Page("pages/labeling/labeling_topic_cluster.py", title="Labeling Topic Cluster", icon=":material/label:")
labeling_viewer_topic_cluster = st.Page("pages/labeling/viewer_topic_cluster.py", title="Viewer Topic Cluster", icon=":material/label:")
labeling_utility = st.Page("pages/labeling/labeling_utility.py", title="Labeling Utility", icon=":material/label:")
# Prediction
load_dataset_prediction = st.Page("pages/prediction/load_dataset_prediction.py", title="Load Dataset Prediction", icon=":material/hub:")
# Evaluation
evaluation_manual_labeling = st.Page("pages/evaluation/evaluation_manual_labeling.py", title="Evaluation Manual Labeling", icon=":material/hub:")
# Modeling - 1
modeling_sintaksis = st.Page("pages/modeling/modeling_sintaksis.py", title="Modeling Sintaksis", icon=":material/hub:")
modeling_sintaksis_visualisasi = st.Page("pages/modeling/modeling_sintaksis_visualisasi.py", title="Modeling Sintaksis Visualisasi", icon=":material/hub:")
modeling_sintaksis_timeline = st.Page("pages/modeling/modeling_sintaksis_timeline.py", title="Modeling Sintaksis Timeline", icon=":material/hub:")
sintaksis_cosine = st.Page("pages/modeling_sintaksis_cosine.py", title="Sintaksis Cosine", icon=":material/hub:")
sintaksis_summary = st.Page("pages/modeling_sintaksis_summary.py", title="Sintaksis Summary", icon=":material/hub:")
evaluasi_tsne = st.Page("pages/modeling_visualisasi_tsne.py", title="Evaluasi t-SNE", icon=":material/hub:")
evaluasi_cosine_treshold = st.Page("pages/modeling_evaluasi_cosine_threshold.py", title="Evaluasi Cosine Threshold", icon=":material/hub:")
modeling_evaluasi_1 = st.Page("pages/modeling_evaluasi.py", title="Modeling Evaluasi", icon=":material/hub:")
# Modeling Sintaksis TF-IDF Unigram
modeling_sintaksis_tfidf_unigram = st.Page("pages/modeling_sintaksis_tfidf_unigram_viewer.py", title="Modeling Sintaksis TF-IDF Unigram", icon=":material/hub:")
# Evaluasi
evaluation_sintaksis = st.Page("pages/evaluation_sintaksis.py", title="Evaluation Sintaksis", icon=":material/hub:")
# Analisis
modeling_semantik_temporal = st.Page("pages/modeling/modeling_semantik_temporal.py", title="Modeling Semantik Temporal", icon=":material/hub:")
modeling_sintaksis_temporal = st.Page("pages/modeling/modeling_sintaksis_temporal.py", title="Modeling Sintaksis Temporal", icon=":material/hub:")
analisis_sintaksis = st.Page("pages/analisis_sintaksis.py", title="Analisis Sintaksis", icon=":material/123:")
analisis_semantik = st.Page("pages/analisis_semantik.py", title="Analisis Semantik", icon=":material/psychology:")
cluster_dashboard = st.Page("pages/cluster_dashboard.py", title="Cluster Dashboard", icon=":material/bubble_chart:")
exec_summary = st.Page("pages/exec_summary.py", title="Executive Summary", icon=":material/insights:")
reporting_summary = st.Page("pages/reporting_summary.py", title="Reporting Summary", icon=":material/stacked_line_chart:")
# Eksperimental
venn_diagram = st.Page("pages/eksperimental/venn_sintaksis_vs_semantik.py", title="Venn Diagram", icon=":material/thumb_up:")
semantik_chart = st.Page("pages/eksperimental/modeling_semantic_hdbscan_charts.py", title="Semantik Chart", icon=":material/thumb_up:")

# ======================================================
# 🧭 NAVIGASI BERDASARKAN STATUS LOGIN
# ======================================================
if ss.logged_in:
    pg = st.navigation(
        {
            "🏠 Home": [home],
            "📂 Data Understanding": [upload_dataset, statistik_dataset, verifikasi_data],
            # "📂 Generate Data": [generate_dataset],
            "⚙️ Data Preparation": [load_dataset],
            "🧹 Text Processing": [data_normalization, sintaksis_preprocessing, semantik_preprocessing],
            "🧹 Feature Extraction": [viewer_sintaksis_tfidf, viewer_semantik_bert],
            "📂 Modeling Sintaksis": [modeling_sintaksis_pra_threshold, modeling_sintaksis_viewer],
            "📂 Modeling Semantik": [modeling_semantik_hdbscan_viewer],
            "📂 Modeling Evaluasi Sintaksis & Semantik": [modeling_evaluasi_dbcv, modeling_evaluasi_sc_dbi_sintaksis, modeling_evaluasi_sc_dbi_semantik, modeling_evaluasi, modeling_evaluasi_temporal, modeling_evaluasi_temporal_sintaksis, modeling_evaluasi_temporal_semantik, modeling_evaluasi_compare, modeling_hasil_evaluasi],
            "📂 Labeling": [labeling_temporal, labeling_utility, labeling_viewer_topic_cluster],
            "📂 Prediction": [load_dataset_prediction],
            "📂 Evaluasi": [evaluation_manual_labeling],
            "📂 Modeling - 1": [sintaksis_cosine, sintaksis_summary, evaluasi_tsne, evaluasi_cosine_treshold, modeling_evaluasi_1, modeling_sintaksis, modeling_sintaksis_visualisasi, modeling_sintaksis_timeline ],
            "📂 Modeling Sintaksis - 1": [modeling_sintaksis_tfidf_unigram],
            "📂 Evaluation": [evaluation_sintaksis],
            "🧮 Analisis": [tfidf_extraction, analisis_sintaksis, analisis_semantik, modeling_sintaksis_temporal, modeling_semantik_temporal],
            "🧮 Eksperimental": [venn_diagram, semantik_chart],
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
