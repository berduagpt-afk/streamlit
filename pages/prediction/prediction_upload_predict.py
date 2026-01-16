# pages/prediction/prediction_upload_predict.py
# ============================================================
# 📤 Upload & Predict (Auto-generate temporal & cluster features)
# - Input upload minimal 7 kolom:
#   tgl_submit, incident_number, site, assignee, modul, sub_modul, detailed_decription
# - Sistem menurunkan fitur yang dibutuhkan model:
#   gap_days, n_member_cluster, n_episode_cluster, n_member_episode, window_days
# - Model dibaca dari file joblib: models/<modeling_id>_<model_name>.joblib
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
import joblib


# ============================================================
# 🔐 Guard Login
# ============================================================
if not st.session_state.get("logged_in", True):
    st.error("Silakan login terlebih dahulu.")
    st.stop()

# ============================================================
# 🔧 KONFIGURASI
# ============================================================
SCHEMA = "lasis_djp"
T_DATA = f"{SCHEMA}.dataset_supervised"
T_EVAL = f"{SCHEMA}.prediction_evaluation_results"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# upload minimal
REQUIRED_UPLOAD_COLS = [
    "tgl_submit",
    "incident_number",
    "site",
    "assignee",
    "modul",
    "sub_modul",
    "detailed_decription",
]

# fitur yang dibutuhkan model training (HARUS TERSEDIA SAAT PREDIKSI)
CAT_COLS = ["site", "modul", "sub_modul", "assignee"]
NUM_COLS = ["gap_days", "n_member_cluster", "n_episode_cluster", "n_member_episode", "window_days"]


# ============================================================
# 🔌 DB Connection (konsisten)
# ============================================================
@st.cache_resource(show_spinner=False)
def get_engine():
    cfg = st.secrets["connections"]["postgres"]
    return create_engine(
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}",
        pool_pre_ping=True,
    )


def read_df(engine, sql: str, params: Optional[dict] = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


engine = get_engine()


# ============================================================
# 🧠 Model Loader
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model(modeling_id: str, model_name: str):
    path = MODEL_DIR / f"{modeling_id}_{model_name}.joblib"
    if not path.exists():
        st.error(f"File model tidak ditemukan: {path}")
        st.stop()
    return joblib.load(path)


# ============================================================
# 🧱 Hist Feature Cache (agregasi dari dataset_supervised)
# ============================================================
@st.cache_data(ttl=300)
def load_hist_aggregates(
    _engine,
    jenis_pendekatan: str,
    modeling_id: str,
    window_days: int,
    time_col: str,
    include_noise: bool,
    eligible_rule: str,
    split_name: str,
) -> dict:
    """
    Mengambil agregasi historis untuk imputasi fitur.
    Fallback bertingkat:
      (site, modul, sub_modul, assignee) → (site, modul, sub_modul) → (modul, sub_modul) → global
    """

    sql = f"""
    SELECT
      site, modul, sub_modul, assignee,
      COALESCE(tgl_submit, event_time) AS t_event,
      COALESCE(n_member_cluster, 1)::double precision AS n_member_cluster,
      COALESCE(n_episode_cluster, 1)::double precision AS n_episode_cluster,
      COALESCE(n_member_episode, 1)::double precision AS n_member_episode
    FROM {T_DATA}
    WHERE jenis_pendekatan = :jenis
      AND modeling_id = CAST(:modeling_id AS uuid)
      AND window_days = :window_days
      AND time_col = :time_col
      AND COALESCE(include_noise,false) = :include_noise
      AND COALESCE(eligible_rule,'') = :eligible_rule
      AND COALESCE(split_name,'') = :split_name
    """

    # ✅ CAST ke native python (hindari numpy/UUID)
    params = {
        "jenis": str(jenis_pendekatan),
        "modeling_id": str(modeling_id),
        "window_days": int(window_days),
        "time_col": str(time_col),
        "include_noise": bool(include_noise),
        "eligible_rule": str(eligible_rule),
        "split_name": str(split_name),
    }

    df = pd.read_sql(text(sql), _engine, params=params)

    if df.empty:
        return {
            "lvl4": pd.DataFrame(),
            "lvl3": pd.DataFrame(),
            "lvl2": pd.DataFrame(),
            "global": {
                "n_member_cluster": 1.0,
                "n_episode_cluster": 1.0,
                "n_member_episode": 1.0,
                "last_time": pd.NaT,
            },
        }

    # normalisasi kategori
    for c in ["site", "modul", "sub_modul", "assignee"]:
        df[c] = df[c].fillna("UNKNOWN").astype(str)

    df["t_event"] = pd.to_datetime(df["t_event"], errors="coerce")

    def agg_by(keys):
        g = (
            df.groupby(keys, dropna=False)
            .agg(
                n_member_cluster=("n_member_cluster", "mean"),
                n_episode_cluster=("n_episode_cluster", "mean"),
                n_member_episode=("n_member_episode", "mean"),
                last_time=("t_event", "max"),
            )
            .reset_index()
        )
        return g

    lvl4 = agg_by(["site", "modul", "sub_modul", "assignee"])
    lvl3 = agg_by(["site", "modul", "sub_modul"])
    lvl2 = agg_by(["modul", "sub_modul"])

    global_stats = {
        "n_member_cluster": float(df["n_member_cluster"].mean()) if df["n_member_cluster"].notna().any() else 1.0,
        "n_episode_cluster": float(df["n_episode_cluster"].mean()) if df["n_episode_cluster"].notna().any() else 1.0,
        "n_member_episode": float(df["n_member_episode"].mean()) if df["n_member_episode"].notna().any() else 1.0,
        "last_time": df["t_event"].max(),
    }

    return {"lvl4": lvl4, "lvl3": lvl3, "lvl2": lvl2, "global": global_stats}


def enrich_features_from_history(df_upload: pd.DataFrame, hist: dict, window_days: int) -> pd.DataFrame:
    """
    Menghasilkan fitur numerik yang dibutuhkan model dengan fallback bertingkat.
    """
    df = df_upload.copy()

    df["tgl_submit"] = pd.to_datetime(df["tgl_submit"], errors="coerce")
    for c in CAT_COLS:
        df[c] = df[c].fillna("UNKNOWN").astype(str)

    g = hist["global"]
    g_nmc = float(g.get("n_member_cluster", 1.0))
    g_nec = float(g.get("n_episode_cluster", 1.0))
    g_nme = float(g.get("n_member_episode", 1.0))

    # lvl4
    if not hist["lvl4"].empty:
        df = df.merge(hist["lvl4"], on=["site", "modul", "sub_modul", "assignee"], how="left", suffixes=("", "_h4"))
    else:
        df["n_member_cluster"] = np.nan
        df["n_episode_cluster"] = np.nan
        df["n_member_episode"] = np.nan
        df["last_time"] = pd.NaT

    # lvl3 fallback
    if not hist["lvl3"].empty:
        m3 = df.merge(hist["lvl3"], on=["site", "modul", "sub_modul"], how="left", suffixes=("", "_h3"))
        for col in ["n_member_cluster", "n_episode_cluster", "n_member_episode", "last_time"]:
            df[col] = df[col].where(df[col].notna(), m3[f"{col}_h3"])

    # lvl2 fallback
    if not hist["lvl2"].empty:
        m2 = df.merge(hist["lvl2"], on=["modul", "sub_modul"], how="left", suffixes=("", "_h2"))
        for col in ["n_member_cluster", "n_episode_cluster", "n_member_episode", "last_time"]:
            df[col] = df[col].where(df[col].notna(), m2[f"{col}_h2"])

    # global fallback
    df["n_member_cluster"] = pd.to_numeric(df["n_member_cluster"], errors="coerce").fillna(g_nmc)
    df["n_episode_cluster"] = pd.to_numeric(df["n_episode_cluster"], errors="coerce").fillna(g_nec)
    df["n_member_episode"] = pd.to_numeric(df["n_member_episode"], errors="coerce").fillna(g_nme)

    # gap_days
    df["gap_days"] = (
        (df["tgl_submit"] - pd.to_datetime(df["last_time"], errors="coerce"))
        .dt.total_seconds()
        .div(86400.0)
    )
    df["gap_days"] = df["gap_days"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["gap_days"] = df["gap_days"].clip(lower=0, upper=3650)

    df["window_days"] = int(window_days)

    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df


# ============================================================
# 🧭 UI
# ============================================================
st.title("📤 Upload & Predict – Prediksi Insiden Berulang")
st.caption(
    "Upload tiket baru (minimal 7 kolom). Sistem akan menurunkan fitur temporal & historis "
    "berdasarkan dataset_supervised untuk melakukan prediksi label_berulang."
)

# ============================================================
# 1) Upload
# ============================================================
st.subheader("1️⃣ Upload Data Tiket")

up = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx"])
if not up:
    st.info("Silakan upload file untuk melanjutkan.")
    st.stop()

df_raw = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)

st.markdown("**Preview data**")
st.dataframe(df_raw.head(), use_container_width=True)

missing = [c for c in REQUIRED_UPLOAD_COLS if c not in df_raw.columns]
if missing:
    st.error(f"Kolom wajib tidak ditemukan: {missing}")
    st.stop()

# ============================================================
# 2) Pilih model run (dari evaluation_results)
#    ✅ pakai DISTINCT ON agar stabil & bisa ambil run terbaru
# ============================================================
st.subheader("2️⃣ Pilih Model Prediktif")
df_models = read_df(
    engine,
    f"""
    SELECT DISTINCT ON (
        jenis_pendekatan,
        modeling_id,
        window_days,
        time_col,
        COALESCE(include_noise,false),
        COALESCE(eligible_rule,''),
        COALESCE(split_name,''),
        model_name
    )
        jenis_pendekatan,
        modeling_id,
        window_days,
        time_col,
        COALESCE(include_noise,false) AS include_noise,
        COALESCE(eligible_rule,'') AS eligible_rule,
        COALESCE(split_name,'') AS split_name,
        model_name,
        run_time
    FROM {T_EVAL}
    ORDER BY
        jenis_pendekatan,
        modeling_id,
        window_days,
        time_col,
        COALESCE(include_noise,false),
        COALESCE(eligible_rule,''),
        COALESCE(split_name,''),
        model_name,
        run_time DESC
    """
)

if df_models.empty:
    st.error(f"Belum ada evaluasi tersimpan di `{T_EVAL}`.")
    st.stop()

i = st.selectbox(
    "Pilih konfigurasi model",
    df_models.index,
    format_func=lambda k: (
        f"{df_models.loc[k,'model_name']} | "
        f"{df_models.loc[k,'jenis_pendekatan']} | "
        f"w={int(df_models.loc[k,'window_days'])} | "
        f"time_col={df_models.loc[k,'time_col']} | "
        f"split={df_models.loc[k,'split_name']} | "
        f"{df_models.loc[k,'run_time']}"
    )
)

sel = df_models.loc[i]

# ✅ cast ke native
model_name = str(sel["model_name"])
modeling_id = str(sel["modeling_id"])
window_days = int(sel["window_days"])

# ============================================================
# 3) Load model + Hist aggregates
# ============================================================
st.subheader("3️⃣ Prediksi")

with st.spinner("Memuat model & agregasi historis..."):
    model = load_model(modeling_id, model_name)

    hist = load_hist_aggregates(
        engine,
        jenis_pendekatan=str(sel["jenis_pendekatan"]),
        modeling_id=str(sel["modeling_id"]),
        window_days=int(sel["window_days"]),
        time_col=str(sel["time_col"]),
        include_noise=bool(sel["include_noise"]),
        eligible_rule=str(sel["eligible_rule"]),
        split_name=str(sel["split_name"]),
    )

# ============================================================
# 4) Enrich features
# ============================================================
with st.spinner("Menurunkan fitur temporal & historis (gap_days, ukuran cluster, episode)..."):
    df_enriched = enrich_features_from_history(df_raw, hist, window_days=window_days)

with st.expander("🩺 Diagnostics fitur turunan", expanded=False):
    show_cols = ["incident_number", "tgl_submit"] + CAT_COLS + NUM_COLS
    st.dataframe(df_enriched[show_cols].head(50), use_container_width=True)

# ============================================================
# 5) Predict
# ============================================================
X = df_enriched[CAT_COLS + NUM_COLS].copy()

for c in CAT_COLS:
    X[c] = X[c].fillna("UNKNOWN").astype(str)
for c in NUM_COLS:
    X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

with st.spinner("Melakukan prediksi..."):
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
        df_enriched["probabilitas_berulang"] = prob.astype(float)
    else:
        df_enriched["probabilitas_berulang"] = np.nan

    df_enriched["label_prediksi"] = model.predict(X).astype(int)

st.success("✅ Prediksi selesai.")

# ============================================================
# 6) Output
# ============================================================
st.subheader("4️⃣ Hasil Prediksi")

OUT_COLS = [
    "incident_number",
    "tgl_submit",
    "site",
    "assignee",
    "modul",
    "sub_modul",
    "label_prediksi",
    "probabilitas_berulang",
] + NUM_COLS

st.dataframe(df_enriched[OUT_COLS], use_container_width=True, height=520)

c1, c2, c3 = st.columns(3)
c1.metric("Jumlah tiket", f"{len(df_enriched):,}")
c2.metric("Prediksi Berulang (1)", f"{int((df_enriched['label_prediksi'] == 1).sum()):,}")
c3.metric("Prediksi Tidak Berulang (0)", f"{int((df_enriched['label_prediksi'] == 0).sum()):,}")

# ============================================================
# 7) Download
# ============================================================
st.subheader("5️⃣ Unduh Hasil")
csv = df_enriched[OUT_COLS].to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download hasil prediksi (CSV)",
    data=csv,
    file_name="hasil_prediksi_upload_insiden_berulang.csv",
    mime="text/csv",
)

with st.expander("📘 Catatan Metodologis", expanded=False):
    st.markdown(
        """
        - Data upload adalah tiket baru yang belum memiliki label kebenaran.
        - Sistem menurunkan fitur temporal & historis berdasarkan dataset_supervised (fallback bertingkat).
        - Jika tidak ditemukan histori yang relevan (cold-start), sistem menggunakan nilai default konservatif.
        """
    )
