# pages/prediction/prediction_upload_predict.py
# ============================================================
# 📤 Upload & Predict (FINAL)
# - Panel pemodelan & hasil evaluasi (RF/XGB/SVM) dari prediction_evaluation_results
# - Prediksi multi-model (tampilkan semua model, bukan hanya terbaik)
# - Fitur numerik upload diimputasi dari HISTORI SELURUH dataset_supervised (TANPA split_name)
#   -> agar lebih stabil dan mendekati fitur "asli" di database
# - Model dibaca dari joblib: models/<modeling_id>_<model_name>.joblib (support spasi/underscore)
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
import joblib


# ============================================================
# 🔐 Guard Login
# ============================================================
if not st.session_state.get("logged_in", False):
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

REQUIRED_BASE_COLS = ["tgl_submit", "incident_number", "site", "assignee", "modul", "sub_modul"]
TEXT_COL_CANDIDATES = ["detailed_description", "detailed_decription"]  # toleransi typo

# konsisten dengan evaluasi
CAT_COLS = ["site", "assignee", "modul", "sub_modul"]
NUM_COLS = ["gap_days", "n_member_cluster", "n_episode_cluster", "n_member_episode", "window_days"]


# ============================================================
# 🔌 DB Connection
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
# 🧠 Model Loader (support spasi/underscore)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model(modeling_id: str, model_name: str):
    p1 = MODEL_DIR / f"{modeling_id}_{model_name}.joblib"
    p2 = MODEL_DIR / f"{modeling_id}_{model_name.replace(' ', '_')}.joblib"
    if p1.exists():
        return joblib.load(p1)
    if p2.exists():
        return joblib.load(p2)
    raise FileNotFoundError(f"Model tidak ditemukan.\n- {p1}\n- {p2}")


# ============================================================
# 🧼 Normalisasi kategori upload
# ============================================================
def norm_cat(s: Any) -> str:
    if s is None:
        return "UNKNOWN"
    s = str(s).strip()
    return s if s else "UNKNOWN"


# ============================================================
# 🧱 Hist Feature Cache (agregasi dari dataset_supervised)
#     ✅ FINAL: histori untuk imputasi TIDAK memfilter split_name
# ============================================================
@st.cache_data(ttl=300)
def load_hist_aggregates_all_splits(
    _engine,
    jenis_pendekatan: str,
    modeling_id: str,
    window_days: int,
    time_col: str,
    include_noise: bool,
    eligible_rule: str,
) -> Dict[str, Any]:
    """
    Ambil agregasi historis untuk imputasi fitur (fallback bertingkat).
    HISTORI DIAMBIL DARI SELURUH dataset_supervised tanpa split_name.
    """

    sql = f"""
    SELECT
      site, assignee, modul, sub_modul,
      COALESCE(event_time, tgl_submit) AS t_event,
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
    """

    params = {
        "jenis": str(jenis_pendekatan),
        "modeling_id": str(modeling_id),
        "window_days": int(window_days),
        "time_col": str(time_col),
        "include_noise": bool(include_noise),
        "eligible_rule": str(eligible_rule),
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

    for c in ["site", "assignee", "modul", "sub_modul"]:
        df[c] = df[c].apply(norm_cat)

    df["t_event"] = pd.to_datetime(df["t_event"], errors="coerce")

    def agg_by(keys: List[str]) -> pd.DataFrame:
        return (
            df.groupby(keys, dropna=False)
            .agg(
                n_member_cluster=("n_member_cluster", "mean"),
                n_episode_cluster=("n_episode_cluster", "mean"),
                n_member_episode=("n_member_episode", "mean"),
                last_time=("t_event", "max"),
            )
            .reset_index()
        )

    lvl4 = agg_by(["site", "assignee", "modul", "sub_modul"])
    lvl3 = agg_by(["site", "modul", "sub_modul"])
    lvl2 = agg_by(["modul", "sub_modul"])

    global_stats = {
        "n_member_cluster": float(df["n_member_cluster"].mean()) if df["n_member_cluster"].notna().any() else 1.0,
        "n_episode_cluster": float(df["n_episode_cluster"].mean()) if df["n_episode_cluster"].notna().any() else 1.0,
        "n_member_episode": float(df["n_member_episode"].mean()) if df["n_member_episode"].notna().any() else 1.0,
        "last_time": df["t_event"].max(),
    }

    return {"lvl4": lvl4, "lvl3": lvl3, "lvl2": lvl2, "global": global_stats}


def enrich_features_from_history(df_upload: pd.DataFrame, hist: Dict[str, Any], window_days: int) -> pd.DataFrame:
    """
    Turunkan fitur numerik (proxy) dari histori.
    """
    df = df_upload.copy()

    df["tgl_submit"] = pd.to_datetime(df["tgl_submit"], errors="coerce")

    for c in CAT_COLS:
        df[c] = df[c].apply(norm_cat)

    g = hist["global"]
    g_nmc = float(g.get("n_member_cluster", 1.0))
    g_nec = float(g.get("n_episode_cluster", 1.0))
    g_nme = float(g.get("n_member_episode", 1.0))

    # lvl4
    if not hist["lvl4"].empty:
        df = df.merge(hist["lvl4"], on=["site", "assignee", "modul", "sub_modul"], how="left")
    else:
        df["n_member_cluster"] = np.nan
        df["n_episode_cluster"] = np.nan
        df["n_member_episode"] = np.nan
        df["last_time"] = pd.NaT

    # lvl3 fallback
    if not hist["lvl3"].empty:
        m3 = df.merge(hist["lvl3"], on=["site", "modul", "sub_modul"], how="left", suffixes=("", "_h3"))
        for col in ["n_member_cluster", "n_episode_cluster", "n_member_episode", "last_time"]:
            df[col] = df[col].where(df[col].notna(), m3.get(f"{col}_h3"))

    # lvl2 fallback
    if not hist["lvl2"].empty:
        m2 = df.merge(hist["lvl2"], on=["modul", "sub_modul"], how="left", suffixes=("", "_h2"))
        for col in ["n_member_cluster", "n_episode_cluster", "n_member_episode", "last_time"]:
            df[col] = df[col].where(df[col].notna(), m2.get(f"{col}_h2"))

    # global fallback
    df["n_member_cluster"] = pd.to_numeric(df["n_member_cluster"], errors="coerce").fillna(g_nmc)
    df["n_episode_cluster"] = pd.to_numeric(df["n_episode_cluster"], errors="coerce").fillna(g_nec)
    df["n_member_episode"] = pd.to_numeric(df["n_member_episode"], errors="coerce").fillna(g_nme)

    # gap_days = tgl_submit - last_time
    df["gap_days"] = (
        (df["tgl_submit"] - pd.to_datetime(df["last_time"], errors="coerce"))
        .dt.total_seconds()
        .div(86400.0)
    )
    df["gap_days"] = df["gap_days"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0, 3650)

    df["window_days"] = int(window_days)

    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df


# ============================================================
# 🧭 UI
# ============================================================
st.title("📤 Upload & Predict – Prediksi Insiden Berulang")
st.caption(
    "Upload tiket baru. Panel evaluasi menampilkan metrik RF/XGBoost/SVM dari tabel evaluasi. "
    "Prediksi dilakukan multi-model dan histori imputasi fitur diambil dari seluruh dataset_supervised (tanpa split_name)."
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

missing = [c for c in REQUIRED_BASE_COLS if c not in df_raw.columns]
if missing:
    st.error(f"Kolom wajib tidak ditemukan: {missing}")
    st.stop()

text_col = next((c for c in TEXT_COL_CANDIDATES if c in df_raw.columns), None)
if text_col is None:
    st.warning(
        "Kolom teks tidak ditemukan (detailed_description / detailed_decription). "
        "Ini tidak menghambat prediksi (model tidak memakai teks)."
    )

st.divider()

# ============================================================
# 2) Panel Pemodelan & Evaluasi
# ============================================================
st.subheader("2️⃣ Pemodelan & Evaluasi (RF/XGBoost/SVM)")

df_eval_all = read_df(
    engine,
    f"""
    SELECT
      run_time,
      jenis_pendekatan,
      modeling_id,
      window_days,
      time_col,
      COALESCE(include_noise,false) AS include_noise,
      COALESCE(eligible_rule,'') AS eligible_rule,
      COALESCE(split_name,'') AS split_name,
      model_name,
      test_size,
      random_state,
      precision_pos,
      recall_pos,
      f1_pos,
      tp, fn, fp, tn
    FROM {T_EVAL}
    ORDER BY run_time DESC
    """
)

if df_eval_all.empty:
    st.error(f"Belum ada evaluasi di `{T_EVAL}`. Jalankan halaman evaluasi model prediksi terlebih dahulu.")
    st.stop()

# filter konfigurasi (mirip halaman evaluasi)
c1, c2, c3 = st.columns(3)
with c1:
    jenis = st.selectbox("jenis_pendekatan", sorted(df_eval_all["jenis_pendekatan"].unique().tolist()))
sub = df_eval_all[df_eval_all["jenis_pendekatan"] == jenis].copy()

with c2:
    modeling_id = st.selectbox("modeling_id", sub["modeling_id"].astype(str).unique().tolist())
sub = sub[sub["modeling_id"].astype(str) == str(modeling_id)].copy()

with c3:
    window_days = st.selectbox("window_days", sorted(sub["window_days"].unique().tolist()))
sub = sub[sub["window_days"] == int(window_days)].copy()

c4, c5, c6 = st.columns([1.2, 1.0, 1.2])
with c4:
    time_col = st.selectbox("time_col", sorted(sub["time_col"].astype(str).unique().tolist()))
sub = sub[sub["time_col"].astype(str) == str(time_col)].copy()

with c5:
    include_noise = st.selectbox("include_noise", [False, True], index=0)
sub = sub[sub["include_noise"].astype(bool) == bool(include_noise)].copy()

with c6:
    eligible_rule = st.selectbox("eligible_rule", sorted(sub["eligible_rule"].astype(str).unique().tolist()))
sub = sub[sub["eligible_rule"].astype(str) == str(eligible_rule)].copy()

split_name = st.selectbox("split_name (identitas evaluasi)", sorted(sub["split_name"].astype(str).unique().tolist()))
sub = sub[sub["split_name"].astype(str) == str(split_name)].copy()

# latest per model (RF/XGB/SVM)
latest_eval = read_df(
    engine,
    f"""
    SELECT DISTINCT ON (model_name)
      run_time,
      model_name,
      precision_pos,
      recall_pos,
      f1_pos,
      tp, fn, fp, tn
    FROM {T_EVAL}
    WHERE jenis_pendekatan = :jenis
      AND modeling_id = CAST(:modeling_id AS uuid)
      AND window_days = :window_days
      AND time_col = :time_col
      AND COALESCE(include_noise,false) = :include_noise
      AND COALESCE(eligible_rule,'') = :eligible_rule
      AND COALESCE(split_name,'') = :split_name
    ORDER BY model_name, run_time DESC
    """,
    dict(
        jenis=str(jenis),
        modeling_id=str(modeling_id),
        window_days=int(window_days),
        time_col=str(time_col),
        include_noise=bool(include_noise),
        eligible_rule=str(eligible_rule),
        split_name=str(split_name),
    ),
)

if latest_eval.empty:
    st.warning("Tidak ada hasil evaluasi untuk konfigurasi yang dipilih.")
    st.stop()

st.caption(
    f"Konfigurasi: jenis={jenis} | modeling_id={modeling_id} | w={int(window_days)} | time_col={time_col} | "
    f"include_noise={include_noise} | eligible_rule={eligible_rule} | split(evaluasi)={split_name}"
)

# tabel metrik
out_eval = pd.DataFrame(
    [
        {
            "Model Prediktif": str(r["model_name"]),
            "Precision (Berulang)": round(float(r["precision_pos"]), 4),
            "Recall (Berulang)": round(float(r["recall_pos"]), 4),
            "F1-Score (Berulang)": round(float(r["f1_pos"]), 4),
            "True Positive (TP)": int(r["tp"]),
            "False Negative (FN)": int(r["fn"]),
            "False Positive (FP)": int(r["fp"]),
        }
        for _, r in latest_eval.iterrows()
    ]
).sort_values(["F1-Score (Berulang)", "Recall (Berulang)"], ascending=False)

st.markdown("### Hasil Evaluasi")
st.dataframe(out_eval, use_container_width=True, hide_index=True)

st.markdown("### Confusion Matrix per Model")
cm_cols = st.columns(len(latest_eval))
for i, (_, r) in enumerate(latest_eval.iterrows()):
    with cm_cols[i]:
        name = str(r["model_name"])
        tn, fp, fn, tp = int(r["tn"]), int(r["fp"]), int(r["fn"]), int(r["tp"])
        st.write(f"**{name}**")
        cm_df = pd.DataFrame([[tn, fp], [fn, tp]], index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
        st.dataframe(cm_df, use_container_width=True)

st.divider()

# ============================================================
# 3) Prediksi multi-model (semua model dieksekusi)
# ============================================================
st.subheader("3️⃣ Prediksi (3 Model)")

available_models = latest_eval["model_name"].astype(str).tolist()
needed = ["Random Forest", "XGBoost", "SVM"]
missing_models = [m for m in needed if m not in available_models]
if missing_models:
    st.warning(f"Sebagian model belum ada hasil evaluasinya untuk konfigurasi ini: {missing_models}. "
               f"Prediksi tetap berjalan untuk model yang tersedia.")

picked_models = [m for m in needed if m in available_models]
st.write("Model yang dijalankan:", ", ".join(picked_models) if picked_models else "(tidak ada)")

with st.spinner("Memuat histori imputasi fitur (tanpa split_name)..."):
    hist = load_hist_aggregates_all_splits(
        engine,
        jenis_pendekatan=str(jenis),
        modeling_id=str(modeling_id),
        window_days=int(window_days),
        time_col=str(time_col),
        include_noise=bool(include_noise),
        eligible_rule=str(eligible_rule),
    )

with st.spinner("Menurunkan fitur numerik (gap_days, n_member_*, n_episode_*, window_days)..."):
    df_enriched = enrich_features_from_history(df_raw, hist, window_days=int(window_days))

with st.expander("🩺 Diagnostics fitur turunan", expanded=False):
    st.dataframe(df_enriched[["incident_number", "tgl_submit"] + CAT_COLS + NUM_COLS].head(50), use_container_width=True)

run_pred = st.button("🚀 Jalankan Prediksi 3 Model", type="primary")

if run_pred:
    if not picked_models:
        st.error("Tidak ada model yang tersedia untuk konfigurasi ini.")
        st.stop()

    X = df_enriched[CAT_COLS + NUM_COLS].copy()
    for c in CAT_COLS:
        X[c] = X[c].apply(norm_cat)
    for c in NUM_COLS:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

    df_pred = df_enriched.copy()
    errors: List[Tuple[str, str]] = []

    for mn in picked_models:
        try:
            model = load_model(str(modeling_id), mn)

            # pred label
            yhat = model.predict(X).astype(int)
            df_pred[f"pred_{mn}"] = yhat

            # prob bila ada
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X)[:, 1]
                df_pred[f"prob_{mn}"] = p.astype(float)
            else:
                df_pred[f"prob_{mn}"] = np.nan

        except Exception as e:
            errors.append((mn, str(e)))

    if errors:
        st.warning("Sebagian model gagal dipakai untuk prediksi:")
        for mn, msg in errors:
            st.write(f"- {mn}: {msg}")

    # ========================================================
    # 4) Hasil Prediksi (tampilkan semua model)
    # ========================================================
    st.subheader("4️⃣ Hasil Prediksi (RF/XGB/SVM)")

    OUT_COLS = [
        "incident_number",
        "tgl_submit",
        "site",
        "assignee",
        "modul",
        "sub_modul",
    ] + NUM_COLS

    # tambah kolom output per model
    for mn in picked_models:
        OUT_COLS += [f"pred_{mn}", f"prob_{mn}"]

    OUT_COLS = [c for c in OUT_COLS if c in df_pred.columns]

    st.dataframe(df_pred[OUT_COLS], use_container_width=True, height=520)

    # ringkasan per model
    st.markdown("### Ringkasan Prediksi per Model")
    cols = st.columns(len(picked_models))
    for i, mn in enumerate(picked_models):
        with cols[i]:
            n1 = int((df_pred[f"pred_{mn}"] == 1).sum()) if f"pred_{mn}" in df_pred.columns else 0
            n0 = int((df_pred[f"pred_{mn}"] == 0).sum()) if f"pred_{mn}" in df_pred.columns else 0
            st.metric(f"{mn} → Pred=1", f"{n1:,}")
            st.metric(f"{mn} → Pred=0", f"{n0:,}")

    # ========================================================
    # 5) Unduh
    # ========================================================
    st.subheader("5️⃣ Unduh Hasil")
    csv = df_pred[OUT_COLS].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download hasil prediksi (CSV)",
        data=csv,
        file_name="hasil_prediksi_upload_insiden_berulang_multi_model.csv",
        mime="text/csv",
    )

    with st.expander("📘 Catatan Metodologis", expanded=False):
        st.markdown(
            """
- Panel evaluasi menampilkan metrik RF/XGBoost/SVM dari tabel `prediction_evaluation_results` untuk konfigurasi terpilih.
- Prediksi upload **tidak menjalankan clustering real-time**; fitur numerik diimputasi dari histori `dataset_supervised`.
- Untuk meningkatkan konsistensi, histori imputasi diambil dari **seluruh dataset_supervised tanpa split_name**.
  Ini mengurangi bias akibat histori yang terlalu sedikit (misalnya hanya train/test saja).
"""
        )
