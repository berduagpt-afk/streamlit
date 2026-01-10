import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1) INPUT: hasil eksperimen
# =========================
CSV_PATH = "knn_sensitivity_results_6155613e-ef6f-48b8-8e3e-f0f95f314350_20260108_004815.csv"
# ^ ini contoh (threshold 0.70). Ganti sesuai nama file kamu.

TARGET_SIM = 0.70       # fokus reasoning threshold 0.70
K_CHOSEN = 25           # k yang mau kamu justifikasi

df = pd.read_csv(CSV_PATH)

# filter threshold (kalau file kamu berisi beberapa threshold)
df = df[df["sim_threshold"].round(2) == round(TARGET_SIM, 2)].copy()

# sort by k
df = df.sort_values("k")

# cek minimal kolom yang dibutuhkan
need_cols = ["k", "singleton_share", "top10_share", "silhouette", "davies_bouldin"]
missing = [c for c in need_cols if c not in df.columns]
if missing:
    raise ValueError(f"Kolom tidak ada di CSV: {missing}")

# =========================
# 2) BUAT DATA ELBOW (delta / marginal gain)
# =========================
# Penurunan singleton_share (lebih besar = lebih baik)
df["delta_singleton_drop"] = -df["singleton_share"].diff()

# Kenaikan top10_share (lebih besar = makin menggumpal, biasanya tidak diinginkan)
df["delta_top10_rise"] = df["top10_share"].diff()

# Perubahan silhouette (lebih besar = lebih baik)
df["delta_sil"] = df["silhouette"].diff()

# Perubahan DBI (lebih kecil = lebih baik, jadi penurunan DBI itu bagus)
df["delta_dbi_drop"] = -df["davies_bouldin"].diff()

# =========================
# 3) PLOT "ELBOW" UTAMA: metrik vs k
# =========================
plt.figure(figsize=(10, 6))
plt.plot(df["k"], df["singleton_share"], marker="o", label="Singleton share (↓ better)")
plt.plot(df["k"], df["top10_share"], marker="o", label="Top-10 share (↑ = gumpal)")
plt.plot(df["k"], df["silhouette"], marker="o", label="Silhouette (↑ better)")
plt.plot(df["k"], df["davies_bouldin"], marker="o", label="DBI (↓ better)")

# garis k terpilih
plt.axvline(K_CHOSEN, linestyle="--", label=f"Chosen k={K_CHOSEN}")

plt.title(f"Elbow Reasoning for k (sim_threshold={TARGET_SIM})")
plt.xlabel("k (number of neighbors)")
plt.ylabel("Metric value")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# 4) PLOT "MARGINAL CHANGE": titik mulai melandai (diminishing returns)
# =========================
plt.figure(figsize=(10, 6))
plt.plot(df["k"], df["delta_singleton_drop"], marker="o", label="Marginal singleton drop (↑ good)")
plt.plot(df["k"], df["delta_top10_rise"], marker="o", label="Marginal top10 rise (↑ bad)")
plt.plot(df["k"], df["delta_sil"], marker="o", label="Marginal silhouette gain (↑ good)")
plt.plot(df["k"], df["delta_dbi_drop"], marker="o", label="Marginal DBI drop (↑ good)")

plt.axvline(K_CHOSEN, linestyle="--", label=f"Chosen k={K_CHOSEN}")

plt.title(f"Marginal Gains vs k (sim_threshold={TARGET_SIM})")
plt.xlabel("k")
plt.ylabel("Delta (vs previous k)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# 5) TABEL RINGKAS UNTUK DIMASUKKAN KE TESIS
# =========================
cols_out = ["k", "singleton_share", "top10_share", "silhouette", "davies_bouldin",
            "delta_singleton_drop", "delta_top10_rise", "delta_sil", "delta_dbi_drop"]
print(df[cols_out].to_string(index=False))
