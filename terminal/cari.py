import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer

engine = create_engine("postgresql+psycopg2://postgres:admin*123@localhost:5432/incident_djp")

sql = """
SELECT modul, text_sintaksis
FROM lasis_djp.incident_clean
WHERE text_sintaksis IS NOT NULL
  AND text_sintaksis <> ''
"""

df = pd.read_sql(text(sql), engine)

# 🔎 pastikan hanya 1 modul (cek variasi penulisan)
print(df["modul"].value_counts().head(20))

# ✅ samakan parameter TF-IDF
vec_params = dict(
    min_df=2,
    max_df=0.95,
    ngram_range=(1,1),
    lowercase=False,
    max_features=100000,   # kalau kamu pakai ini di modeling, set juga di sini
)

# (A) Hitung feature untuk seluruh df (kalau memang hanya 1 modul, hasil harus sama dengan modul itu)
v_all = TfidfVectorizer(**vec_params)
v_all.fit(df["text_sintaksis"].astype(str).tolist())
print("Jumlah feature (ALL):", len(v_all.get_feature_names_out()))

# (B) Hitung feature untuk modul E-Registration saja
g = df[df["modul"].astype(str).str.strip().str.lower() == "e-registration"]
v_mod = TfidfVectorizer(**vec_params)
v_mod.fit(g["text_sintaksis"].astype(str).tolist())
print("Jumlah feature (E-Registration):", len(v_mod.get_feature_names_out()))
