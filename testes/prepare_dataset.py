import pandas as pd

df = pd.read_csv("data/raw/meajor.csv")

print(df.columns)

# ===== TEXTO =====
df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")

# ===== LABEL =====
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

# ===== LIMPEZA =====
df = df.dropna(subset=["text", "label"])
df = df.drop_duplicates(subset=["text"])

# ===== DATASET TEXTO =====
df_text = df[["text", "label"]].copy()
df_text.to_csv("data/processed/meajor_text.csv", index=False)

print("Dataset pronto")
print(df_text.head())
print(df_text["label"].value_counts())