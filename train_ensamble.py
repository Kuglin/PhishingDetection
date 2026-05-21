import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from models.text_model import train_text_model
from models.url_model import train_url_model
from models.header_model import train_header_model

# CARREGAR DADOS
url = "https://zenodo.org/records/18471483/files/meajor_cleaned_preprocessed.csv?download=1"
df = pd.read_csv(url, low_memory=False)

# LIMPEZA
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)
df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")

# SPLIT ÚNICO
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# TREINAR MODELOS BASE
text_model, vectorizer, text_train, text_test = train_text_model(train_df, test_df)
url_model, url_train, url_test = train_url_model(train_df, test_df)
header_model, header_train, header_test = train_header_model(train_df, test_df)

# STACKING
X_train_meta = np.vstack([
    text_train,
    url_train,
    header_train
]).T

X_test_meta = np.vstack([
    text_test,
    url_test,
    header_test
]).T

#  META-MODEL
meta_model = LogisticRegression()
meta_model.fit(X_train_meta, train_df["label"])

# AVALIAÇÃO
y_pred = meta_model.predict(X_test_meta)

print("\n===== RESULT =====")
print(classification_report(test_df["label"], y_pred))

# PESOS DO META-MODEL
print("\nPesos do ensemble (texto, url, header):")
print(meta_model.coef_)

# SALVAR MODELOS
joblib.dump(vectorizer, "models/vectorizer.pkl")
joblib.dump(text_model, "models/text_model.pkl")
joblib.dump(url_model, "models/url_model.pkl")
joblib.dump(header_model, "models/header_model.pkl")
joblib.dump(meta_model, "models/meta_model.pkl")