import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv("data/processed/meajor_text.csv")

# dividir dados
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# vetorizar texto
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# modelo
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# previsão
y_pred = model.predict(X_test_vec)

# avaliação
print(classification_report(y_test, y_pred))