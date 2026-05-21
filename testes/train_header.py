import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("data/raw/meajor.csv")

# limpar label
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

print(df.groupby("sender_domain")["label"].mean().sort_values(ascending=False).head(10))

# selecionar colunas
df = df[
    [
        "attachment_count",
        "has_attachments",
        "sender_domain",
        "receiver_domain",
        "language",
        "label"
    ]
].copy()

# tratar nulos
df = df.fillna("unknown")

# encoding
le_sender = LabelEncoder()
le_receiver = LabelEncoder()
le_lang = LabelEncoder()

df["sender_domain"] = le_sender.fit_transform(df["sender_domain"])
df["receiver_domain"] = le_receiver.fit_transform(df["receiver_domain"])
df["language"] = le_lang.fit_transform(df["language"])

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))