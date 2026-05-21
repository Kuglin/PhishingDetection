import joblib
import numpy as np
import pandas as pd
import os

from email import policy
from email.parser import BytesParser
import re

# LOAD MODELS
vectorizer = joblib.load("models/vectorizer.pkl")
text_model = joblib.load("models/text_model.pkl")
url_model = joblib.load("models/url_model.pkl")
header_model = joblib.load("models/header_model.pkl")
meta_model = joblib.load("models/meta_model.pkl")

# LOAD EMAIL
def load_eml(path):
    with open(path, "rb") as f:
        return BytesParser(policy=policy.default).parse(f)

# EXTRACTION
def extract(msg):
    subject = msg["subject"] or ""
    body = ""

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body += part.get_content()
    else:
        body = msg.get_content()

    urls = re.findall(r'https?://\S+', body)

    return subject, body, urls, msg

# PREDICTION
def predict_eml(path):
    msg = load_eml(path)
    subject, body, urls, msg = extract(msg)

    # TEXTO
    text = subject + " " + body
    X_text = vectorizer.transform([text])
    text_prob = text_model.predict_proba(X_text)[:, 1][0]

    # URL
    url_features = extract_url_features(urls)
    X_url = pd.DataFrame([url_features])
    url_prob = url_model.predict_proba(X_url)[:, 1][0]

    # HEADER
    header_features = extract_header_features(msg)
    X_header = pd.DataFrame([header_features])
    header_prob = header_model.predict_proba(X_header)[:, 1][0]

    print("Text prob:", text_prob)
    print("URL prob:", url_prob)
    print("Header prob:", header_prob)

    # ENSEMBLE
    X_meta = np.array([[text_prob, url_prob, header_prob]])
    final_prob = meta_model.predict_proba(X_meta)[:, 1][0]

    return final_prob

def extract_url_features(urls):
    lengths = [len(u) for u in urls]
    subdomains = [u.count('.') for u in urls]

    return {
        "url_count": len(urls),
        "url_length_max": max(lengths) if lengths else 0,
        "url_length_avg": sum(lengths)/len(lengths) if lengths else 0,
        "url_subdom_max": max(subdomains) if subdomains else 0,
        "url_subdom_avg": sum(subdomains)/len(subdomains) if subdomains else 0
    }

def extract_header_features(msg):
    attachments = 0

    for part in msg.walk():
        if part.get_filename():
            attachments += 1

    return {
        "attachment_count": attachments,
        "has_attachments": int(attachments > 0),
        "language": 0  # simplificado
    }

def scan_folder(folder_path):
    results = []

    for file in os.listdir(folder_path):
        if file.endswith(".eml"):
            path = os.path.join(folder_path, file)

            try:
                prob = predict_eml(path)

                if prob > 0.8:
                    label = " PHISHING"
                elif prob > 0.5:
                    label = " SUSPEITO"
                else:
                    label = " LEGÍTIMO"

                print(f"{file} → {label} ({prob:.4f})")

                results.append({
                    "file": file,
                    "probability": prob,
                    "classification": label
                })

            except Exception as e:
                print(f"{file} → ERRO: {e}")

    return results

if __name__ == "__main__":
    folder = "emails"  

    if not os.path.exists(folder):
        print("Pasta não encontrada.")
    else:
        scan_folder(folder)
    
    df = pd.DataFrame(scan_folder(folder))
    df.to_csv("resultado_scan.csv", index=False)