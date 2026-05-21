def train_text_model(train_df, test_df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report

    vectorizer = TfidfVectorizer(max_features=5000)

    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    model = LogisticRegression()
    model.fit(X_train, train_df["label"])
    
    # AVALIAÇÃO
    y_pred = model.predict(X_test)

    print("\n===== RESULT TEXT =====")
    print(classification_report(test_df["label"], y_pred))

    return (
        model,
        vectorizer,
        model.predict_proba(X_train)[:, 1],
        model.predict_proba(X_test)[:, 1]
    )