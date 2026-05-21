def train_header_model(train_df, test_df):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report

    train_df["language"] = train_df["language"].astype("category").cat.codes
    test_df["language"] = test_df["language"].astype("category").cat.codes

    features = [
        "attachment_count",
        "has_attachments",
        "language"
    ]

    X_train = train_df[features].fillna(0)
    X_test = test_df[features].fillna(0)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, train_df["label"])

    # AVALIAÇÃO
    y_pred = model.predict(X_test)

    print("\n===== RESULT HEADER =====")
    print(classification_report(test_df["label"], y_pred))


    return (
        model,
        model.predict_proba(X_train)[:, 1],
        model.predict_proba(X_test)[:, 1]
    )