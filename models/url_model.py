def train_url_model(train_df, test_df):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report

    features = [
        "url_count",
        "url_length_max",
        "url_length_avg",
        "url_subdom_max",
        "url_subdom_avg"
    ]

    X_train = train_df[features].fillna(0)
    X_test = test_df[features].fillna(0)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, train_df["label"])
    
    # AVALIAÇÃO
    y_pred = model.predict(X_test)

    print("\n===== RESULT URL =====")
    print(classification_report(test_df["label"], y_pred))

    return (
        model,
        model.predict_proba(X_train)[:, 1],
        model.predict_proba(X_test)[:, 1]
    )