import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features_layer import build_feature_dataframe


def train_model_for_ticker(ticker: str, period: str = "2y"):
    """
    Verilen ticker için (örn. TSLA, AAPL, MSFT):
    1) Feature'lı veriyi hazırlar (X, y)
    2) Train/test split yapar
    3) Logistic Regression modeli eğitir
    4) Performansı yazdırır
    5) Modeli models/{TICKER}_logreg_model.joblib olarak kaydeder
    """

    ticker = ticker.upper()
    print(f"\n[INFO] {ticker} için feature DataFrame hazırlanıyor...")
    df, X, y = build_feature_dataframe(ticker, period=period)

    print(f"[INFO] Toplam örnek sayısı: {len(df)}")
    print(f"[INFO] Pozitif sınıf (up=1) oranı: {y.mean():.2f}")

    # 1) Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    print(f"[INFO] Train set boyutu: {X_train.shape}, Test set boyutu: {X_test.shape}")

    # 2) Pipeline: StandardScaler + RandomForestClassifier
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=42))
        ]
    )

    # 3) GridSearchCV with TimeSeriesSplit
    param_grid = {
        "clf__n_estimators": [50, 100, 200],
        "clf__max_depth": [None, 5, 10],
        "clf__min_samples_split": [2, 5]
    }
    tscv = TimeSeriesSplit(n_splits=3)
    
    model = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='accuracy', n_jobs=-1)

    print("[INFO] Model eğitiliyor ve hiperparametre optimizasyonu yapılıyor (GridSearch)...")
    model.fit(X_train, y_train)
    print(f"[INFO] En iyi parametreler: {model.best_params_}")

    # 3) Test set üzerinde performans
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n[RESULT] [{ticker}] Test Accuracy: {acc:.3f}\n")

    print("[RESULT] Classification Report:")
    print(classification_report(y_test, y_pred, digits=3))

    print("[RESULT] Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 4) models klasörünü oluştur
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, f"{ticker}_rf_model.joblib")

    # 5) Eğitilen modeli kaydet
    joblib.dump(model, model_path)
    print(f"\n[INFO] [{ticker}] Model '{model_path}' dosyasına kaydedildi.")

    return model


if __name__ == "__main__":
    # İstediğin hisseleri burada eğitebilirsin
    tickers = ["TSLA", "AAPL", "MSFT"]
    for t in tickers:
        print(f"\n========== {t} modeli eğitiliyor ==========")
        train_model_for_ticker(t, period="2y")
