# predictor.py
import os
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd

from features_layer import build_feature_dataframe

# feature kolonları
# (features_layer.py içindeki feature_cols ile birebir aynı olmalı)
FEATURE_COLS = [
    "return_1d",
    "return_5d",
    "return_10d",
    "ma_5",
    "ma_10",
    "ma_20",
    "price_ma5_ratio",
    "price_ma20_ratio",
    "vol_5",
    "vol_10",
    "volatility_5d",
    "volatility_10d",
    "volume_change_1d",
    "spy_return_1d",
    "spy_return_5d",
    "vix_close",
    "rsi_14",
    "macd",
    "macd_diff",
    "bb_width"
]

MODELS_DIR = Path("models")

# Her ticker için model cache'i
_models: Dict[str, object] = {}


def get_model_path(ticker: str) -> Path:
    """
    Verilen ticker için model dosyasının yolunu döndür.
    Örn: models/TSLA_logreg_model.joblib
    """
    ticker = ticker.upper()
    return MODELS_DIR / f"{ticker}_rf_model.joblib"


def load_model(ticker: str):
    """
    Verilen ticker için modeli memory'ye alır.
    Daha önce yüklendiyse _models sözlüğünden kullanır.
    """
    global _models
    ticker = ticker.upper()

    if ticker not in _models:
        model_path = get_model_path(ticker)
        if not model_path.exists():
            raise FileNotFoundError(
                f"{ticker} için model dosyası bulunamadı: {model_path}. "
                f"Önce train_model_for_ticker('{ticker}') ile eğitmelisin."
            )
        _models[ticker] = joblib.load(model_path)
        print(f"[INFO] {ticker} modeli yüklendi: {model_path}")

    return _models[ticker]


def build_latest_feature_row(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Eğitimde kullandığımız ile TAM AYNI feature pipeline'ını kullanarak
    son satırın feature vektörünü üretir.

    - features_layer.build_feature_dataframe çağrılır
    - X içinden FEATURE_COLS sırasıyla son satır alınır
    """
    ticker = ticker.upper()

    # Aynı fonksiyon: download + feature engineering + target + temizlik
    df, X, y = build_feature_dataframe(ticker, period=period)

    if X.empty:
        raise ValueError(f"{ticker} için feature hesaplanamadı (X boş).")

    # Kolon sırası eğitimle birebir aynı olsun
    missing = [c for c in FEATURE_COLS if c not in X.columns]
    if missing:
        raise ValueError(
            f"Aşağıdaki feature'lar X içinde yok: {missing}. "
            f"features_layer.feature_cols ile predictor.FEATURE_COLS eşit mi kontrol et."
        )

    X_latest = X[FEATURE_COLS].tail(1)

    return X_latest


def predict_with_proba(ticker: str, period: str = "2y"):
    """
    Verilen ticker için:
    - direction: 'up' veya 'down'
    - prob_up : 1 (UP) sınıfının olasılığı
    döndürür.
    """
    ticker = ticker.upper()

    model = load_model(ticker)
    X_latest = build_latest_feature_row(ticker, period=period)

    proba = model.predict_proba(X_latest)[0]   # [P(DOWN), P(UP)]
    prob_up = float(proba[1])

    # Threshold mantığı: %45 - %55 arası kararsız / nötr kabul edilir
    if 0.45 <= prob_up <= 0.55:
        direction = "neutral"
    elif prob_up > 0.55:
        direction = "up"
    else:
        direction = "down"

    return {
        "direction": direction,
        "prob_up": prob_up,
    }
