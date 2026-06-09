# features_layer.py
import pandas as pd
import numpy as np
from data_layer import download_price_data
import ta
import yfinance as yf


def build_feature_dataframe(ticker: str, period: str = "2y", drop_neutral_targets: bool = True):
    ticker = ticker.upper()

    # 1) Ham veriyi al
    df = download_price_data(ticker, period=period).copy()

    # --- MAKRO VERİ ---
    print("[INFO] Makro veriler (SPY, ^VIX) indiriliyor...")
    spy_df = yf.download("SPY", period=period, interval="1d", auto_adjust=False)
    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_df.columns = [col[0] for col in spy_df.columns]
    
    vix_df = yf.download("^VIX", period=period, interval="1d", auto_adjust=False)
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = [col[0] for col in vix_df.columns]
        
    df["spy_close"] = spy_df["Close"]
    df["vix_close"] = vix_df["Close"]
    df["spy_close"] = df["spy_close"].ffill()
    df["vix_close"] = df["vix_close"].ffill()
    
    df["spy_return_1d"] = df["spy_close"].pct_change(1)
    df["spy_return_5d"] = df["spy_close"].pct_change(5)

    # 2) Feature engineering
    #  Getiriler
    df["return_1d"] = df["Close"].pct_change(1)
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_10d"] = df["Close"].pct_change(10)

    #  Hareketli ortalamalar
    df["ma_5"] = df["Close"].rolling(window=5).mean()
    df["ma_10"] = df["Close"].rolling(window=10).mean()
    df["ma_20"] = df["Close"].rolling(window=20).mean()

    #  Fiyat / MA oranları
    df["price_ma5_ratio"] = df["Close"] / df["ma_5"]
    df["price_ma20_ratio"] = df["Close"] / df["ma_20"]

    #  Fiyat volatilitesi
    df["vol_5"] = df["Close"].rolling(window=5).std()
    df["vol_10"] = df["Close"].rolling(window=10).std()

    df["volatility_5d"] = df["return_1d"].rolling(window=5).std()
    df["volatility_10d"] = df["return_1d"].rolling(window=10).std()

    #  Hacim değişimi
    df["volume_change_1d"] = df["Volume"].pct_change()

    # --- TEKNİK GÖSTERGELER (TA) ---
    df["rsi_14"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    macd = ta.trend.MACD(df["Close"])
    df["macd"] = macd.macd()
    df["macd_diff"] = macd.macd_diff()
    
    bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
    df["bb_width"] = bb.bollinger_wband()

    # 3) 5 GÜNLÜK TREND TAHMİNİ (Daha az gürültülü)
    df["future_close_5d"] = df["Close"].shift(-5)
    df["future_return_5d"] = (df["future_close_5d"] - df["Close"]) / df["Close"]

    UP_THRESHOLD = 0.01    # 5 günde +%1 ve üzeri -> UP
    DOWN_THRESHOLD = -0.01  # 5 günde -%1 ve altı -> DOWN

    df["target"] = np.where(
        df["future_return_5d"] > UP_THRESHOLD, 1,
        np.where(df["future_return_5d"] < DOWN_THRESHOLD, 0, np.nan)
    )

    # Modelin gerçekten kullandığı feature listesi
    feature_cols = [
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

    if drop_neutral_targets:
        # Flat günleri at
        df = df.dropna(subset=["target"])
        df["target"] = df["target"].astype(int)

        # Feature hesaplarından gelen NaN'leri de temizle
        df = df.dropna()
    else:
        # Simulasyon icin gelecekteki target bilgisiyle satir eleme yapma.
        df = df.dropna(subset=feature_cols)

    X = df[feature_cols].copy()
    y = df["target"].copy()

    # CSV olarak da kaydediyoruz
    output_path = f"data/{ticker}_features.csv"
    df.to_csv(output_path)
    print(f"[INFO] Feature DataFrame '{output_path}' dosyasına kaydedildi.")
    print(f"[INFO] X shape: {X.shape}, y length: {len(y)}")

    return df, X, y
