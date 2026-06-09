import yfinance as yf
import pandas as pd


def download_price_data(ticker: str, period: str = "2y") -> pd.DataFrame: #seçilen ticker için veriyi çekmek, kontrol etmek ve CSV’ye yazmak.
    print(f"[INFO] '{ticker}' için veri indiriliyor...")

    df = yf.download(
        ticker,  #sembol (TSLA, AAPL, BTC-USD, BIST hisseleri vs.)
        period=period,       # son 2 yıl, 6 ay, 1y gibi değerler olabilir
        interval="1d",       # günlük veriler
        auto_adjust=False    # Adj Close kolonunu ayrıca görelim diye
    )

    # 🔹 MultiIndex kolonları düzleştir (örn. (Close, TSLA) → Close)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    print("[INFO] İlk 5 satır:")
    print(df.head())

    print("\n[INFO] Son 5 satır:")
    print(df.tail())

    print("\n[INFO] DataFrame bilgisi:")
    print(df.info())

    # CSV olarak kaydedelim
    output_path = f"data/{ticker}_historical.csv"
    df.to_csv(output_path)#Bu veri pipeline’ın ham verisi gibi. Sonraki katmanlarda istersen yfinance yerine bu CSV’den devam edebilirsin.
    print(f"\n[INFO] Veri '{output_path}' dosyasına kaydedildi.")

    return df


if __name__ == "__main__":
    for t in ["TSLA", "AAPL", "MSFT"]:
        download_price_data(t, period="1y")
