import os
from math import sqrt

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from features_layer import build_feature_dataframe


TICKER = "TSLA"
PERIOD = "2y"
REGRESSION_TARGET = "future_close_5d"


def run_regression_eval(ticker: str = TICKER, period: str = PERIOD):
    df, X_features, _ = build_feature_dataframe(ticker, period=period)

    print("=== Columns ===")
    print(df.columns.tolist())
    print("===============")
    print(f"Total samples: {len(df)}")

    if df.empty or X_features.empty:
        raise ValueError(
            "Not enough feature data was produced for regression eval. "
            "Check the yfinance output and macro data download step."
        )

    if REGRESSION_TARGET not in df.columns:
        raise ValueError(f"Column '{REGRESSION_TARGET}' was not found in df.")

    y_reg = df[REGRESSION_TARGET].astype(float).values
    X = X_features.values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_reg,
        test_size=0.2,
        shuffle=False,
    )

    print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)

    print(f"\nTICKER: {ticker.upper()}")
    print(f"R2   : {r2:.4f}")
    print(f"RMSE : {rmse:.4f}")

    os.makedirs("figures", exist_ok=True)
    figure_path = f"figures/{ticker.upper()}_regression_r2.png"

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.4, label="Predictions")

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="x = y")

    plt.xlabel(f"Actual value ({REGRESSION_TARGET})")
    plt.ylabel("Predicted value")
    plt.title(f"Actual vs Predicted ({ticker.upper()}, 5-day close regression)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()

    print(f"Figure saved: {figure_path}")
    return {
        "ticker": ticker.upper(),
        "r2": r2,
        "rmse": rmse,
        "figure_path": figure_path,
    }


if __name__ == "__main__":
    run_regression_eval()
