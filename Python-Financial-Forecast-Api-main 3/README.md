# Financial Forecast API

An end-to-end machine learning application for short-term stock trend forecasting, built with Python, FastAPI, scikit-learn, and an interactive dashboard.

The system collects market data, engineers technical and macro features, trains Random Forest models, exposes prediction services through a REST API, and visualizes forecasts, backtests, model metrics, feature importance, and virtual portfolio simulation in a browser UI.

> Educational project only. This repository is not financial advice and does not execute real trades.

## Why This Project Matters

This project demonstrates a complete applied ML workflow rather than a notebook-only experiment:

- Data ingestion from an external market data source.
- Reusable feature engineering pipeline.
- Time-series-aware model training and evaluation.
- Model persistence with joblib.
- Production-style API layer with FastAPI.
- Interactive dashboard for non-technical users.
- Backtesting and virtual USD portfolio simulation.
- Explainability through feature importance.

It is designed to show practical Python engineering across data processing, machine learning, API development, and user-facing analytics.

## Core Capabilities

| Capability | Description |
| :--- | :--- |
| 5-day trend prediction | Predicts whether selected stocks are likely to move up, down, or remain neutral over a 5-trading-day horizon. |
| Technical feature engineering | Uses returns, moving averages, volatility, RSI, MACD, MACD difference, and Bollinger Band width. |
| Macro context | Adds SPY and VIX-derived market context features. |
| Random Forest model | Trains ticker-specific Random Forest classifiers with GridSearchCV and TimeSeriesSplit. |
| FastAPI service | Serves predictions, metrics, backtests, simulations, chart data, and feature importance. |
| Dashboard UI | Provides prediction cards, metrics, technical chart overlays, backtest results, and strategy simulation. |
| USD test-money simulation | Tests model-generated BUY/SELL/HOLD signals against historical prices using a virtual USD balance. |
| Explainability | Shows the most influential model features in the dashboard. |

## Architecture


<img width="256" height="743" alt="image" src="https://github.com/user-attachments/assets/23157712-6c65-4b87-a85c-b2b667d7ba1b" />


## Tech Stack

| Area | Tools |
| :--- | :--- |
| Language | Python |
| API | FastAPI, Uvicorn |
| Machine learning | scikit-learn, RandomForestClassifier, GridSearchCV, TimeSeriesSplit |
| Data processing | Pandas, NumPy |
| Market data | yFinance |
| Technical indicators | ta |
| Model persistence | joblib |
| Visualization | Matplotlib, Chart.js |
| Frontend | HTML, CSS, vanilla JavaScript |

## Supported Tickers

The project is configured for:

- TSLA
- AAPL
- MSFT

Additional tickers can be added by training and saving a corresponding model.

## Model Design

The project predicts a 5-day trend target:

- `future_close_5d = Close.shift(-5)`
- `future_return_5d = (future_close_5d - Close) / Close`
- `UP` when `future_return_5d > +1%`
- `DOWN` when `future_return_5d < -1%`
- Neutral rows are excluded from supervised model training

The API prediction output is mapped into user-facing signals:

- `P(up) >= 55%`: UP / BUY signal
- `45% < P(up) < 55%`: NEUTRAL / HOLD signal
- `P(up) <= 45%`: DOWN / SELL signal

## Getting Started

### 1. Create a Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train Models

```bash
python train_model.py
```

This trains ticker-specific Random Forest models and saves them under `models/`.

### 3. Run the API

```bash
uvicorn app.main:app --reload
```

Open the dashboard:

[http://127.0.0.1:8000/ui](http://127.0.0.1:8000/ui)

## API Endpoints

| Endpoint | Purpose |
| :--- | :--- |
| `GET /predict?ticker=TSLA` | Returns latest 5-day prediction and upside probability. |
| `GET /metrics?ticker=TSLA&last_n=90` | Returns recent model accuracy metrics. |
| `GET /backtest?ticker=TSLA&date=2025-01-10` | Tests what the model would have predicted on a historical date. |
| `GET /simulate?ticker=TSLA&initial_cash=10000` | Runs a virtual USD strategy simulation. |
| `GET /price_series?ticker=TSLA&days=90` | Returns chart-ready close price, RSI, and MACD data. |
| `GET /feature_importance?ticker=TSLA&top_n=8` | Returns top model feature importance values. |
| `GET /ui` | Serves the interactive dashboard. |

## Evaluation

The repository includes separate evaluation scripts:

```bash
python regression_eval.py
python classification_eval.py
```

Recent verification on TSLA produced:

| Evaluation | Result |
| :--- | :--- |
| Regression target | `future_close_5d` |
| R2 | `0.1384` |
| RMSE | `24.2992` |
| Classification accuracy | `0.5714` |
| Classification F1-score | `0.3077` |

These values are intentionally reported as measured outputs, not as performance guarantees. Short-term market prediction is noisy and changes with new data.

## Virtual Portfolio Simulation

The `/simulate` endpoint tests model signals using a virtual USD balance.

Default behavior:

- Initial cash: `$10,000`
- Commission: `0.1%`
- BUY when `P(up) >= 55%`
- SELL when `P(up) <= 45%`
- HOLD between the thresholds

The simulation returns:

- Final portfolio value
- Total profit/loss
- Buy-and-hold comparison
- Max drawdown
- Trade count
- Recent trades
- Open position and cash balance

## Project Structure

```text
app/
  main.py              FastAPI routes, dashboard UI, API endpoints
  predictor.py         Model loading and prediction helper
data/
  *.csv                Downloaded and engineered market data
figures/
  *.png                Regression/evaluation plots
models/
  *.joblib             Trained Random Forest models
data_layer.py          Market data download layer
features_layer.py      Feature engineering and target creation
train_model.py         Model training entry point
regression_eval.py     Regression evaluation script
classification_eval.py Classification metric script
requirements.txt       Python dependencies
README.md              Project documentation
```

## Current Limitations

- Market data is noisy and short-term forecasts are inherently difficult.
- The strategy simulation is simplified and does not model slippage, taxes, liquidity, or market impact.
- The model uses historical technical and macro indicators, not live news or fundamental data.
- Results may change as yFinance returns newer market data.
- This project is educational and should not be used for real investment decisions.

## Future Improvements

- Add more tickers and longer historical datasets.
- Add XGBoost, Gradient Boosting, LSTM, or Transformer-based time-series models.
- Add news and social media sentiment features, such as Twitter/X or financial-news sentiment scores.
- Add richer risk controls: stop-loss, position sizing, volatility-adjusted exposure.
- Deploy the API and dashboard to a cloud provider.
- Add CI tests for API endpoints and feature-generation contracts.

## Disclaimer

This repository is for education and experimentation only. It does not provide financial advice, does not guarantee future market performance, and does not execute real trades.
