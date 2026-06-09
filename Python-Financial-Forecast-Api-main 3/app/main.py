import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from sklearn.metrics import accuracy_score

from features_layer import build_feature_dataframe
from .predictor import load_model, predict_with_proba


app = FastAPI(
    title="Financial Forecast API",
    version="0.3.0",
    description="FastAPI service that predicts 5-day trend direction for multiple stocks.",
)


@app.on_event("startup")
def startup_event():
    try:
        load_model("TSLA")
    except FileNotFoundError as e:
        print(f"[WARN] TSLA model could not be loaded on startup: {e}")


@app.get("/")
def root():
    return {"message": "Financial Forecast API is running. Example: /predict?ticker=TSLA or /ui"}


@app.get("/predict")
def predict(ticker: str = "TSLA"):
    try:
        result = predict_with_proba(ticker)
        return {
            "ticker": ticker.upper(),
            "prediction": result["direction"],
            "prob_up": result["prob_up"],
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@app.get("/metrics")
def metrics(ticker: str = "TSLA", period: str = "2y", last_n: int = 90):
    ticker = ticker.upper()
    try:
        df, X, y = build_feature_dataframe(ticker, period=period)
        if len(df) < last_n + 5:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough data for {ticker} (len={len(df)}, last_n={last_n}).",
            )

        X_eval = X.tail(last_n)
        y_eval = y.tail(last_n)
        model = load_model(ticker)
        y_pred = model.predict(X_eval)
        acc = accuracy_score(y_eval, y_pred)
        correct = int((y_pred == y_eval).sum())
        total = int(len(y_eval))

        return {
            "ticker": ticker,
            "period_used": period,
            "samples": total,
            "accuracy_last_n": acc,
            "correct": correct,
            "wrong": total - correct,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@app.get("/backtest")
def backtest(ticker: str, date: str = "", period: str = "2y"):
    ticker = ticker.upper()
    try:
        df, X, y = build_feature_dataframe(ticker, period=period)

        if X.empty:
            raise HTTPException(status_code=400, detail=f"Not enough backtest data for {ticker}.")

        if date:
            try:
                target_date = pd.to_datetime(date).normalize()
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid date format. Example: 2025-01-10")
        else:
            target_date = X.index[-1].normalize()
            date = target_date.strftime("%Y-%m-%d")

        date_str = target_date.strftime("%Y-%m-%d")

        def to_date_strings(idx):
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            return idx.normalize().strftime("%Y-%m-%d")

        X_dates = to_date_strings(X.index)
        matching_rows = X[X_dates == date_str]

        if matching_rows.empty:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"No data was found for {ticker} on {date}. "
                    "The market may have been closed that day, or the date may be outside the 2-year window."
                ),
            )

        y_actual = int(y[to_date_strings(y.index) == date_str].iloc[0])
        actual_movement = "up" if y_actual == 1 else "down"

        model = load_model(ticker)
        y_pred = model.predict(matching_rows)[0]
        proba = model.predict_proba(matching_rows)[0]
        prob_up = float(proba[1])
        predicted_movement = "up" if int(y_pred) == 1 else "down"

        df_row = df[to_date_strings(df.index) == date_str]
        close_on_date = float(df_row["Close"].iloc[0])
        next_close = float(df_row["future_close_5d"].iloc[0])

        return {
            "ticker": ticker,
            "date": target_date.strftime("%Y-%m-%d"),
            "prediction": predicted_movement,
            "prob_up": prob_up,
            "actual_movement": actual_movement,
            "close_on_date": close_on_date,
            "close_after_5d": next_close,
        }
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except KeyError:
        raise HTTPException(status_code=400, detail=f"No data was found for {ticker} on {date}.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


def _signal_from_prob(prob_up: float, buy_threshold: float, sell_threshold: float) -> str:
    if prob_up >= buy_threshold:
        return "buy"
    if prob_up <= sell_threshold:
        return "sell"
    return "hold"


def _max_drawdown(values):
    peak = values[0]
    max_dd = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            max_dd = min(max_dd, (value - peak) / peak)
    return abs(max_dd)


@app.get("/simulate")
def simulate(
    ticker: str = "TSLA",
    initial_cash: float = 10000.0,
    period: str = "2y",
    last_n: int = 90,
    buy_threshold: float = 0.55,
    sell_threshold: float = 0.45,
    commission_pct: float = 0.1,
):
    ticker = ticker.upper()
    try:
        if initial_cash <= 0:
            raise HTTPException(status_code=400, detail="initial_cash must be greater than 0.")
        if not 0 <= commission_pct < 10:
            raise HTTPException(status_code=400, detail="commission_pct must be between 0 and 10.")
        if not 0 < sell_threshold < buy_threshold < 1:
            raise HTTPException(status_code=400, detail="Threshold order must be 0 < sell < buy < 1.")
        if last_n < 2:
            raise HTTPException(status_code=400, detail="last_n must be at least 2.")

        df, X, _ = build_feature_dataframe(ticker, period=period, drop_neutral_targets=False)
        if df.empty or X.empty:
            raise HTTPException(status_code=400, detail=f"Not enough data could be generated for {ticker}.")

        model = load_model(ticker)
        X_sim = X.tail(last_n)
        probabilities = model.predict_proba(X_sim)[:, 1]
        closes = df.loc[X_sim.index, "Close"].astype(float)

        cash = float(initial_cash)
        shares = 0.0
        entry_price = None
        fee_rate = commission_pct / 100
        portfolio_values = []
        trades = []
        closed_trade_pnls = []

        for date, close_price, prob_up in zip(X_sim.index, closes.values, probabilities):
            signal = _signal_from_prob(float(prob_up), buy_threshold, sell_threshold)

            if signal == "buy" and shares == 0 and cash > 0:
                fee = cash * fee_rate
                investable_cash = cash - fee
                shares = investable_cash / close_price
                cash = 0.0
                entry_price = close_price
                trades.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "action": "buy",
                    "price": round(close_price, 2),
                    "prob_up": round(float(prob_up), 4),
                    "shares": round(shares, 6),
                    "fee": round(fee, 2),
                })
            elif signal == "sell" and shares > 0:
                gross_cash = shares * close_price
                fee = gross_cash * fee_rate
                cash = gross_cash - fee
                pnl = cash - (shares * entry_price)
                closed_trade_pnls.append(pnl)
                trades.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "action": "sell",
                    "price": round(close_price, 2),
                    "prob_up": round(float(prob_up), 4),
                    "shares": round(shares, 6),
                    "fee": round(fee, 2),
                    "pnl": round(pnl, 2),
                })
                shares = 0.0
                entry_price = None

            portfolio_values.append(cash + shares * close_price)

        last_price = float(closes.iloc[-1])
        final_value = cash + shares * last_price
        total_profit = final_value - initial_cash
        total_return_pct = (total_profit / initial_cash) * 100

        first_price = float(closes.iloc[0])
        buy_hold_shares = (initial_cash * (1 - fee_rate)) / first_price
        buy_hold_exit_fee = buy_hold_shares * last_price * fee_rate
        buy_hold_final = buy_hold_shares * last_price - buy_hold_exit_fee
        buy_hold_profit = buy_hold_final - initial_cash

        return {
            "ticker": ticker,
            "currency": "USD",
            "period_used": period,
            "samples": len(X_sim),
            "initial_cash": round(initial_cash, 2),
            "final_value": round(final_value, 2),
            "cash": round(cash, 2),
            "shares": round(shares, 6),
            "last_price": round(last_price, 2),
            "total_profit": round(total_profit, 2),
            "total_return_pct": round(total_return_pct, 2),
            "buy_hold_final_value": round(buy_hold_final, 2),
            "buy_hold_profit": round(buy_hold_profit, 2),
            "buy_hold_return_pct": round((buy_hold_profit / initial_cash) * 100, 2),
            "max_drawdown_pct": round(_max_drawdown(portfolio_values) * 100, 2),
            "trade_count": len(trades),
            "buy_count": sum(1 for trade in trades if trade["action"] == "buy"),
            "sell_count": sum(1 for trade in trades if trade["action"] == "sell"),
            "winning_trades": sum(1 for pnl in closed_trade_pnls if pnl > 0),
            "losing_trades": sum(1 for pnl in closed_trade_pnls if pnl <= 0),
            "thresholds": {
                "buy_threshold": buy_threshold,
                "sell_threshold": sell_threshold,
                "commission_pct": commission_pct,
            },
            "recent_trades": trades[-10:],
        }
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation error: {e}")


@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Financial Forecast Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
          :root {
            --bg: #f6f8fb;
            --panel: #ffffff;
            --panel-soft: #f8fafc;
            --text: #0f172a;
            --muted: #64748b;
            --line: #e2e8f0;
            --blue: #2563eb;
            --green: #16a34a;
            --red: #dc2626;
            --orange: #f97316;
            --purple: #7c3aed;
            --shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
          }
          * { box-sizing: border-box; }
          body {
            margin: 0;
            min-height: 100vh;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background:
              radial-gradient(circle at top left, rgba(37, 99, 235, 0.16), transparent 30%),
              radial-gradient(circle at top right, rgba(34, 197, 94, 0.12), transparent 28%),
              var(--bg);
            color: var(--text);
          }
          .page { max-width: 1180px; margin: 0 auto; padding: 34px 20px 46px; }
          .hero { display: flex; justify-content: space-between; align-items: flex-end; gap: 18px; margin-bottom: 22px; }
          .hero h1 { margin: 0; font-size: 40px; line-height: 1; }
          .hero p { max-width: 700px; margin: 12px 0 0; color: var(--muted); font-size: 15px; line-height: 1.6; }
          .hero-badge { white-space: nowrap; border: 1px solid var(--line); background: rgba(255,255,255,0.78); color: var(--muted); padding: 10px 14px; border-radius: 999px; font-size: 13px; font-weight: 700; }
          .dashboard-grid { display: grid; grid-template-columns: 0.9fr 1.1fr; gap: 22px; }
          .triple-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 22px; margin-top: 22px; }
          .card, .chart-card { background: rgba(255,255,255,0.94); border: 1px solid rgba(226,232,240,0.95); border-radius: 18px; box-shadow: var(--shadow); padding: 22px; }
          .card + .card { margin-top: 22px; }
          .card-header { display: flex; justify-content: space-between; align-items: flex-start; gap: 14px; margin-bottom: 18px; }
          .eyebrow { color: var(--blue); font-weight: 800; text-transform: uppercase; letter-spacing: .08em; font-size: 11px; margin-bottom: 6px; }
          h2 { margin: 0; font-size: 20px; }
          .card-desc { color: var(--muted); margin: 7px 0 0; font-size: 13px; line-height: 1.5; }
          label { display: block; font-size: 13px; font-weight: 750; color: #334155; margin-bottom: 7px; }
          input, select { width: 100%; border: 1px solid var(--line); background: var(--panel-soft); color: var(--text); border-radius: 14px; padding: 12px 13px; font-size: 14px; outline: none; }
          input:focus, select:focus { border-color: var(--blue); background: #fff; box-shadow: 0 0 0 4px rgba(37,99,235,0.11); }
          button { border: none; cursor: pointer; border-radius: 999px; font-weight: 800; transition: transform .12s ease, opacity .12s ease; }
          button:hover { transform: translateY(-1px); }
          button:disabled { opacity: .7; cursor: default; transform: none; }
          .primary-btn { width: 100%; margin-top: 14px; padding: 13px 16px; color: white; background: linear-gradient(135deg, #2563eb, #16a34a); box-shadow: 0 14px 26px rgba(37,99,235,0.20); }
          .secondary-btn { padding: 12px 15px; background: #0f172a; color: white; white-space: nowrap; }
          .date-row { display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: end; }
          .money-row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
          .status-text { min-height: 20px; margin: 10px 0 0; color: var(--orange); font-size: 13px; font-weight: 650; }
          .status-text.ok { color: var(--muted); }
          .result-box { margin-top: 16px; border-radius: 14px; background: linear-gradient(180deg, #f8fafc, #ffffff); border: 1px solid var(--line); padding: 16px; }
          .result-label { color: var(--muted); font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: .06em; margin-bottom: 8px; }
          .result { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; white-space: pre-wrap; line-height: 1.65; font-size: 13px; }
          .tag { display: inline-flex; align-items: center; justify-content: center; min-width: 72px; padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 900; color: #fff; }
          .tag-up { background: var(--green); }
          .tag-down { background: var(--red); }
          .tag-neutral { background: #64748b; }
          .metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 14px; }
          .metric-card { border: 1px solid var(--line); background: #fff; border-radius: 16px; padding: 15px; }
          .metric-card span { display: block; color: var(--muted); font-size: 12px; font-weight: 750; margin-bottom: 8px; }
          .metric-card strong { display: block; font-size: 23px; }
          .chart-card { margin-top: 22px; }
          .chart-title-row { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin-bottom: 12px; }
          .legend-mini { display: flex; gap: 10px; flex-wrap: wrap; color: var(--muted); font-size: 12px; font-weight: 700; }
          .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 5px; }
          .dot.blue { background: var(--blue); }
          .dot.purple { background: var(--purple); }
          .dot.orange { background: var(--orange); }
          #priceChart { width: 100%; max-height: 360px; }
          .toggle-row { display: flex; flex-wrap: wrap; gap: 14px; margin: 12px 0 8px; }
          .toggle-item { display: inline-flex; align-items: center; gap: 7px; font-size: 13px; font-weight: 750; color: var(--muted); }
          .toggle-item input { width: auto; }
          .bar-list { display: grid; gap: 10px; }
          .bar-row { display: grid; grid-template-columns: 1.1fr 1.4fr auto; gap: 10px; align-items: center; font-size: 12px; }
          .bar-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #334155; font-weight: 800; }
          .bar-track { height: 9px; background: #e2e8f0; border-radius: 999px; overflow: hidden; }
          .bar-fill { height: 100%; background: linear-gradient(135deg, #2563eb, #16a34a); border-radius: 999px; }
          .profit, .positive { color: var(--green); }
          .loss, .negative { color: var(--red); }
          .disclaimer { margin-top: 18px; text-align: right; color: var(--muted); font-size: 12px; }
          @media (max-width: 1050px) { .triple-grid { grid-template-columns: 1fr; } }
          @media (max-width: 920px) { .hero { align-items: flex-start; flex-direction: column; } .dashboard-grid { grid-template-columns: 1fr; } }
          @media (max-width: 640px) { .page { padding: 22px 12px 34px; } .card, .chart-card { padding: 17px; } .date-row, .money-row { grid-template-columns: 1fr; } .secondary-btn { width: 100%; } .metric-grid { grid-template-columns: 1fr; } .chart-title-row { flex-direction: column; } .hero h1 { font-size: 30px; } }
        </style>
      </head>
      <body>
        <main class="page">
          <section class="hero">
            <div>
              <h1>Financial Forecast Dashboard</h1>
              <p>5-day trend forecasts, technical indicators, feature importance, and USD test-money simulation for the selected stock.</p>
            </div>
            <div class="hero-badge">Educational ML Project - Not Financial Advice</div>
          </section>

          <div class="dashboard-grid">
            <section class="card">
              <div class="card-header">
                <div>
                  <div class="eyebrow">Prediction Panel</div>
                  <h2>Run Prediction</h2>
                  <p class="card-desc">Choose a ticker and view the model's expected 5-day direction with upside probability.</p>
                </div>
              </div>

              <input id="tickerInput" type="hidden" value="TSLA" />
              <label for="tickerSelect">Ticker</label>
              <select id="tickerSelect" onchange="onSelectChange()">
                <option value="TSLA">TSLA</option>
                <option value="AAPL">AAPL</option>
                <option value="MSFT">MSFT</option>
              </select>
              <button class="primary-btn" id="predictBtn" onclick="predict()">Run Prediction</button>
              <div id="status" class="status-text"></div>

              <div class="result-box">
                <div class="result-label">Current Prediction</div>
                <div id="cardResult" class="result">No prediction has been run yet.</div>
              </div>

              <div class="result-box">
                <div class="result-label">Test Money (USD)</div>
                <div class="money-row">
                  <div>
                    <label for="initialCashInput">Starting Cash</label>
                    <input id="initialCashInput" type="number" min="100" step="100" value="10000" />
                  </div>
                  <div>
                    <label for="commissionInput">Commission (%)</label>
                    <input id="commissionInput" type="number" min="0" max="10" step="0.05" value="0.1" />
                  </div>
                  <div>
                    <label for="simulationDaysInput">Test Days</label>
                    <input id="simulationDaysInput" type="number" min="2" max="500" step="1" value="90" />
                  </div>
                </div>
                <button class="primary-btn" id="simulateBtn" onclick="runSimulation()">Run Test Money</button>
                <div id="simulationResult" class="result" style="margin-top:12px;">Test money has not been run yet.</div>
              </div>
            </section>

            <section class="card">
              <div class="card-header">
                <div>
                  <div class="eyebrow">Backtest Dashboard</div>
                  <h2>Model Performance</h2>
                  <p class="card-desc">Last-90-day accuracy summary and a backtest view for a selected day.</p>
                </div>
              </div>

              <div class="metric-grid">
                <div class="metric-card"><span>Accuracy</span><strong id="metricAccuracy">--</strong></div>
                <div class="metric-card"><span>Correct</span><strong id="metricCorrect">--</strong></div>
                <div class="metric-card"><span>Wrong</span><strong id="metricWrong">--</strong></div>
              </div>

              <div class="result-box">
                <div class="result-label">Last 90 Days</div>
                <div id="metricsResult" class="result">No data has been loaded yet.</div>
              </div>

              <div style="height:18px;"></div>
              <label for="backtestDate">Select Date</label>
              <div class="date-row">
                <input id="backtestDate" type="date" />
                <button class="secondary-btn" id="backtestBtn" onclick="runBacktest()">Run Backtest</button>
              </div>
              <div class="result-box">
                <div class="result-label">Backtest Result</div>
                <div id="backtestResult" class="result">No backtest has been run yet.</div>
              </div>
            </section>
          </div>

          <section class="chart-card">
            <div class="chart-title-row">
              <div>
                <div class="eyebrow">Technical Chart</div>
                <h2>Price Chart</h2>
                <p class="card-desc">Close-price line with optional RSI/MACD overlays.</p>
              </div>
              <div class="legend-mini">
                <span><i class="dot blue"></i>Close</span>
                <span><i class="dot purple"></i>RSI</span>
                <span><i class="dot orange"></i>MACD</span>
              </div>
            </div>
            <div class="toggle-row">
              <label class="toggle-item"><input id="toggleRSI" type="checkbox" onchange="refreshChart()" /> Show RSI</label>
              <label class="toggle-item"><input id="toggleMACD" type="checkbox" onchange="refreshChart()" /> Show MACD</label>
            </div>
            <canvas id="priceChart"></canvas>
          </section>

          <div class="triple-grid">
            <section class="card">
              <div class="eyebrow">Feature Importance</div>
              <h2>What Drives the Model?</h2>
              <p class="card-desc">The features the Random Forest model uses most in its decisions.</p>
              <div id="importanceList" class="bar-list" style="margin-top:16px;">Not loaded yet.</div>
            </section>
            <section class="card">
              <div class="eyebrow">Strategy Note</div>
              <h2>Test Money</h2>
              <p class="card-desc">This panel does not use real money. It tests model signals against historical prices with a virtual USD balance.</p>
              <div class="result-box">
                <div class="result-label">Signals</div>
                <div class="result">P(up) >= 55%: BUY\nP(up) <= 45%: SELL\nBetween: HOLD</div>
              </div>
            </section>

            <section class="card">
              <div class="eyebrow">Portfolio Simulator</div>
              <h2>Portfolio Strategy Simulation</h2>
              <p class="card-desc">
                Simple investment scenario based on model predictions.
              </p>

              <label for="portfolioInitialCash">Starting Cash (USD)</label>
              <input id="portfolioInitialCash" type="number" value="1000" />

              <button class="primary-btn" onclick="runPortfolioSim()">
                Run Simulation
              </button>

              <div class="result-box">
                <div class="result-label">Simulation Result</div>
                <div id="portfolioSimResult">Simulation has not been run yet.</div>
              </div>
            </section>

            <section class="card">
              <div class="eyebrow">Explainability Note</div>
              <h2>Model Commentary</h2>
              <p class="card-desc">Brief explanation of the model output.</p>

              <div class="result-box">
                <div class="result-label">WHAT CHANGED?</div>
                <div>
                  RSI/MACD overlays were added.<br>
                  Feature importance improves model explainability.<br>
                  Investment scenarios can be tested using portfolio simulation.
                </div>
              </div>
            </section>

          </div>

          <div class="disclaimer">Educational project only. Not financial advice.</div>
        </main>

        <script>
          let priceChart = null;
          let lastChartData = null;
          let currentTicker = "TSLA";

          function setStatus(msg, isError = false) {
            const statusDiv = document.getElementById("status");
            statusDiv.innerText = msg || "";
            statusDiv.className = isError ? "status-text" : "status-text ok";
          }

          function formatMoney(value) {
            return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 }).format(value);
          }

          function onSelectChange() {
            document.getElementById("tickerInput").value = document.getElementById("tickerSelect").value;
          }

          async function predict() {
            const t = document.getElementById("tickerInput").value.trim();
            const resultDiv = document.getElementById("cardResult");
            const btn = document.getElementById("predictBtn");
            if (!t) { setStatus("Please select a ticker.", true); return; }

            setStatus("Running prediction...");
            btn.disabled = true;
            btn.innerText = "Running...";

            try {
              const res = await fetch("/predict?ticker=" + encodeURIComponent(t));
              const data = await res.json();
              if (!res.ok) { setStatus("Prediction error: " + (data.detail || "Unknown error"), true); return; }

              setStatus("");
              const dir = data.prediction;
              const prob = (data.prob_up * 100).toFixed(1);
              let tagClass = "tag tag-neutral";
              let tagText = "NEUTRAL";
              if (dir === "up") { tagClass = "tag tag-up"; tagText = "UP"; }
              else if (dir === "down") { tagClass = "tag tag-down"; tagText = "DOWN"; }

              resultDiv.innerHTML =
                "Ticker: <b>" + data.ticker + "</b><br>" +
                "5-Day Expected Trend: <span class='" + tagClass + "'>" + tagText + "</span><br>" +
                "Probability of UP: <b>" + prob + "%</b>";

              currentTicker = t;
              await loadMetrics(t);
              await loadPriceChart(t);
              await loadFeatureImportance(t);
              await runSimulation();
            } catch (err) {
              setStatus("Request failed: " + err, true);
            } finally {
              btn.disabled = false;
              btn.innerText = "Run Prediction";
            }
          }

          async function loadMetrics(ticker) {
            const metricsDiv = document.getElementById("metricsResult");
            const metricAccuracy = document.getElementById("metricAccuracy");
            const metricCorrect = document.getElementById("metricCorrect");
            const metricWrong = document.getElementById("metricWrong");
            try {
              const res = await fetch("/metrics?ticker=" + encodeURIComponent(ticker) + "&last_n=90");
              const data = await res.json();
              if (!res.ok) {
                metricsDiv.innerText = "Metrics could not be loaded: " + (data.detail || "Unknown error");
                metricAccuracy.innerText = "--";
                metricCorrect.innerText = "--";
                metricWrong.innerText = "--";
                return;
              }
              const acc = (data.accuracy_last_n * 100).toFixed(1);
              metricAccuracy.innerText = acc + "%";
              metricCorrect.innerText = data.correct;
              metricWrong.innerText = data.wrong;
              metricsDiv.innerHTML =
                "Ticker: " + data.ticker + "<br>" +
                "Samples: " + data.samples + "<br>" +
                "Accuracy: " + acc + "% (" + data.correct + " correct, " + data.wrong + " wrong)";
            } catch (err) {
              metricsDiv.innerText = "Metrics request failed: " + err;
            }
          }

          async function runBacktest() {
            const t = document.getElementById("tickerInput").value.trim();
            const date = document.getElementById("backtestDate").value;
            const backtestDiv = document.getElementById("backtestResult");
            const btn = document.getElementById("backtestBtn");
            if (!t) {
              backtestDiv.innerText = "Please select a ticker.";
              setStatus("Please select a ticker.", true);
              return;
            }

            setStatus("Running backtest...");
            backtestDiv.innerText = "Calculating backtest...";
            btn.disabled = true;
            btn.innerText = "Calculating...";

            try {
              let url = "/backtest?ticker=" + encodeURIComponent(t);
              if (date) { url += "&date=" + encodeURIComponent(date); }
              const res = await fetch(url);
              const data = await res.json();
              if (!res.ok) {
                const detail = data.detail || "Unknown error";
                backtestDiv.innerText = "Backtest could not be loaded: " + detail;
                setStatus("Backtest error: " + detail, true);
                return;
              }

              setStatus("");
              const dir = data.prediction;
              const actual = data.actual_movement;
              const prob = (data.prob_up * 100).toFixed(1);
              const close = data.close_on_date.toFixed(2);
              const nextClose = data.close_after_5d.toFixed(2);
              const verdict = dir === actual ? "The model prediction would have been CORRECT." : "The model prediction would have been WRONG.";
              backtestDiv.innerText =
                "Date: " + data.date + "\\n" +
                "Prediction: " + dir.toUpperCase() + " (P(up)=" + prob + "%)\\n" +
                "Actual movement: " + close + " -> " + nextClose + " (" + actual.toUpperCase() + ")\\n" +
                verdict;
            } catch (err) {
              backtestDiv.innerText = "Backtest request failed: " + err;
              setStatus("Backtest request failed: " + err, true);
            } finally {
              btn.disabled = false;
              btn.innerText = "Run Backtest";
            }
          }

          async function runSimulation() {
            const t = document.getElementById("tickerInput").value.trim();
            const initialCash = Number(document.getElementById("initialCashInput").value || 0);
            const commission = Number(document.getElementById("commissionInput").value || 0);
            const days = Number(document.getElementById("simulationDaysInput").value || 90);
            const resultDiv = document.getElementById("simulationResult");
            const btn = document.getElementById("simulateBtn");

            if (!t) { setStatus("Please select a ticker.", true); return; }
            if (!initialCash || initialCash <= 0) {
              resultDiv.innerText = "Starting cash must be greater than 0.";
              setStatus("Starting cash must be greater than 0.", true);
              return;
            }
            

            btn.disabled = true;
            btn.innerText = "Simulating...";

            try {
              const url = "/simulate?ticker=" + encodeURIComponent(t) +
                "&initial_cash=" + encodeURIComponent(initialCash) +
                "&commission_pct=" + encodeURIComponent(commission) +
                "&last_n=" + encodeURIComponent(days);
              const res = await fetch(url);
              const data = await res.json();
              if (!res.ok) { resultDiv.innerText = "Simulation could not be loaded: " + (data.detail || "Unknown error"); return; }

              const profitClass = data.total_profit >= 0 ? "profit" : "loss";
              const buyHoldClass = data.buy_hold_profit >= 0 ? "profit" : "loss";
              const recentTrades = (data.recent_trades || []).map((trade) => {
                const pnl = trade.pnl === undefined ? "" : " | PnL " + formatMoney(trade.pnl);
                return trade.date + " | " + trade.action.toUpperCase() +
                  " | $" + trade.price.toFixed(2) +
                  " | P(up) " + (trade.prob_up * 100).toFixed(1) + "%" + pnl;
              }).join("\\n");

              resultDiv.innerHTML =
                "Starting cash: " + formatMoney(data.initial_cash) + "<br>" +
                "Test window: last " + data.samples + " trading days<br>" +
                "Final portfolio: <span class='" + profitClass + "'>" + formatMoney(data.final_value) + "</span><br>" +
                "Model profit/loss: <span class='" + profitClass + "'>" + formatMoney(data.total_profit) +
                  " (" + data.total_return_pct.toFixed(2) + "%)</span><br>" +
                "Buy & Hold: <span class='" + buyHoldClass + "'>" + formatMoney(data.buy_hold_profit) +
                  " (" + data.buy_hold_return_pct.toFixed(2) + "%)</span><br>" +
                "Trades: " + data.trade_count + " (" + data.buy_count + " buy, " + data.sell_count + " sell)<br>" +
                "Winning/Losing exits: " + data.winning_trades + " / " + data.losing_trades + "<br>" +
                "Max drawdown: " + data.max_drawdown_pct.toFixed(2) + "%<br>" +
                "Open position: " + data.shares + " shares, cash " + formatMoney(data.cash) +
                (recentTrades ? "<br><br>Recent trades:\\n" + recentTrades : "");
            } catch (err) {
              resultDiv.innerText = "Simulation request failed: " + err;
            } finally {
              btn.disabled = false;
              btn.innerText = "Run Test Money";
            }
          }
          async function runPortfolioSim() {
            const initialCash = Number(document.getElementById("portfolioInitialCash").value || 1000);
            const ticker = document.getElementById("tickerSelect").value;

            try {
              const res = await fetch(
                "/simulate?ticker=" + encodeURIComponent(ticker) +
                "&initial_cash=" + encodeURIComponent(initialCash)
              );

              const data = await res.json();
              
              if (!res.ok) {
                document.getElementById("portfolioSimResult").innerText =
                  "Simülasyon alınamadı: " + (data.detail || "Bilinmeyen hata");
                return;
              }

              document.getElementById("portfolioSimResult").innerHTML =
                "Initial Cash: $" + initialCash + "<br>" +
                "Final Value: $" + data.final_value + "<br>" +
                "Total Return: " + data.total_return_pct + "%";

            } catch (err) {
              document.getElementById("portfolioSimResult").innerText =
                "Simülasyon hatası: " + err;
            }
          }

          async function loadPriceChart(ticker) {
            try {
              const res = await fetch("/price_series?ticker=" + encodeURIComponent(ticker) + "&days=90");
              const data = await res.json();
              if (!res.ok) { setStatus("Price data could not be loaded: " + (data.detail || "Unknown error"), true); return; }
              lastChartData = data;
              renderPriceChart();
            } catch (err) {
              setStatus("Price data request failed: " + err, true);
            }
          }

          function refreshChart() {
            if (lastChartData) { renderPriceChart(); }
          }

          function renderPriceChart() {
            const data = lastChartData;
            if (!data) return;
            const labels = data.dates;
            const prices = data.prices;
            const showRSI = document.getElementById("toggleRSI").checked;
            const showMACD = document.getElementById("toggleMACD").checked;
            if (!prices || !labels || prices.length !== labels.length) {
              setStatus("Price data is not in the expected format.", true);
              return;
            }

            const datasets = [{
              label: "Close Price",
              data: prices,
              borderColor: "#2563eb",
              backgroundColor: "rgba(37,99,235,0.10)",
              tension: 0.25,
              pointRadius: 2,
              borderWidth: 2.5,
              fill: false,
              yAxisID: "priceAxis"
            }];

            if (showRSI && data.rsi) {
              datasets.push({
                label: "RSI",
                data: data.rsi,
                borderColor: "#7c3aed",
                tension: 0.25,
                pointRadius: 0,
                borderWidth: 2,
                spanGaps: true,
                fill: false,
                yAxisID: "rsiAxis"
              });
            }
            if (showMACD && data.macd) {
              datasets.push({
                label: "MACD",
                data: data.macd,
                borderColor: "#f97316",
                tension: 0.25,
                pointRadius: 0,
                borderWidth: 2,
                spanGaps: true,
                fill: false,
                yAxisID: "macdAxis"
              });
            }

            const ctx = document.getElementById("priceChart").getContext("2d");
            if (priceChart) { priceChart.destroy(); }
            priceChart = new Chart(ctx, {
              type: "line",
              data: { labels: labels, datasets: datasets },
              options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: { legend: { display: true } },
                scales: {
                  x: { ticks: { color: "#64748b", maxRotation: 0, autoSkipPadding: 16 }, grid: { color: "#f1f5f9" } },
                  priceAxis: { position: "left", beginAtZero: false, ticks: { color: "#64748b" }, grid: { color: "#f1f5f9" } },
                  rsiAxis: { display: showRSI, position: "right", min: 0, max: 100, grid: { drawOnChartArea: false }, ticks: { color: "#7c3aed" } },
                  macdAxis: { display: showMACD, position: "right", grid: { drawOnChartArea: false }, ticks: { color: "#f97316" } }
                }
              }
            });
          }

          async function loadFeatureImportance(ticker) {
            const list = document.getElementById("importanceList");
            list.innerText = "Loading...";
            try {
              const res = await fetch("/feature_importance?ticker=" + encodeURIComponent(ticker) + "&top_n=8");
              const data = await res.json();
              if (!res.ok) { list.innerText = "Feature importance could not be loaded: " + (data.detail || "Unknown error"); return; }
              const maxScore = Math.max(...data.features.map(f => f.importance));
              list.innerHTML = "";
              data.features.forEach(f => {
                const pct = maxScore > 0 ? (f.importance / maxScore) * 100 : 0;
                const row = document.createElement("div");
                row.className = "bar-row";
                row.innerHTML =
                  "<div class='bar-name' title='" + f.name + "'>" + f.name + "</div>" +
                  "<div class='bar-track'><div class='bar-fill' style='width:" + pct + "%'></div></div>" +
                  "<div>" + (f.importance * 100).toFixed(2) + "%</div>";
                list.appendChild(row);
              });
            } catch (err) {
              list.innerText = "Feature importance request failed: " + err;
            }
          }

          window.onload = () => { predict(); };
        </script>
      </body>
    </html>
    """


def safe_float(value):
    try:
        if value is None or pd.isna(value):
            return None
        return round(float(value), 4)
    except Exception:
        return None


@app.get("/price_series")
def price_series(ticker: str, days: int = 90):
    ticker = ticker.upper()
    try:
        hist = yf.download(ticker, period="180d", interval="1d", progress=False)
        if hist.empty:
            raise HTTPException(status_code=404, detail="No data found")
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)

        hist = hist.copy()
        hist.index = pd.to_datetime(hist.index).normalize()
        close = hist["Close"].astype(float)

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        hist["rsi_14"] = 100 - (100 / (1 + rs))

        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        hist["macd"] = ema_12 - ema_26

        hist = hist.tail(days).copy()
        dates = [idx.strftime("%Y-%m-%d") for idx in hist.index]
        prices = [round(float(v), 2) for v in hist["Close"].values]
        rsi_values = [safe_float(v) for v in hist["rsi_14"].values]
        macd_values = [safe_float(v) for v in hist["macd"].values]

        return {
            "ticker": ticker,
            "days": len(prices),
            "dates": dates,
            "prices": prices,
            "rsi": rsi_values,
            "macd": macd_values,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Price series could not be loaded: {e}")


@app.get("/feature_importance")
def feature_importance(ticker: str = "TSLA", period: str = "2y", top_n: int = 10):
    ticker = ticker.upper()
    try:
        _, X, _ = build_feature_dataframe(ticker, period=period)
        model = load_model(ticker)

        estimator = model.best_estimator_ if hasattr(model, "best_estimator_") else model
        if hasattr(estimator, "named_steps"):
            for step in estimator.named_steps.values():
                if hasattr(step, "feature_importances_"):
                    estimator = step
                    break

        if not hasattr(estimator, "feature_importances_"):
            raise HTTPException(
                status_code=400,
                detail="This model does not support feature_importances_. Make sure a RandomForest model is loaded.",
            )

        importances = estimator.feature_importances_
        feature_names = list(X.columns)
        if len(importances) != len(feature_names):
            raise HTTPException(
                status_code=500,
                detail=f"Feature count mismatch: model={len(importances)}, X={len(feature_names)}",
            )

        pairs = sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)[:top_n]
        return {
            "ticker": ticker,
            "features": [
                {"name": name, "importance": round(float(score), 6)}
                for name, score in pairs
            ],
        }
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature importance could not be loaded: {e}")
