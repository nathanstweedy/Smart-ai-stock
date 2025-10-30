from flask import Flask, render_template_string, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from textblob import TextBlob

app = Flask(__name__)

WATCHLIST = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN"]
FINNHUB_API_KEY = "YOUR_FINNHUB_API_KEY"

HTML = """[PASTE YOUR HTML TEMPLATE HERE]"""  # Keep the HTML from your original code

# -------------------- UTILITIES --------------------
def get_sentiment(symbol):
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={week_ago}&to={today}&token={FINNHUB_API_KEY}"
        data = requests.get(url).json()
        if not isinstance(data, list) or not data:
            return 0.0
        headlines = [d.get("headline", "") for d in data[:10]]
        polarity = np.mean([TextBlob(h).sentiment.polarity for h in headlines])
        return round(polarity, 3)
    except:
        return 0.0

def add_indicators(df):
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df.dropna(inplace=True)
    return df

def predict_next(symbol):
    end = datetime.now()
    start = end - timedelta(days=90)
    df = yf.download(symbol, start=start, end=end)
    if df.empty:
        return "neutral", 0, 0, 0
    df = add_indicators(df)
    sentiment = get_sentiment(symbol)
    df["MA_ratio"] = df["MA5"] / df["MA20"]
    trend_strength = df["MA_ratio"].iloc[-1]
    direction = "up" if trend_strength > 1 else "down"
    confidence = round(abs(trend_strength - 1) * 100 + abs(sentiment) * 50, 2)
    predicted_price = df["Close"].iloc[-1] * (1.01 if direction == "up" else 0.99)
    return direction, confidence, sentiment, round(predicted_price, 2)

@app.route("/")
def index():
    movers = []
    for s in WATCHLIST:
        direction, conf, sentiment, _ = predict_next(s)
        movers.append({"symbol": s, "direction": direction, "confidence": conf, "sentiment": sentiment})
    movers = sorted(movers, key=lambda x: x["confidence"], reverse=True)
    return render_template_string(HTML, top_movers=movers, watchlist=WATCHLIST)

@app.route("/chart-data")
def chart_data():
    symbol = request.args.get("symbol", "AAPL")
    end = datetime.now()
    start = end - timedelta(days=30)
    df = yf.download(symbol, start=start, end=end)
    if df.empty:
        return jsonify({"labels": [], "prices": []})
    labels = [d.strftime("%Y-%m-%d") for d in df.index]
    prices = [round(float(p), 2) for p in df["Close"]]
    direction, conf, sentiment, pred_price = predict_next(symbol)
    return jsonify({
        "labels": labels,
        "prices": prices,
        "pred_direction": direction,
        "pred_price": pred_price,
        "sentiment": sentiment
    })

if __name__ == "__main__":
    app.run(debug=True)
