import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

WATCHLIST = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN"]

def add_indicators(df):
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df.dropna(inplace=True)
    return df

def train_model(symbol):
    end = datetime.now()
    start = end - timedelta(days=180)
    df = yf.download(symbol, start=start, end=end)
    if df.empty:
        return None
    df = add_indicators(df)
    df["MA_ratio"] = df["MA5"] / df["MA20"]
    # Simple predictive model: last MA_ratio + last close
    model = {
        "last_close": df["Close"].iloc[-1],
        "last_ma_ratio": df["MA_ratio"].iloc[-1]
    }
    return model

def main():
    all_models = {}
    for s in WATCHLIST:
        model = train_model(s)
        if model:
            all_models[s] = model

    joblib.dump(all_models, "model.pkl")
    print("Models updated and saved as model.pkl")

if __name__ == "__main__":
    main()
