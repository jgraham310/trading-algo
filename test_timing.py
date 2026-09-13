"""Self-checks for the timing strategy. No download, no model.

    python test_timing.py
"""
from trading_algo import engine, stocks, timing

engine.demo()   # backtest conventions, session split, leakage tripwire
timing.demo()   # signal warmup, look-ahead, sleeve behaviour
stocks.demo()   # top-k weights, rebalance schedule, warmup
print("all ok")
