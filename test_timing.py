"""Self-checks for the timing strategy. No download, no model.

    python test_timing.py
"""
from trading_algo import engine, mandate, options, overfit, stocks, stress, timing

engine.demo()   # backtest conventions, session split, leakage tripwire
timing.demo()   # signal warmup, look-ahead, sleeve behaviour
stocks.demo()   # top-k weights, rebalance schedule, warmup
overfit.demo()  # condition bank cannot read the future; position is lagged one bar
options.demo()  # put-call parity identity; recovered premia are real option prices
mandate.demo()  # withdrawal schedule, depletion, drawdowns deepen not shrink
stress.demo()   # window boundaries; sustainable rate tracks the growth rate
print("all ok")
