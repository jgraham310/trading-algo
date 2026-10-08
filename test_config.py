from trading_algo.config import load

c = load()
assert set(c["universe"]) == {"quality", "speculative", "etf"}
assert c["execution"]["mode"] == "paper" and not c["execution"]["live_enabled"]
assert c["risk"]["long_only"] and c["costs"]["fill"] == "next_open"
assert sum(c["regime"]["weights"].values()) == 100
print("config ok")
