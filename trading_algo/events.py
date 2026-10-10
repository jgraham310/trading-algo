"""Event tags for a Bollinger hit: was the weekly move probably news-driven?

Deterministic proxies only -- "no tag" means "no identified event", NOT "no news" (analyst
rating changes, sector/theme moves, short reports, rumours and macro news are invisible here).
Tags are columns, not a filter: test whether they matter on the history before filtering on them.

Price tags (daily bars, signal week = Mon..Fri ending the Friday label; "prior" = bars before it):
  gap_atr      largest overnight gap in the week, in units of the prior 20-day average daily range
  day_conc     largest single-day |return| / sum of the week's daily |returns|  (1.0 = one-day move)
  vol_spike    largest daily volume in the week / prior 20-day average volume
  resid_1w     weekly return minus beta x SPY's weekly return (beta: prior 60 daily returns)
Filing tags (SEC EDGAR, free, no key; 14 days ending that Friday):
  filing_event   1 if the issuer filed an 8-K (or 6-K) -- NaN if the ticker has no CIK / fetch failed
  earnings_event 1 if an 8-K carried Item 2.02 (results). Foreign issuers (6-K) aren't detected.
  filing_items   the 8-K item numbers, ';'-joined
"""

from __future__ import annotations

import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

PRICE_COLS = ["gap_atr", "day_conc", "vol_spike", "resid_1w"]
FILING_COLS = ["filing_event", "earnings_event", "filing_items"]
EVENT_COLS = PRICE_COLS + FILING_COLS
FORMS = {"8-K", "8-K/A", "6-K"}
# SEC rejects (403) a User-Agent with no contact email. Set SEC_USER_AGENT="trading-algo-research you@example.com";
# keep the address out of the (public) repo -- it comes from the environment / a GitHub secret.
UA = os.environ.get("SEC_USER_AGENT", "trading-algo-research")


def price_tags(d: pd.DataFrame, spy: pd.DataFrame, friday: pd.Timestamp) -> dict | None:
    """Price/volume event signature of the week ending `friday` from daily OHLCV, or None if too short."""
    wk = d.loc[friday - pd.Timedelta(days=6):friday]
    prior = d[d.index < wk.index[0]] if len(wk) else d.iloc[:0]
    if len(wk) == 0 or len(prior) < 25:
        return None
    chain = pd.concat([prior["Close"].iloc[-1:], wk["Close"]])
    gap = (wk["Open"].to_numpy() / chain.iloc[:-1].to_numpy() - 1)
    day = chain.pct_change().iloc[1:].abs()
    rng = ((prior["High"] - prior["Low"]) / prior["Close"].shift(1)).iloc[-20:].mean()
    vol = prior["Volume"].iloc[-20:].mean()
    out = {"gap_atr": round(float(np.nanmax(np.abs(gap)) / rng), 2) if rng > 0 else np.nan,
           "day_conc": round(float(day.max() / day.sum()), 3) if day.sum() > 0 else np.nan,
           "vol_spike": round(float(wk["Volume"].max() / vol), 2) if vol > 0 else np.nan,
           "resid_1w": np.nan}
    s_all = spy["Close"]
    s_prior, s_wk = s_all[s_all.index < wk.index[0]], s_all.loc[wk.index[0]:friday]
    if len(s_wk) and len(s_prior) > 30:
        r = prior["Close"].pct_change().iloc[-60:]
        sr = s_all.pct_change().reindex(r.index)
        ok = r.notna() & sr.notna()
        if ok.sum() > 30 and sr[ok].var() > 0:
            beta = np.cov(r[ok], sr[ok])[0, 1] / sr[ok].var()
            out["resid_1w"] = round(float(chain.iloc[-1] / chain.iloc[0] - 1
                                          - beta * (s_wk.iloc[-1] / s_prior.iloc[-1] - 1)), 3)
    return out


class Edgar:
    """Minimal EDGAR client: ticker -> CIK map once, submissions cached per ticker, <=10 req/s."""

    def __init__(self, ua: str = UA):
        self.ua, self.cache, self.ciks, self.failed, self.last_error = ua, {}, None, 0, None

    def _get(self, url: str):
        time.sleep(0.12)
        req = urllib.request.Request(url, headers={"User-Agent": self.ua})
        return json.loads(urllib.request.urlopen(req, timeout=30).read())

    def filings(self, ticker: str) -> pd.DataFrame | None:
        """Recent 8-K/6-K filings (date, form, items) for `ticker`; None = unknown (no CIK / fetch failed)."""
        if ticker in self.cache:
            return self.cache[ticker]
        out = None
        try:
            if self.ciks is None:
                self.ciks = {v["ticker"].upper(): v["cik_str"]
                             for v in self._get("https://www.sec.gov/files/company_tickers.json").values()}
            cik = self.ciks.get(ticker.upper())
            if cik is not None:
                r = self._get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json")["filings"]["recent"]
                f = pd.DataFrame({"date": pd.to_datetime(r["filingDate"]), "form": r["form"], "items": r["items"]})
                out = f[f["form"].isin(FORMS)]
        except Exception as e:
            self.failed, self.last_error = self.failed + 1, repr(e)[:200]
        self.cache[ticker] = out
        return out


def tag(h: pd.DataFrame, data: dict[str, pd.DataFrame], spy: pd.DataFrame | None,
        edgar: Edgar | None = None) -> pd.DataFrame:
    """Fill missing event tags on history rows `h` (columns week, ticker). Idempotent; retries NaNs."""
    h = h.copy()
    for c in EVENT_COLS:
        if c not in h:
            h[c] = np.nan if c != "filing_items" else None
    h["filing_items"] = h["filing_items"].astype(object)
    edgar = edgar or Edgar()
    if spy is not None:
        for i in h.index[h["gap_atr"].isna()]:
            d = data.get(h.at[i, "ticker"])
            if d is None or not len(d):
                continue
            t = price_tags(d, spy, pd.Timestamp(h.at[i, "week"]))
            if t:
                for k, v in t.items():
                    h.at[i, k] = v
    for i in h.index[h["filing_event"].isna()]:
        f = edgar.filings(h.at[i, "ticker"])
        if f is None:
            continue
        fri = pd.Timestamp(h.at[i, "week"])
        w = f[(f["date"] > fri - pd.Timedelta(days=14)) & (f["date"] <= fri)]
        items = sorted({x for s in w["items"] for x in str(s).split(",") if x})
        h.at[i, "filing_event"] = int(len(w) > 0)
        h.at[i, "earnings_event"] = int("2.02" in items)
        h.at[i, "filing_items"] = ";".join(items)
    if getattr(edgar, "failed", 0):   # NaN, never 0: unknown is not 'no filing'. Retried on the next run.
        print(f"WARNING: EDGAR failed for {edgar.failed} ticker(s); filing tags left blank. Last error: {edgar.last_error}")
    return h
