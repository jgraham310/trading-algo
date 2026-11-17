"""Research-only point-in-time observations and long-stock next-open event studies.

No broker, portfolio sizing, historical signal reconstruction, or options pricing.
Bars must contain explicit exchange session open/close timestamps, not dates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

import pandas as pd


def timestamp(value):
    t = pd.Timestamp(value)
    if pd.isna(t) or t.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return t.tz_convert("UTC")


@dataclass(frozen=True)
class Observation:
    signal_id: str
    symbol: str
    observed_at: str
    data_as_of: str
    source_url: str
    source_kind: str  # hand_drawn, engine, reconstruction
    timeframe: str
    pattern: str
    trigger: float
    invalidation: float
    target: float
    confirmation_rule: str
    disposition: str = "eligible"  # retain rejected/watch observations too
    reason: str = ""
    revision: int = 1

    def validate(self):
        observed, cutoff = timestamp(self.observed_at), timestamp(self.data_as_of)
        if cutoff > observed:
            raise ValueError("data cutoff follows observation")
        if not self.signal_id or not re.fullmatch(r"[A-Z0-9.^=-]+", self.symbol):
            raise ValueError("signal identity and normalized symbol required")
        if self.source_kind not in {"hand_drawn", "engine", "reconstruction"}:
            raise ValueError("unknown source kind")
        if self.disposition not in {"eligible", "watch", "rejected"}:
            raise ValueError("unknown disposition")
        if not all((self.source_url.startswith("https://"), self.timeframe,
                    self.pattern, self.confirmation_rule)):
            raise ValueError("source and frozen rule metadata required")
        if self.disposition == "rejected" and not self.reason:
            raise ValueError("rejected observations require a reason")
        if self.revision < 1 or (self.disposition != "rejected" and not all(
                v is not None and math.isfinite(v) and v > 0 for v in
                (self.trigger, self.invalidation, self.target))):
            raise ValueError("positive finite levels and revision required")


def capture(directory, observation: Observation):
    """Create one immutable revision file, never overwrite; not tamper-proof storage.

    recorded_at is the actual ingestion clock, separate from reported observed_at.
    Preserve raw evidence separately. A reconstruction is never a live AFPR record.
    """
    observation.validate()
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    # IDs cannot escape the capture directory.
    if not re.fullmatch(r"[A-Za-z0-9_-]+", observation.signal_id):
        raise ValueError("unsafe signal_id")
    prior = sorted(folder.glob(f"{observation.signal_id}.*.json"))
    if observation.revision != len(prior) + 1:
        raise ValueError("revisions must be contiguous, starting at one")
    if prior:
        old = json.loads(prior[-1].read_text())
        if (old["symbol"] != observation.symbol or
                timestamp(old["observed_at"]) >= timestamp(observation.observed_at)):
            raise ValueError("revision must preserve symbol and advance observation time")
    path = folder / f"{observation.signal_id}.{observation.revision:06d}.json"
    record = asdict(observation)
    record["recorded_at"] = datetime.now(timezone.utc).isoformat()
    with path.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    return path


def validate_bars(bars):
    """No dropping/forward-filling missing sessions. Caller supplies full calendar."""
    b = bars.copy()
    for col in ("open_at", "close_at"):
        b[col] = b[col].map(timestamp)
    if b.empty:
        return b
    if not b.open_at.is_monotonic_increasing or b.open_at.duplicated().any():
        raise ValueError("bars must be unique and chronological")
    if (b.close_at <= b.open_at).any() or (b.open_at.iloc[1:].reset_index(drop=True)
            <= b.close_at.iloc[:-1].reset_index(drop=True)).any():
        raise ValueError("overlapping or invalid sessions")
    for col in ("open", "close"):
        if not b[col].map(lambda v: pd.notna(v) and math.isfinite(v) and v > 0).all():
            raise ValueError("missing or invalid price; do not silently skip bars")
    return b.reset_index(drop=True)


def event_study(observation: Observation, bars: pd.DataFrame, *, horizon=20,
                cost_bps=10.0, benchmark: pd.DataFrame | None = None):
    """Enter first open strictly after a confirmed observation, exit next open.

    Caller records a *confirmed* signal, with confirmation_rule as evidence;
    this function does not infer a breakout or historical publication time.
    Close <= invalidation / >= target triggers next-open exit. Otherwise exit
    after horizon closes. Gap fills are actual opens, never stop/target prices.
    Costs are per side (commission + half-spread + slippage), cash-funded one
    share. All prices must use one consistent split-adjustment basis. No taxes
    or dividend cash flows modeled. Results are events, not a portfolio return.
    """
    observation.validate()
    if horizon < 1 or not isinstance(horizon, int) or not 0 <= cost_bps < 10000:
        raise ValueError("positive integer horizon and valid per-side costs required")
    if observation.disposition != "eligible":
        return {"status": "not_eligible", "signal_id": observation.signal_id}
    if not observation.invalidation < observation.trigger < observation.target:
        raise ValueError("this lane supports long-stock levels only")
    b = validate_bars(bars)
    candidates = b.index[b.open_at > timestamp(observation.observed_at)]
    result = {"signal_id": observation.signal_id, "revision": observation.revision,
              "source_kind": observation.source_kind, "status": "pending_entry",
              "net_return": None, "benchmark_return": None, "excess_return": None,
              "r_multiple": None, "cost_bps_per_side": cost_bps}
    if not len(candidates):
        return result
    start = int(candidates[0])
    entry = float(b.loc[start, "open"])
    result.update(entry_at=b.loc[start, "open_at"].isoformat(), entry=entry)
    # Do not enter a setup already invalidated or beyond its target at the open.
    if not observation.invalidation < entry < observation.target:
        return {**result, "status": "skipped_gap"}
    for i in range(start, min(start + horizon, len(b))):
        close = b.loc[i, "close"]
        reason = ("invalidation" if close <= observation.invalidation else
                  "target" if close >= observation.target else
                  "horizon" if i == start + horizon - 1 else None)
        if reason:
            if i + 1 >= len(b):
                return {**result, "status": "pending_exit", "exit_reason": reason}
            exit_price = float(b.loc[i + 1, "open"])
            cost = cost_bps / 10000
            pnl = exit_price * (1 - cost) - entry * (1 + cost)
            result.update(status="resolved", exit_reason=reason,
                          exit_at=b.loc[i + 1, "open_at"].isoformat(), exit=exit_price,
                          gross_return=exit_price / entry - 1,
                          net_return=pnl / (entry * (1 + cost)),
                          r_multiple=pnl / (entry - observation.invalidation),
                          # Close-sampled, not intraday excursions; include exit gap.
                          close_mae=min(entry, b.loc[start:i, "close"].min(), exit_price) / entry - 1,
                          close_mfe=max(entry, b.loc[start:i, "close"].max(), exit_price) / entry - 1)
            if benchmark is not None:
                bm = validate_bars(benchmark).set_index("open_at")
                first, last = b.loc[start, "open_at"], b.loc[i + 1, "open_at"]
                if first in bm.index and last in bm.index:
                    br = bm.loc[last, "open"] * (1-cost) / (bm.loc[first, "open"] * (1+cost)) - 1
                    result.update(benchmark_return=float(br), excess_return=result["net_return"]-br)
            return result
    return {**result, "status": "pending_horizon"}
