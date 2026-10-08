# ShortSeller Playbook — Codified Rules for the Scanner

Source: "Annualize This" Substack (ShortSeller / Andy), archive label "Annualize This Archive", Mar–Sep 2026.
Purpose: the reference spec Claude Code builds against. Drop this file in the repo at `docs/SHORTSELLER_PLAYBOOK.md`.

Everything below is paraphrased and restated as testable rules. Where ShortSeller is discretionary (hand-drawn lines, "bullseyes"), the rule is marked **[DISCRETIONARY → APPROXIMATED]**. Where his published logic has a flaw, it is marked **[FIX]**.

---

## 0. What this methodology actually is (read before building)

- It is a **weekly-timeframe, scale-in position-building system** for a long-biased investor in high-beta tech. It is not a day-trading system. An algo built from it should make decisions on weekly closes (daily for alerts), and should add in tranches over weeks/months.
- His core claim is not "predict tops and bottoms." It is "map levels in advance where institutions buy or sell, and act at those levels."
- His self-reported biggest failure mode is **being early** (calling bottoms), not greed. He would rather pay up late than catch a falling knife. The scanner must encode that bias.
- His own results in this period were rough: he reported being down 20–40% at points, a −9% single day despite ~20% cash, and giving back gains. The rules below are a hypothesis set. Every rule must earn its place in a backtest (Section 9) before it touches money.

---

## 1. Universe and sleeves

- Start from **largest-TAM themes**: AI/semis, photonics, memory/storage, datacenter/neocloud, power/grid/nuclear/solar+BESS, space, robotics, defense tech, cybersecurity, quantum, gene editing.
- Split each theme into three pools:
  - **Quality**: real revenue and growth, still early in the cycle. Buy-and-hold core; ride through drawdowns; add on dips.
  - **Speculative ("shitcos")**: little or no revenue. Must be managed actively: cut or sharply size down when out of favor. He notes most sub-$10 stocks eventually fail.
  - **ETF**: the conservative way to own a theme (watch fees and liquidity).
- Portfolio shape he uses: a **barbell**, roughly half quality, half speculative adds, with a cash buffer he raises ahead of weakness.
- Don't chase "shiny new things" everyone is talking about. New names get small starter positions only.
- A stock being up 1,400% is not by itself a reason to skip it; great compounders pass through that level on the way to much more.

## 2. Market regime gate (always runs first)

Market is **"guilty until proven innocent"** when below its moving averages.

- **MA stack**: count how many of the 5/9/20/50/100/200-day SMAs SPY (and QQQ) closes above. Below all = dangerous. The more it is below, the less safe.
- **Downtrend line**: no full-size buying until the index closes above its active downtrend line. **[DISCRETIONARY → APPROXIMATED]** with a pivot-high trendline fit (Section 6).
- **VIX** above its own SMA = fear present.
- **Credit**: high-yield (JNK/HYG) breaking down is a warning for equities.
- **Watch-list macro tells** he tracks: KOSPI/EWY (leveraged Korean money in the AI/memory trade), WTI oil, global bond yields.
- **"Is it safe for new buying?" checklist** (his indicator #2, weighted score):
  - 13-week and 34-week MAs rising
  - Price above 13-week EMA or 50-day SMA
  - MACD histogram rising on daily and weekly
  - VIX trend falling
  - NYSE new lows < 500
  - Elder Impulse not red on weekly and daily
- **Market internals** (his "Masters Extremes" indicator): Zweig Breadth Thrust, Whaley breadth thrust, Follow-Through Day (day 4+ of a rally attempt with a price/volume surge), McClellan Oscillator extremes, equity put/call extremes (contrarian), NYSE TICK extremes, Lowry down-volume % (capitulation), Hindenburg Omen. Combined into a 0–100 "conviction score": >60 after a capitulation bottom = bullish; <40 with Hindenburg/greed/divergences = caution.
- **Drawdown frequency priors** he uses for expectations: ~3–4 dips of ~5% per year; ~3 per year of 5–10%; ~1 per year of 10–20%; a 20%+ roughly every 6 years.
- **Big gap-up days (≥2.5%)**: in high-stress regimes they often fade intraday; they hold when VIX does not accelerate in the next 48 hours.
- **Big down Fridays (QQQ < −3%)**: small-sample (n≈17) tendency toward a positive 2-week return. Treat as a weak prior, never a trigger.
- Options expiration weeks (monthly/quarterly opex) are noisy; widen tolerances or pause new entries.

Output: a regime state **RED / AMBER / GREEN** that caps how much the system may buy.

## 3. Stage analysis (Weinstein) — primary trend filter, weekly

His "Stage Sniper" logic, restated:

- 30-week SMA = anchor. Slope = current 30W SMA vs the 30W SMA computed 3 weeks ago.
- 8-week EMA = "speedline" (short-term momentum).
- Relative strength vs SPY must be positive for Stage 2.
- Stages:
  - **Stage 1 (basing)**: close > 30W MA, MA flat/falling. Watch only.
  - **Stage 2 (markup)**: close > 30W MA, MA rising, RS > 0. The only stage for new full-size longs.
  - **"1→2, wait for RS"**: close > rising MA but RS ≤ 0. Not yet.
  - **Stage 3 (topping)**: close < 30W MA, MA still flat/rising. Trim.
  - **Stage 4 (decline)**: close < 30W MA, MA falling. Avoid / exit speculative.
- Stage 2 with 8W EMA **broken** = early warning that momentum is fading.
- **[FIX]** His RS formula is an approximation: `((price/bench) / (SMA52(price)/SMA52(bench)) − 1) × 100`. True Mansfield RS is `(ratio / SMA52(ratio) − 1) × 100` where `ratio = price/bench`. Implement both; default to the true one.
- **[FIX]** His stage rules flip on a single weekly close. Add hysteresis (e.g., require 2 consecutive weekly closes, or a small % buffer around the MA) to stop flicker.

## 4. Buy zones and scaling in

- **Drawdown-from-high zones for quality leaders**: when a large-TAM leader is **20–35% (up to ~40%) off its all-time high**, start nibbling. He adds at 5%, 10%, 20% index-drawdown levels too, at prices mapped **in advance**.
- **Scale in slowly**: starter → add → add, over weeks to years. Never all-in at once.
- **Go heavier only after confirmation**: close back above the downtrend line, and/or a higher low.
- **Add higher on de-risking**: a company hitting a milestone can justify adding at higher prices.
- **Time vs price**: stocks can correct by going sideways (time), not only down (price). A long sideways base is a valid reset.
- **Channel bottoms** = where funds accumulate; **channel tops** = where funds distribute ("you are their exit liquidity").
- **Uptrend-line touches** (UTL) and prior demand zones are add levels.
- **Gaps** below price are magnets in corrections (e.g., "a 30% drawdown would close that gap").

## 5. Trend-health and trim/exit rules

- **Long upper wick** = buyer exhaustion. **Long lower tail** = dip buying.
- **Exhaustion top** (his Sniper "orange dot"): price pierces the **+2.5σ** band, **RSI > 70**, then an **immediate rejection**: a red candle closing below the prior bar's low. Trim, don't exit core.
- **Gatekeeper**: don't sell just because something is overbought. Sell trend positions only when price breaks the 50-day SMA (daily) or 10-week EMA (weekly/macro).
- **Riding the bands**: strong trends hold between +1σ and +2σ. A firm close back under +1σ = momentum decaying.
- **Trim at channel tops/resistance, add at channel bottoms/support**: his standard core-position management.
- **Stage 3/4 transitions**: trim quality, cut speculative.
- **Speculative sleeve**: punt or size down hard when the theme is out of favor; many never come back.
- **"No man's land"**: price between support and resistance with no structure = no action.

## 6. Price-structure signals

His most frequent bullish trigger in chart notes: **higher low + close above the downtrend line**. Also:

- **Downtrend/uptrend lines (DTL/UTL)** **[DISCRETIONARY → APPROXIMATED]**: fit a line through the last 2–3 confirmed pivot highs (DTL) or pivot lows (UTL). Signal on a close beyond the line, not an intrabar touch.
- **Higher low**: latest confirmed pivot low > previous pivot low.
- **Bull/bear flag**: a pole (one or more large % candles) then a tight sideways consolidation. Resolution usually follows the pole direction. He suggests the breakout move can approach ~90% of the pole height — **untested; must be verified**.
- **Wedges**: converging range; the breakout direction usually sets the next move.
- **Cup & handle / inverse head & shoulders**: measured-move targets.
- **Leaving a long-term channel** upward can allow a parabolic phase.
- **Inside candles** (esp. monthly/weekly) after a drop = pause, possibly a buy opportunity rather than a top.
- **VPOC** (volume point of control): the price with the most volume over a window = magnet/support/resistance.
- **Megaphone (broadening) patterns**: volatile, low-confidence.

## 7. Statistical-stretch signals

- **Sigma bands** (his "ShortSeller's Sigma"): 20-period SMA ± 1σ/2σ/3σ of close.
  - Close beyond ±3σ = extreme outlier → mean-reversion candidate toward the SMA.
  - Bands narrowing = volatility squeeze → breakout likely. Bands widening fast = trend start.
- **Washout tiers** (his "Sniper"):
  - Standard buy: Keltner channel lower pierce + low RSI.
  - Strong buy: RSI < 25.
  - Max buy ("washout"): price pierces the −3σ Bollinger band.
  - Lockout: after a signal fires, mute that signal for N bars to prevent clusters.
  - Auto-adapt: on weekly/monthly bars use faster params (e.g., 10 EMA instead of 50 SMA).
- **[FIX] Conflict to resolve**: washout buys are knife-catching, which contradicts his "late, not early" lesson. In the scanner, washout signals on quality names are only actionable when the regime gate is not RED, and on speculative names only as watch alerts.

## 8. Volume and momentum

- **Volume climax**: volume ≥ 2× its 20-period average at a price extreme = capitulation (at lows) or exhaustion (at highs).
- **Elliott-style proxy** (his "EW Divergence Sniper"): Awesome Oscillator (5/34 SMA of median price). Price and AO make new highs together = "wave 3" trend surge. Price makes a higher high but AO makes a lower high = "wave 5" exhaustion. Add a volume climax = high-conviction turn. Pivots must be confirmed (no repainting).
- **RSI and OBV divergences** = moves not supported by momentum or volume flow.
- **Monthly TSI/RSI color** he uses for long-term momentum resets; monthly RSI under its own MA = room to fall further.

## 9. Candle patterns (his indicator thresholds)

Defaults from his "Candles" teaching indicator: wick multiplier 2.0, opposite-wick max 1.0× body, doji body ≤ 10% of range, marubozu body ≥ 95% of range, tweezer tolerance 5% of range.
- Single: doji (standard/dragonfly/gravestone), hammer, shooting star, spinning top, bull/bear marubozu.
- Two-bar: bull/bear engulfing, bull/bear harami (inside bar), piercing line, dark cloud cover, tweezer top/bottom.
- Three-bar: morning/evening star, three white soldiers, three black crows.
He uses candles as **context**, not standalone triggers. Score them only at a mapped level (support, resistance, band extreme).

## 10. Risk and behavior rules (the part most people skip)

- Protecting capital is priority #1.
- Being early is the account killer. Prefer late confirmation over bottom-calling.
- Know your levels **in advance** and set alerts; don't decide in the moment.
- Raise cash when the regime deteriorates; nibble only while the market is unsafe.
- No revenge trading / averaging down because "this drop is overdone."
- Expect drawdowns; size so you can hold through them.
- Trim at resistance, rebuy at support on core positions.
- Too many positions becomes unmanageable; he is cutting position count.

## 11. What does not translate to code (keep human)

- Theme/TAM selection and the quality-vs-speculative call per ticker.
- His hand-drawn bullseye levels (the scanner approximates; it will not match his charts).
- Narrative judgment (fraud probes, dilutive raises, competitive threats like SpaceX vs ASTS).
