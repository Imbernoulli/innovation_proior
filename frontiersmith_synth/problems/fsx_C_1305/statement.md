# Two-Sided Quotes on a Tape That Talks Back

You run a small market-making desk on one instrument for a fixed session of `T`
ticks. Every tick you have already posted a bid and an ask around the last
observed price, using a policy you committed to **once, before the session
starts** (there is no interaction — you never see the live session unfold).

Two kinds of counterparties can trade against your quotes. **Uninformed (noise)**
flow trades for liquidity reasons and predicts nothing about where the price is
headed; it fills against you more when your spread is tighter. **Informed** flow
only shows up shortly before the price is about to make a real move, and only
trades against you when your quote is actually mispriced relative to where the
price is heading — it will buy your ask if your ask is below the price's near
destination, and sell your bid if your bid is above it. Quoting the tightest
possible symmetric spread maximizes how much noise flow you capture, but it also
lets informed flow pick you off for free, right before the move happens.
Managing this requires two kinds of skew: shading your quotes toward flat as
your **inventory** grows in one direction, and shading your quotes in the
direction that recent **order-flow imbalance (OFI)** predicts the price is about
to go.

## Your policy

You submit exactly three numbers, applied identically every tick by the
evaluator:
```
skew   = inv_coef * (inventory / Qmax)  +  ofi_coef * ofi_t
bid_t  = last_price - half_spread - skew
ask_t  = last_price + half_spread - skew
```
`inventory` is your running signed position (clipped to `[-Qmax, Qmax]`, never
directly visible to you before the session). `ofi_t` is a causal feature — net
signed order-flow over the last `ofi_window` ticks, using only ticks strictly
before `t` — that the evaluator computes and plugs in each tick; you never see
the live tick data. `skew` is clamped to `[-max_skew, max_skew]`.

Noise flow fills in FULL when `half_spread <= fill_band[0]`, fills NOT AT ALL
when `half_spread >= fill_band[1]`, and interpolates linearly in between.
Informed flow fills in full whenever your quote is on the wrong side of its
(hidden) target, else not at all. Fills are clipped so inventory never exceeds
`Qmax`. Each tick accrues an inventory-risk cost proportional to
`inventory^2`. At the end, remaining inventory is marked at the final price.

## What you're given (public instance, on stdin)

```json
{
  "name": "mix_a", "T": 160, "Qmax": 40,
  "hs_bounds": [0.005, 50.0], "max_skew": 5.0,
  "fill_band": [0.15, 0.45], "ofi_window": 5, "ret_horizon": 6,
  "vol_hint": 0.05,
  "calibration": {
    "n": 160,
    "ofi": [0.0, -1.0, 3.0, ...],
    "next_ret": [0.02, -0.31, 0.88, ...]
  }
}
```
`calibration` is a RESOLVED, independent past session from the same regime (same
mechanics, different draws) — not the live session. `calibration.ofi[i]` is the
causal OFI feature at that historical tick, and `calibration.next_ret[i]` is the
REALIZED price change from that tick to `ret_horizon` ticks later. This is your
only chance to learn how predictive order flow is in this regime before
committing your policy.

## Answer (stdout)

```json
{"half_spread": 0.12, "inv_coef": 0.15, "ofi_coef": -0.004}
```
`half_spread` finite in `[0, 1e6]`; `inv_coef`, `ofi_coef` finite in
`[-1e4, 1e4]`. Anything else — wrong type, missing key, NaN/Inf, a crash, a
timeout, or non-JSON output — scores that instance `0.0`.

## Scoring (deterministic)

The evaluator re-simulates the live session tick by tick with your `(h, a, b)`,
producing `pnl_cand` (cash + final mark-to-market − accumulated inventory-risk
cost). It also computes, itself, `pnl_oracle`: the best PnL over a fixed grid of
`(h, a, b)` triples on the same live session (using full hidden information you
never see — a reference, not a target you can reach). Never trading is the
anchor:
```
r = clamp(0.1 + 0.9 * pnl_cand / max(pnl_oracle, 1e-6), 0, 1)
```
The reported **Ratio** is the mean of `r` over 10 seeded sessions: 3 pure-noise
warm-ups (no informed flow at all) and 7 mixed sessions with informed flow of
varying intensity, some larger/held-out. Because the grid is coarse, even a very
good policy stays below the oracle reference — there is real headroom.

## Suggested strategies

1. **Fixed tight symmetric spread** (no skew at all): maximizes noise fill rate,
   fine when there's no informed flow, gets run over when there is.
2. **Inventory-only skew**: mean-revert your position, but still walk straight
   into predictable moves you never anticipated.
3. **OFI-only skew** with a guessed coefficient: reacts to flow but with no
   principled sizing, and ignores position risk.
4. **Calibration-driven joint policy**: regress `next_ret` on `ofi` in the
   calibration window to size `ofi_coef` with the right sign and horizon-aware
   magnitude, and size `inv_coef`/`half_spread` off the volatility hint —
   combining both skews to move out of the way before the move happens.
