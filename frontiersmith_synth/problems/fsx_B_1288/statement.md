# Velocity Rules Against a Rational Attacker

## Problem
You run fraud defense for a merchant. Every incoming transaction belongs to one
**entity** (a card/device fingerprint): its own private stream of `(time, amount)`
events. You publish a rule set once, before anyone acts:

```
V W A
```

A transaction is **blocked** iff, within its own entity's stream, its amount exceeds
`A`, **or** it is one of more than `V` transactions of that entity falling in the
trailing window `(t-W, t]`. (Two entities' events never interact.)

The merchant faces `K` **attack waves**. Each wave's raw pattern is `n0` transactions
of amount `a0`, packed into a window of width `w0` starting at a fixed time. The
attacker is **rational and adaptive**: for each wave, after seeing your published
`(V,W,A)`, it deterministically picks whichever of 7 recipes maximizes its own net
profit (value captured minus its own cost), and executes that recipe:
- `NONE` — skip the wave (net 0).
- `BASE` — send the raw pattern unchanged.
- `SPLIT` — break the `n0` transactions into `n0*m` transactions of amount `a0/m`
  each, same window `w0` (cost `cost_split * n0*(m-1)`).
- `DELAY_LO` / `DELAY_HI` — keep `n0` transactions of amount `a0` but spread them
  over a widened window `w0+d_lo` or `w0+d_hi` (cost `cost_delay * d`).
- `SPLIT_DELAY_LO` / `SPLIT_DELAY_HI` — combine both (cost = sum of both costs).
In every recipe, transactions are placed at evenly spaced times across the recipe's
window, and the attacker nets `value_frac * (amount that got through) - cost`.
Whatever fraud amount gets through is a loss to the merchant.

Separately, the merchant has `Lc` **legitimate clusters** (ordinary customers), each
also `(nl, tl, wl, al)`: `nl` genuine transactions of amount `al` spread over window
`wl` starting at `tl`. Any of these your rule blocks is **customer friction**.

*Illustrative FORM only (not this problem's mechanism):* imagine a single knob
"block anything over $500" — that toy example has no adaptive attacker and no
window logic; it only shows that a threshold trades recall against friction.

## Input (stdin)
```
T
n0 a0 w0 K
ts_1 ... ts_K
m d_lo d_hi
cost_split cost_delay value_frac
Lc
nl_1 tl_1 wl_1 al_1
...
nl_Lc tl_Lc wl_Lc al_Lc
c1 c2 p
Vmax Wmax Amax
```
`T` is horizon length (context only). `ts_k` are wave start times. Friction for a
rule set is `c1 * (sum of blocked legit amounts) + c2 * (count of blocked legit
transactions)^p`, with `p>1` (over-blocking is super-linearly punished).

## Output (stdout)
Exactly one line: `V W A` — `V` and `W` integers, `A` a real number.

## Feasibility
`0 <= V <= Vmax`, `1 <= W <= Wmax`, `0 <= A <= Amax`, all finite. Any violation
(wrong token count, non-numeric, out of range, non-finite) scores `Ratio: 0.0`.

## Objective
Let `fraud_prevented = value_frac * K * n0 * a0 - (value captured by the attacker's
chosen recipes across all K waves)`. Let `friction` be as defined above. Maximize
`F = fraud_prevented - friction`.

## Scoring
The checker computes its own reference `B > 0`: the fixed rule `V=0, W=1, A=0`
(block every transaction — no attacker- or customer-awareness needed), scored the
same way. With `Fc = max(0, F)`:
```
sc = min(1000.0, 100.0 * Fc / max(1e-9, B))
Ratio = sc / 1000.0
```
Matching `B` scores `0.1`; `10x` more net value caps at `1.0`.

## Constraints
- `3 <= K <= 6`, `4 <= Lc <= 8`, `4 <= n0 <= 8`, `1 <= w0 <= 3`, `3 <= m <= 5`.
- Time limit 5s, memory 512MB.

## Example
Suppose one wave has `n0=5, a0=100, w0=1`, `value_frac=0.5`, and evasion (any
recipe) always costs more than `value_frac*n0*a0 = 250`. Publishing `V=4, W=1,
A=99` blocks `BASE` outright (5 > V=4, and 100 > A=99); since no recipe is
profitable the attacker sends nothing, so `fraud_prevented = 250`. If no legit
cluster is caught, `friction=0` and `F=250`.
