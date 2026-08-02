# Layering Reinsurance Against a Ruinous Year

## Problem

You run an insurer with starting capital `C0`. A catalog of `M` excess-of-loss
reinsurance layers is on offer, forming a **contiguous tower** directly above
your retention: layer `i` (`1..M`, sorted by attachment) is described by
`A_i W_i Prem_i K_i RP_i` -- it attaches at loss level `A_i` and exhausts at
`A_i + W_i` (so `A_{i+1} = A_i + W_i`); buying it at 100% participation costs
premium `Prem_i`; it may **reinstate** up to `K_i` extra times per year; each
reinstatement (full or partial) costs `RP_i` percent of the amount just
recovered, billed immediately from your capital.

You choose an integer **participation share** `p_i` in `[0,100]` for every
layer (a % placement -- fractional buying is normal in reinsurance), subject
to a premium budget: `sum(p_i * Prem_i) <= 100 * Pmax`.

**How a layer pays.** For a loss event of size `X`, layer `i` (bought at
share `p_i`) recovers `occ = min(max(X - A_i, 0), W_i)` scaled by `p_i/100`
-- capped at `W_i`, the layer's own width (**occurrence cap**). But every
layer also has a hard **annual aggregate capacity**, `p_i/100 * (K_i+1) *
W_i`, drawn down by every recovery it pays all year; once exhausted it pays
nothing more, no matter how well the loss would otherwise fit its band
(**aggregate-vs-occurrence**, **reinstatement-limits**). A layer you did not
buy pays nothing for its band, ever -- if a bigger loss punches through, only
the OTHER layers you own catch what falls in their own bands; nothing
magically covers a hole (**layer-attachment-exhaustion**).

**A policy year.** `S` scenarios are given, each a chronological list of loss
events. Simulate each one: start capital `= C0 -` (your total premium, share
-scaled), then for each event subtract `(loss - total recovered)` plus any
reinstatement costs. If capital ever goes negative, the year is a **ruin**:
it stops there, and that scenario's outcome is `3 x` the (negative) capital at
the moment of breach -- a harsher, continuous penalty for a deeper shortfall.
If capital never goes negative, the outcome is the final capital.

**Objective** = the average outcome over all `S` scenarios. Maximize it.

## Input (stdin)
```
M
A_1 W_1 Prem_1 K_1 RP_1
...
A_M W_M Prem_M K_M RP_M
C0 Pmax
S
n_1 e_1_1 ... e_1_n1
...
n_S e_S_1 ... e_S_nS
```
`1 <= A_1`, integers throughout, `M = 6`, `S = 40`.

## Output (stdout)
`M` integers `p_1 ... p_M`, each in `[0,100]`.

## Feasibility
Exactly `M` tokens, each a finite integer in `[0,100]`, with
`sum(p_i * Prem_i) <= 100 * Pmax`. Any violation scores `0`.

## Scoring
Let `F` be your average outcome (bigger is better). The checker also builds
its own reference program `B`: the same *equal* participation share on every
layer, as large as the budget allows. Your score is
`min(1000, 100 * F / max(1e-9, B)) / 1000`, clamped at 0, so matching the
reference scores about `0.1` and beating it scores higher (capped at `1.0`).

## Constraints
Time limit 5s, memory 512MB. 10 test cases of increasing scale.

## Example (worked, illustrative only -- smaller than the real tests, and a
different shape than the actual catalog/scenario data)

Toy catalog: `L0: A=5 W=5 Prem=5 K=0 RP=50`, `L1: A=10 W=5 Prem=3 K=1 RP=20`,
`L2: A=15 W=90 Prem=2 K=2 RP=10`. `C0=12, Pmax=5`.

Rate-on-line (`Prem/W`) is cheapest for `L2` (0.022), then `L1` (0.6), then
`L0` (1.0, priciest). A "buy the cheapest rate-on-line first" shopper spends
the whole budget on `L2` (cost 2) then `L1` (cost 3) -- skipping `L0`
entirely, leaving `[5,10)` (right above retention) completely uncovered.

One scenario, one event: loss `8`, landing in that uncovered band. Capital
starts at `12 - 5 = 7`; recovery is `0`; capital falls straight to
`7 - 8 = -1` -- **ruin**, even though this shopper's catalog-wide nominal
coverage width (`5 + 90 = 95`) dwarfs the alternative. Spending the same
budget on `L0` alone instead (skipping the wide, rarely-needed `L2`
entirely) keeps `[5,10)` covered: recovery `3` (capped by `L0`'s width),
retained `5`, a reinstatement cost of `50% * 3 = 1.5`, capital
`7 - 5 - 1.5 = 0.5` -- survives.
