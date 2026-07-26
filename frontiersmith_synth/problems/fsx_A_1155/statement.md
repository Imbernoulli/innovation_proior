# Scrip-Ward: Monetary Policy for a Nurse Shift-Trading Market

## Problem
A hospital ward lets its `N` nurses trade one flexible shift-slot per week using
non-transferable-from-outside token **scrip**. Each week `r` (round `r = 1..R`), nurse
`i` has a printed *disutility-relief value* `v[r][i] >= 0`: how much relief they would
get if they win that week's slot. Winning is decided by a fixed **priority-bidding
rule**, replayed week by week: nurse `i` bids `bid_i = min(floor(v[r][i]*ALPHA_NUM /
ALPHA_DEN), wallet_i)` (their natural interest in the slot, capped by scrip actually in
their wallet). The highest bidder wins (ties broken by the smallest nurse index);
the winner receives relief `v[r][winner]` and **pays their bid to the other `N-1`
nurses**, split as evenly as possible (`bid // (N-1)` each, the remainder `bid % (N-1)`
going one extra unit to the lowest-indexed non-winners) -- scrip never leaves the
system, it only moves. After the trade, a **demurrage tax** at rate `T[r]/TAX_DEN`
(your choice per week) is collected from every wallet (`floor(wallet*T[r]/TAX_DEN)`),
pooled, and immediately **refilled** back out that same week proportional to weights
`W[r][i] >= 0` you also choose (`floor(pool*W[r][i]/sum(W[r]))` each, remainder to the
highest-weight, lowest-index nurses; if all weights are 0, the pool splits evenly).
You design the market's **initial scrip endowment** and its **ongoing tax/refill
policy**; you do not choose who wins each week -- that emerges from the replay.

## Input (stdin)
```
N R
ALPHA_NUM ALPHA_DEN S TAX_DEN
v[1][1] v[1][2] ... v[1][N]
...
v[R][1] v[R][2] ... v[R][N]
```
`S` is the total scrip supply in the closed economy.

## Output (stdout)
```
E[1] E[2] ... E[N]
T[1] T[2] ... T[R]
W[1][1] ... W[1][N]
...
W[R][1] ... W[R][N]
```
`E` = initial endowments, `T` = per-week tax rate, `W[r]` = per-week refill weights.

## Feasibility
- All `N + R + N*R` numbers must be present and parse as integers (else `Ratio: 0.0`).
- `E[i] >= 0` for all `i`, and `sum(E) == S` **exactly** -- you must allocate the whole
  budget, no more, no less.
- `0 <= T[r] <= TAX_DEN` for every week.
- `0 <= W[r][i] <= 1000000` for every week and nurse.
Any violation scores `Ratio: 0.0`.

## Objective
Replay all `R` weeks with your `(E, T, W)` and maximize `F`, the total disutility
relief `sum_r v[r][winner(r)]` delivered to the winning nurse each week.

## Scoring
Let `B` be the checker's own baseline: equal-split endowment (`S` divided as evenly as
integers allow), zero tax, zero refill, replayed the same way. With `F` your replayed
total relief:
```
sc = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Matching the baseline scores `0.1`; delivering `10x` the baseline relief caps at `1.0`.

## Constraints
- `4 <= N <= 100`, `5 <= R <= 60`.
- `0 <= v[r][i] <= 1000`. `ALPHA_NUM=1, ALPHA_DEN=2, TAX_DEN=1000` in every test.
- Time limit 5s, memory 512m.

## Example (illustrative -- a different shape than the actual hidden test cases)
`N=3, R=2, ALPHA_NUM=1, ALPHA_DEN=2, S=30, TAX_DEN=1000`, and both weeks have
`v = [40, 10, 10]`. Baseline `E0=[10,10,10]`, no tax: week 1, bids `[10,5,5]`, nurse 0
wins (relief 40), pays 10 split `[5,5]` to the others -> wallets `[0,15,15]`. Week 2,
bids `[0,5,5]`, nurse 1 wins by index tie-break (relief 10). `B = 50`.
A submitted policy `E=[10,10,10]`, `T=[500,0]`, `W[1]=[1,0,0]` (week-2 row unused since
its tax is 0): week 1 is identical (relief 40, wallets `[0,15,15]`), then a 50% tax
collects `[0,7,7]` (pool 14) and refills it entirely to nurse 0, giving wallets
`[14,8,8]`. Week 2: bids `[14,5,5]`, nurse 0 now affords to win again (relief 40).
`F = 80`, so `Ratio = min(1000, 100*80/50)/1000 = 0.16` -- recirculating the tax back
to the nurse who actually needs it let them keep bidding instead of freezing out.
