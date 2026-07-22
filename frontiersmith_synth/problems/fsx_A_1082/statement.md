# Honest Frontage: Minimum-Gate VCG Payments for a Row of Land Plots

## Problem
A municipality auctions `n` land plots standing in a fixed row along a coastal
access road, numbered `1..n`. Plot `i` belongs to bidder `i`, who declares a
value `v_i`. Adjacent plots `i` and `i+1` share a single access easement, so
**at most one of every adjacent pair may be awarded**; non-adjacent plots
never conflict. The welfare-maximizing allocation is the maximum-weight
subset of `1..n` with no two adjacent indices both chosen.

To keep bidders honest, the auctioneer pays each winner not their bid but a
**VCG (Clarke-pivot) payment**: the externality they impose on everyone else.
For bidder `i`, let `OPT` be the optimal welfare over all `n` bidders and
`OPT(-i)` be the optimal welfare of the economy with bidder `i` removed
entirely (their plot given back, no easement constraint on it). Then
```
pay_i = OPT(-i) - (OPT - v_i * x_i)
```
where `x_i = 1` if bidder `i` is used by the (fixed, tie-broken) optimal
allocation and `0` otherwise; ties are broken by treating `i` as **not**
selected whenever excluding it still reaches `OPT` (`x_i = 1` iff
`OPT(-i) < OPT`, i.e. `i` is truly essential to the optimum).

Computing `OPT(-i)` fresh for all `n` bidders means `n` separate welfare
optimizations. Your job: emit an explicit **arithmetic circuit** that outputs
all `n` payments, using as few gates as possible.

The instance actually bundles `T` independently-random bid vectors that all
share the same `n` (same row of plots, same easement structure). You submit
**one fixed circuit**; the judge re-evaluates that same circuit once per
trial, feeding each trial's bid values onto input wires `0..n-1`, and
requires every trial's outputs to match exactly. A circuit only has to have
the right *shape* — it never needs to know which trial it is running on. (An
astute reader will notice this rules out "hardcoding" a numeric answer for
one bid vector: it would not survive re-evaluation on the others.)

## Input (stdin)
```
n T
v_1_1 v_1_2 ... v_1_n
v_2_1 v_2_2 ... v_2_n
...
v_T_1 v_T_2 ... v_T_n
```
Line `t` (`1<=t<=T`) is trial `t`'s bid vector. `1 <= n <= 2000`,
`1 <= T <= 20`, `1 <= v_t_i <= 1000000`, all integers.

## Output (stdout)
```
G
```
then `G` gate lines, then one `OUT` line. Wires are numbered: inputs are
wires `0..n-1` (wire `i-1` holds bidder `i`'s value for whichever trial is
being evaluated); the `g`-th gate line (0-indexed) creates wire `n+g`. Every
gate may reference only **strictly earlier** wires. A gate line is one of:
```
CONST c        ADD a b        SUB a b        MUL a b        MAX a b        GT a b
```
`CONST c` introduces literal integer `c` (no operands). `GT a b` evaluates to
`1` if the value on wire `a` is strictly greater than the value on wire `b`,
else `0`. All values are exact integers (no floating point). Finally:
```
OUT w_1 w_2 ... w_n
```
names the `n` wires holding `pay_1 .. pay_n`, each `0 <= w_i < n+G`. The
circuit is the same for every trial; only what is fed onto wires `0..n-1`
changes. `0 <= G <= 200000`.

## Feasibility
For EVERY trial `t`, evaluate the circuit with exact integer arithmetic
(wires `0..n-1` = trial `t`'s bids). The output is feasible iff every `OUT`
wire equals trial `t`'s reference `pay_i` **exactly**, for every `i` and
every `t`. Any parse error, out-of-range/forward wire reference,
out-of-range literal, non-finite/non-integer token, or payment mismatch on
any trial scores `0`.

## Objective
Minimize `G`, the number of gate lines.

## Scoring
Let `B = 4*n^2 + 1` — the exact gate count of the straightforward
construction that reruns the standard `O(n)` welfare recurrence completely
fresh for each of the `n` leave-one-out economies (no reuse across
economies). With your gate count `G`:
```
Ratio = min(1, 0.1 * B / G)
```
Reproducing that per-economy-recompute cost scores about `0.1`. The `n`
leave-one-out economies overlap heavily — most of what changes between
"welfare without bidder `i`" and "welfare without bidder `i+1`" is nothing at
all — so a circuit that shares one prefix and one suffix computation across
every economy needs only `O(n)` gates, not `O(n^2)`. The exact best gate
count is not known; there is headroom above any reference construction.

## Constraints
- `1 <= n <= 2000`; `1 <= T <= 20`; `1 <= v_t_i <= 1000000`.
- `0 <= G <= 200000`; every intermediate/literal value stays within
  `[-10^12, 10^12]`.
- Deterministic exact-integer scoring; no timing.

## Example
`n = 3`, one trial `v = (5, 9, 3)` (a real instance bundles several trials;
one is shown here for clarity). The unique welfare optimum is `{2}` (value
`9`, beating `{1,3}` at `8`). Removing bidder 1: still `9` (via `{2}`) →
`pay_1 = 0`. Removing bidder 2: only `8` reachable (via `{1,3}`) → bidder 2
is essential, `pay_2 = 8 - 9 + 9 = 8`. Removing bidder 3: still `9` → `pay_3
= 0`. A circuit computing these three payments correctly **on every trial**
with `G = 24` gates (sharing one forward + one backward pass) scores
`min(1, 0.1 * 37 / 24) ≈ 0.154` (here `B = 4*3^2+1 = 37`). Recomputing each
economy independently from scratch would need far more gates and score
close to `0.1`.
