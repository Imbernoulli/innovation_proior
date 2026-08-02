# Liquidity Ladder: Smoothing the Rollover Cliff

## Problem
A treasury desk must fund a schedule of legacy liabilities `L[1..T]` (integers, one per
future period) with total size `F = sum(L)`. It restructures this into a **maturity
ladder**: a face value `p[t] >= 0` due at each period `t`, with `sum(p) = F`. Placing a
unit of face value at maturity `t` costs `y[t]` (a per-maturity yield, given in the
input; not necessarily monotonic in `t`).

Whatever matures at period `t` must be **rolled over** (refinanced) in the market at
that date. The market's normal rollover capacity at date `t` is `Base[t]`. During a
stress window, capacity at the affected dates shrinks to `floor(Base[t]*hn/hd)` for a
haircut fraction `hn/hd` given in the input. The desk does **not** know in advance which
dates will be stressed: the checker evaluates the submitted ladder against several
internally-generated stress scenarios, each marking a contiguous run of `window_len`
dates as stressed. These scenarios are deterministic but not shown in the input, and are
deliberately weighted to include the dates around the two largest entries of `L` (any
sharp legacy peak is always tested), plus additional sweep scenarios.

Placing maturities to exactly match the legacy schedule (`p = L`) trivially satisfies
prefunding, but if `L` has a sharp peak, that peak's entire face value must roll over on
one date -- exactly the date most likely to be tested under stress.

## Input (stdin)
```
T F
L[1] L[2] ... L[T]
y[1] y[2] ... y[T]
Base[1] Base[2] ... Base[T]
hn hd
S window_len
```
`S` is how many hidden stress scenarios the checker will run (their exact windows are
not given). `hn/hd` is the stress haircut fraction applied to capacity inside a window.

## Output (stdout)
`T` non-negative integers `p[1] ... p[T]` (whitespace-separated), the face value
maturing at each period.

## Feasibility (hard; any violation scores `Ratio: 0.0`)
- exactly `T` integer tokens, each `>= 0`;
- `sum(p) == F` exactly.

## Objective (minimize)
```
cost(p) = yield_cost + 4.0 * prefund_gap + 18.0 * avg_stress_gap
```
- `yield_cost = sum(p[t]*y[t]) / 100`.
- `prefund_gap = sum over t of max(0, cumL[t] - cumP[t])`, where `cumL`/`cumP` are
  running totals of `L`/`p` up to `t`. This is a **soft** penalty (not a hard
  constraint): falling behind the legacy schedule's own cumulative funding pace is
  costly but not automatically rejected. Note pulling a unit of face value to an
  *earlier* maturity than its legacy date can never increase this gap.
- `avg_stress_gap` = average, over the checker's hidden stress scenarios, of
  `sum over t of max(0, p[t] - capacity_at(t))`, where `capacity_at(t)` is `Base[t]`
  outside the scenario's stress window and the haircut-reduced value inside it.

## Scoring
Let `B` be the cost of the checker's own reference ladder: dump the entire `F` into the
first maturity (`p[1]=F`, always feasible). Then
```
sc = min(1000.0, 100.0 * B / max(1e-9, cost(p)))
Ratio = sc / 1000.0
```
Reproducing `B` scores `Ratio = 0.1`; a ladder with `10x` lower cost caps at `1.0`.

## Constraints
- `11 <= T <= 20`, `8 <= L[t] <= a few hundred` under a planted peak, `300 <= y[t] <=
  700`, `Base[t] >= 6`, `hn/hd in {1/3, 2/5, 1/2}`, `2 <= window_len <= T`.
- Time limit 5s, memory 512m.

## Example
Suppose `T=4`, `L=[10,10,40,10]`, `F=70`, all `y[t]=500`, all `Base[t]=20`, haircut
`1/2`, and a stress scenario covers period 3. Matching (`p=L`) leaves `40` due at period
3 against a haircut capacity of `10`, an overflow of `30` in that scenario. Shifting `20`
units of period 3's face value back to periods 1-2 (raising `p` there to `20` each, at
their normal-condition capacity) cuts that overflow to `10` at a small extra yield cost.
The prefund gap stays at zero throughout, since shifting a unit to an *earlier* maturity
than its legacy date can never make `cumP` fall behind `cumL`. (This is an illustrative
shape only -- the actual checker's baseline `B` and hidden scenarios follow the rules
above.)
