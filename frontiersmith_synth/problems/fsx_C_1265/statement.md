# Stacking Chips Without Cooking the Middle One: TSV Thermal Via Placement

## Problem

A chip stack of `M` dies sits on a heat sink: die `1` rests directly on the sink, die `2` on
top of die `1`, ..., die `M` at the very top of the stack. The stack is discretized into `N`
shared columns (an XY grid position that lines up across every die). Die `m` dissipates power
`p[m][c]` at column `c`.

A **thermal via (TSV)** can be drilled straight down through column `c`, replacing the
per-layer thermal resistance in that column, for every layer it passes through, from a
baseline `R0` down to `Rv < R0`. Because a via runs the full height of the stack, placing one
is an all-or-nothing decision per column. Each via at column `c` costs `a[c]` units of a shared
**via area budget** `A`: the total cost of all columns you via must not exceed `A` (vias
everywhere is never affordable).

Heat generated on die `m` must cross every layer boundary between die `m` and the sink -- that
is `m` boundaries of resistance in column `c`. Every die that shares column `c` shares that
exact same vertical path down to the sink, so their heat contributions stack. The realized
peak temperature at column `c` is
```
T[c] = R(c) * W[c],   where   W[c] = sum_{m=1}^{M} m * p[m][c]
```
and `R(c)` is `Rv` if a via is placed at `c`, else `R0`. `W[c]` is the **depth-weighted,
stacked** hotspot profile of the whole stack -- it is generally NOT maximized at the column
where any single die's own power map peaks. Your goal is to choose which columns get a via,
within the area budget, to **MINIMIZE the peak stack temperature** `max_c T[c]`.

## Input (stdin)
```
M N A
R0 Rv
a_1 a_2 ... a_N
p_1_1 p_1_2 ... p_1_N        (die 1, nearest the sink)
...
p_M_1 p_M_2 ... p_M_N        (die M, top of the stack)
```

## Output (stdout)
One line of `N` tokens `x_1 x_2 ... x_N`, each `0` or `1`: `x_c=1` means a via is placed at
column `c`.

## Feasibility
Rejected (score 0) if: the output does not have exactly `N` tokens; any token is not an exact
integer `0` or `1` (garbage, non-finite, floats, extra/missing tokens all reject); or the total
area cost `sum(a[c] for x[c]=1)` exceeds `A`.

## Scoring
Let `F = max_c ( R(c) * W[c] )` for your placement. Let `B = R0 * max_c W[c]`, the peak
temperature of the checker's own trivial reference (no vias placed at all -- always feasible,
since it spends none of the budget). Score:
```
ratio = min(1.0, 0.1 * B / max(1e-9, F))
```
Lower `F` (relative to `B`) scores higher; placing no vias scores about `0.1`.

## Example (worked, not to scale with the real tests)
`M=2` dies, `N=3` columns, `R0=100`, `Rv=25`. Power: die 1 (near sink) `p_1 = [0, 40, 0]`, die 2
(top) `p_2 = [50, 0, 0]`. Depth-weighted profile: `W = [1*0+2*50, 1*40+2*0, 1*0+2*0] = [100, 40,
0]`. Budget `A=5`, costs `a = [6, 4, 1]` (column 0's via is too expensive to afford).
- No vias: `F = max(100*100, 100*40, 100*0) = 10000 = B`, ratio `= 0.1`.
- Via column 1 only (looking only at column 0's raw power, which is highest for die 2 alone,
  would tempt you toward column 0 -- but it costs `6 > A=5` and is unaffordable): cost `4<=5`,
  `F = max(100*100, 25*40, 100*0) = 10000` -- unchanged, because column 0 (`W=100`) is still
  the worst column and it was never touched.
- Via column 0 instead (the true highest-`W` column, `W=100`, even though die 2's OWN reading
  there is only `50` -- die 1's `p=0` there but the depth weight on die 2 dominates): but its
  cost `6 > A=5`, so it cannot be afforded either in this tiny example -- the achievable optimum
  here is column 1 alone, `F=10000` unchanged (`ratio=0.1`); enlarging `A` to `6` would let you
  via column 0 and drop `F` to `max(25*100,100*40,100*0)=4000`, ratio `=0.25`. The lesson is
  reading `W[c]`, not any single die's map, against the actual budget.

## Constraints
`1 <= M <= 8`, `2 <= N <= 50`, `1 <= a[c] <= 20`, `0 <= A <= sum(a)`,
`1 <= Rv < R0 <= 200`, `0 <= p[m][c] <= 150`. Time limit 5s, memory 512MB.
