# River Alchemist: The Settling Cascade

## Problem
An alchemist draws river water carrying **seven powders** (classes `1..7`), each with
its own residence-time threshold: a grain of class `c` settles out of the flow the
first time it spends at least `thr_c` time units inside a basin. Water enters the
cascade at a fixed inflow rate `Q` and passes through a sequence of open basins you
design, one after another; whatever does not settle in a basin flows on to the next.

You choose a **sequence of basins**. Basin `i` has an integer **depth** `d_i` and
**length** `l_i`; its residence time is proportional to its volume, and a class
`c` still present in the flow settles there iff `d_i * l_i >= Q * thr_c`. But depth
also controls the flow **velocity** through the basin (`~ Q / d_i`): if that
velocity exceeds a class's scour speed `scour_c`, anything of that class that would
otherwise settle is instead re-suspended and swept on downstream — a basin that is
too shallow for its length loses classes it should have caught. For a fixed volume
`d_i * l_i`, you can always shift the split toward more depth (safer, less scour)
or more length (cheaper velocity is irrelevant if you don't need it) — the trade is
free in volume but constrained by per-basin caps.

Each basin also drains its captured sediment into a **bin** `b_i` (bin `0` = discard,
bins `1..7` = one bin per class). A basin whose captured sediment is a mix of
classes contaminates whichever bin it drains to.

## Input (stdin)
```
Q VolumeBudget M Dmax Lmax
PenNum PenDen
thr_1 mass_1 value_1 scour_1
...
thr_7 mass_7 value_7 scour_7
```
All values are positive integers. `thr_c` are seven **distinct** thresholds, listed
in class-id order `c = 1..7` (NOT sorted by size — you must determine the order
yourself). `mass_c` is the inflow rate of class `c`'s powder. `PenNum/PenDen` is the
contamination-penalty multiplier `P` applied per unit of wrongly-routed mass.

## Output (stdout)
```
M'
d_1 l_1 b_1
...
d_M' l_M' b_M'
```
`M'` basins (`0 <= M' <= M`), each with `1<=d_i<=Dmax`, `1<=l_i<=Lmax`,
`0<=b_i<=7`, listed in the order the flow passes through them.

## Feasibility
- `0 <= M' <= M`, all fields well-formed finite integers in range.
- `sum(d_i * l_i) <= VolumeBudget`.
Any violation scores **0**.

## Simulation
Track each class's remaining mass, starting at `mass_c`. Process basins in the
given order. At basin `i`, every class `c` still fully present that satisfies BOTH
`d_i*l_i >= Q*thr_c` (residence time cleared) AND `Q <= scour_c*d_i` (no
resuspension) has its **entire** remaining mass captured there and routed to bin
`b_i` (bin `0` discards it — no value, no penalty).

## Objective / Scoring
For each bin `t = 1..7`, let `correct_t` = mass of class `t` routed to bin `t`, and
`wrong_t` = mass of any *other* class routed to bin `t`. Bin `t` contributes
`max(0, value_t*correct_t - P*value_t*wrong_t)` (a contaminated bin can be worth
zero, never negative). The objective `F` is the sum over all seven bins. The
checker compares `F` against its own reference `B`: a single basin that resolves
only the smallest threshold and ignores the other six classes, and reports
`Ratio = min(1, F / (12 * B))`. Maximize the ratio.

## Constraints
`3 <= Q <= 35`, `1 <= Dmax <= 40`, `1 <= Lmax <= 6000`, `M <= 10`, thresholds and
masses fit in 32-bit integers. Time limit 5s.

## Example (illustrative shape only — not the hidden instance)
Suppose two classes only, thresholds `5` and `9`, masses `10` and `10`, values `4`
and `4`, `P=1`. A basin with `d*l=5..8` captures only class 1: bin 1 gets
`4*10 - 0 = 40`. A basin with `d*l>=9` in a *single* basin captures both at once
into one bin: that bin gets `4*10 - 1*4*10 = 0` (wiped out by contamination). Two
basins — one sized `5<=d*l<8` draining to bin 1, a second sized `d*l>=9` draining
to bin 2 — gets `40 + 40 = 80`: sequencing the cut points, not just sizing one big
basin, is what separates the powders.
