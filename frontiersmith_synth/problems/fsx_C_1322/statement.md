# Batch Cooling Crystallizer: Cool Schedule and Seed Choice Under a Steepening Solubility Curve

## Problem
A batch crystallizer starts exactly saturated at temperature `T_start` and must
be cooled, over `N` discrete steps, down to no lower than `T_min`. You choose
(a) one seed option from a given library to charge into the vessel at step 0,
and (b) the temperature after each of the `N` steps (a cooling schedule that
must never increase). A deterministic simulation then grows and nucleates
crystals from your choices and reports the mean final crystal size, subject to
crystallizing at least a required fraction of the dissolved solute.

Crystal population is tracked via its first four size moments (count, total
length, total area, total volume) under the standard McCabe "Delta-L" law
(every existing crystal grows by the same length increment each instant,
proportional to the current supersaturation `S` = dissolved concentration
minus the equilibrium solubility at the current temperature) plus a
supersaturation-driven nucleation term `B = kb * S^b` (new crystals born per
unit time, `b > 3`: strongly convex in `S`). Dissolved concentration decreases
as mass moves into the crystal phase. All of this is computed by the checker
from your submitted schedule; nothing about the simulation is hidden -- every
constant it uses is given in the input below.

## Input (stdin)
```
N T_start T_min
kb b kg g r0 kv_rho
M
T_1 Ceq_1
...
T_M Ceq_M
required_yield
K
count_1 radius_1
...
count_K radius_K
```
`kb,b` are the nucleation-rate constant and exponent; `kg,g` the growth-rate
constant and exponent (`growth_rate = kg * S^g`); `r0` the length of a
newly-born nucleus; `kv_rho` converts total crystal volume-moment into mass.
The `M` `(T, Ceq)` pairs give the equilibrium solubility curve by linear
interpolation, `T` strictly increasing from `T_min` to `T_start`.
`required_yield` is the minimum fraction of the initially-dissolved solute
mass that must have crystallized by step `N`. The `K` seed options each give a
particle count and radius to charge at step 0 (any one option, not a mix).

## Output (stdout)
One line: `seed_idx` (an integer `1..K`) followed by exactly `N` more
numbers -- the temperature after each step, `T_1 .. T_N`.

## Feasibility
`1 <= seed_idx <= K`. The temperature sequence must be non-increasing
(`T_start >= T_1 >= T_2 >= ... >= T_N`) and stay within `[T_min, T_start]`.
Simulating the chosen seed option through your schedule must reach at least
`required_yield`. Any violation scores 0.

## Objective (what the score rewards)
The mean crystal size at the end of the batch: total length-moment divided by
total count. Larger is better -- one big crystal beats many small ones even
if both configurations crystallize the same total mass.

## Scoring
The checker also simulates its own reference schedule (jump straight to
`T_min` on step 1 and hold it there, charged with seed option 1) and reports
`Ratio = min(1000, 100 * your_mean_size / reference_mean_size) / 1000`.
Matching the reference gives ~0.1; the reference is deliberately bad (maximum
instantaneous supersaturation detonates the nucleation term), so there is
room well above it.

## Constraints
`20 <= N <= 60`, `4 <= K <= 6`, `40 <= T_min < T_start = 100`,
`b` in `[3.0, 3.4]`. The `(T, Ceq)` curve is always steeper in its bottom
slice (near `T_min`) than its top slice (near `T_start`) -- check the given
breakpoints rather than assuming a shape. Time limit 5s.

## Example (worked, small illustrative numbers)
`N=2`, `T_start=10, T_min=0`. Curve: `(0,0),(8,2),(10,10)` (flat 8..10, steep
0..8). One seed option: `count=1, radius=0.1`. `required_yield=0.1`.
Schedule A (linear): `T_1=5, T_2=0`. Step 1 moves 10->5: `Ceq` drops from 10
to ~5.5 (mild, in the flat zone) -- little supersaturation yet. Step 2 moves
5->0, crossing the steep zone in one step with little time left to grow
anything nucleated there. Schedule B (front-loaded): `T_1=1, T_2=0`. Step 1
already reaches the steep zone (`Ceq` drops from 10 to ~2), generating
supersaturation with a full remaining step for growth to consume it before
any further cooling is needed. B ends with fewer, larger crystals than A for
the same total mass crystallized.
