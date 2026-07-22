# The Swordsmith's Tempering Chart

## Problem

A freshly forged billet is a line of `L` grains. Grain `i` carries an initial
defect count `d_i >= 0` and a hidden **nucleation threshold** `theta_i`. You
plan a tempering schedule: `n` heating steps with integer temperature levels
`T_1, ..., T_n` (each `0 <= T_j <= Tmax`), applied to the *whole* billet at
once — you cannot address individual grains.

Each step costs `C0 + T_j` fuel (a fixed per-heat overhead `C0` plus fuel
proportional to the level). The total cost of all `n` steps must not exceed
budget `B`. Use as few as `n = 0` steps (quench as-is) up to `n_max`.

**Kinetics (every grain, every step, in order):** heat gives atoms mobility,
but mobility only kicks in above a fixed activation floor of `2` (a bare
simmer moves nothing), so a grain heals by up to `T_j - 2` defects per step.
Heat also nucleates brand-new defects once the temperature exceeds that
grain's own threshold. If grain `i` holds `d` defects and the step
temperature is `T`:
```
mobility = max(0, T - 2)
heal     = min(d, mobility)
nucleate = max(0, T - theta_i)
d_new    = d - heal + nucleate
```
Both effects are monotone non-decreasing in `T`: hotter always heals at
least as fast, and hotter always nucleates at least as much — you cannot turn
one off without weakening the other. Holding at or below the activation floor
heals nothing at all, so cheap low-heat steps are not a safe fallback, they
are pure fuel waste. The only lever is *which* integer temperature (for how
long, in how many separately-costed steps) you choose, calibrated to the
actual `theta_i` values given explicitly in the input — nothing is hidden
from your program, but nothing about the right temperature is stated here.

Grains differ: a low-threshold grain nucleates violently even at modest heat,
while a high-threshold grain tolerates much hotter holds. A single global
schedule touches every grain at once, so a level safe for most of the billet
may be well above threshold for a fragile minority — and running that level
for many steps compounds the damage every single step, since nucleation is
not capped by how many defects the grain currently holds.

After your `n` steps (or immediately if `n = 0`), final defect count
`F = sum_i d_i`. **Lower `F` is better.**

## Input (stdin)
```
L Tmax C0 n_max B
d_1 d_2 ... d_L
theta_1 theta_2 ... theta_L
```
Non-negative integers; `1 <= theta_i <= Tmax - 1`.

## Output (stdout)
```
n
T_1 T_2 ... T_n      (omit this line entirely if n = 0)
```
`0 <= n <= n_max`, each `0 <= T_j <= Tmax`, `sum_{j=1..n} (C0 + T_j) <= B`.

## Feasibility
Score `0` on: `n` outside `[0, n_max]`; wrong count of temperature tokens; a
token that is not a base-10 integer (includes `nan`/`inf`/floats); any `T_j`
outside `[0, Tmax]`; total fuel cost exceeding `B`; trailing garbage.

## Scoring
The checker replays the kinetics deterministically from your schedule and
computes `F`. Let `D = sum_i d_i` (the `n=0` result). The raw ratio is `D / F`
(capped), rewarding schedules ending with fewer defects than you started with,
and increasingly little marginal reward as `F` approaches 0. Doing nothing
always scores a small fixed baseline.

## Constraints
`1 <= L <= 2000`, `Tmax = 12`, `1 <= n_max <= 40`, `1 <= C0 <= 6`, `B >= C0`.

## Example (worked, illustrative shape only — not one of the 10 graded cases)
`L=3 Tmax=12 C0=2 n_max=5 B=30`, `d=[10,10,10]`, `theta=[8,8,5]`. Grain 3 is
fragile. **Safe plateau:** `T=4` for 5 steps (cost `5*6=30=B`) gives mobility
`2`, never exceeds any threshold: every grain heals `2`/step, zero
nucleation, `10->8->6->4->2->0` for all three — final `F=0`, whole budget
used. **Hot plateau:** `T=11` for 2 steps (cost `2*13=26<=B`) gives mobility
`9`, clearing grains 1,2 in two steps (`10-9=1`, then `1-1=0`), but grain 3's
threshold is `5`, so each such step also nucleates `11-5=6` new defects there:
step 1 gives `10-9+6=7`, step 2 gives `7-7+6=6` — final `d=[0,0,6]`, `F=6`,
strictly worse despite healing the majority faster per step. The plateau that
respected the minority grain's threshold won outright; a schedule calibrated
only to the robust majority would have missed it.
