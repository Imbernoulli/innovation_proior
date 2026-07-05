# Resonance-Free Traffic Signal Deployments

## Title
Deploy the largest resonance-free family of city-wide traffic-signal schedules.

## Problem
A city has `n` signalized intersections. Each intersection can run exactly one of
three fixed phase programs, encoded `0`, `1`, or `2` (e.g. `0` = North–South green,
`1` = East–West green, `2` = all-red pedestrian scramble). A *city-wide schedule* is
therefore a vector in `{0,1,2}^n` — one phase program per intersection.

Traffic engineers have discovered a destructive interference pattern: three deployed
schedules `x`, `y`, `z` cause a **grid-lock resonance** when, at *every* intersection
`i`, the three chosen phase programs `x_i, y_i, z_i` are **either all equal or all
distinct**. (Equivalently: `x_i + y_i + z_i ≡ 0 (mod 3)` at every intersection.)

You must select a set `S` of distinct city-wide schedules to deploy such that **no
three distinct schedules in `S` form a grid-lock resonance**. Deploy as many schedules
as you can.

This is exactly the *cap set* problem over the affine space AG(n,3): a resonance is a
line, and a resonance-free family is a cap. Maximum cap sizes are known only for small
`n` and are the subject of active research, so there is no easy optimum — many
orderings and product constructions give different families.

## Input (stdin)
One line with two integers:
```
n q
```
`n` is the number of intersections (`4 ≤ n ≤ 7`) and `q` is the number of phase
programs per intersection (always `3`).

## Output (stdout)
Print the deployed set `S`, one schedule per line. Each schedule is a string of
exactly `n` characters, each character in `{0,1,2}`, giving the phase program at
intersections `1..n`. Schedules must be pairwise distinct. Blank lines are ignored.
You may print them in any order.

## Feasibility
The output is feasible iff:
- every printed token has length exactly `n` and uses only characters `0`,`1`,`2`;
- all printed schedules are distinct;
- no three distinct schedules `x,y,z ∈ S` satisfy `x_i+y_i+z_i ≡ 0 (mod 3)` for all
  `i` (no grid-lock resonance).

Any violation scores `0`.

## Objective
Maximize `|S|`, the number of deployed schedules.

## Scoring
Let `F = |S|` for a feasible output. The checker uses an internal trivial baseline
`B = 2^(n-2)` (a resonance-free family that fixes the last two intersections and uses
only phases `{0,1}` on the rest). The score is
```
sc    = min(1000, 100 * F / B)
Ratio = sc / 1000
```
so reproducing the baseline scores `≈ 0.1`, and a family ten times larger caps at
`1.0`. The reported grade is the mean `Ratio` over all test cases.

## Constraints
- `4 ≤ n ≤ 7`, `q = 3`.
- Deterministic scoring only; the checker uses exact integer arithmetic over `F_3`.

## Example
Suppose `n = 4`, `q = 3`. Deploying the four schedules
```
0000
0100
1000
1100
```
uses only phases `{0,1}` on the first two intersections and fixes the last two, so no
three of them can be all-distinct at any intersection — resonance-free. Here `F = 4`,
`B = 2^(4-2) = 4`, giving `Ratio = 0.100`. A serious deployment (e.g. a product of
maximal 9-schedule caps on blocks of three intersections) reaches `F = 18` for
`n = 4`, i.e. `Ratio = 0.45`.
