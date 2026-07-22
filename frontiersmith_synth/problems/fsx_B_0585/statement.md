# Hedged Cargo Manifest at the 20th Percentile

## Problem
You load a single cargo hold from a catalogue of `N` items under two capacity
limits (weight `W` and volume `V`). Each item pays a different amount in each of
`S` future market **scenarios**. Your manifest is not judged by its average
payoff — it is judged by its **20th-percentile scenario total**. A manifest that
is spectacular on average but collapses in a handful of bad scenarios scores
badly; you must build a bundle that holds up even in its weaker scenarios.

The catalogue is adversarial by construction. Items arrive in **anti-correlated
pairs** tied to a hidden common market factor: within a pair, one item pays well
when the market is strong and poorly when it is weak, and its partner is the
mirror image. High-mean items are strong *together* in the same scenarios and
weak *together* in the rest, so a bundle packed with them concentrates all its
risk in one place — exactly where the 20th percentile looks.

## Input (stdin)
```
N S W V k
w_1 v_1  a_{1,1} a_{1,2} ... a_{1,S}
w_2 v_2  a_{2,1} a_{2,2} ... a_{2,S}
...
w_N v_N  a_{N,1} a_{N,2} ... a_{N,S}
```
- `N` items, `S` scenarios, integer capacities `W` (weight) and `V` (volume).
- `k` is the quantile rank used to score you (`k = ceil(0.20 * S)`; here the
  `k`-th smallest scenario total, 1-indexed).
- Item `i` has integer weight `w_i > 0`, volume `v_i > 0`, and integer payoffs
  `a_{i,s} >= 1` for each scenario `s`. All the per-item, per-scenario numbers
  (including which items are anti-correlated and how hard each crashes) live in
  the input — you must read them, not guess them from this statement.

## Output (stdout)
The indices (0-based) of the items you load, whitespace-separated, in any order.
Duplicates are illegal. An empty line is a legal (empty) manifest.

## Feasibility
Let `M` be your chosen set. It is feasible iff every index is in `[0, N)`, no
index repeats, `sum_{i in M} w_i <= W`, and `sum_{i in M} v_i <= V`. Any breach
(out-of-range index, duplicate, non-integer/`nan`/`inf` token, or either
capacity exceeded) scores `0`.

## Objective (maximize)
For a feasible manifest `M`, its scenario total in scenario `s` is
`T_s = sum_{i in M} a_{i,s}`. Sort the `S` totals ascending and take the `k`-th
smallest:
```
Q(M) = sort(T_1, ..., T_S)[k]      # 20th-percentile scenario total
```
Maximize `Q(M)`.

## Scoring
Your raw objective `Q(M)` is normalized against a fixed internal baseline `B`
that the grader builds itself (a value-blind, lightest-items-first fill to about
one third of capacity):
```
Ratio = min(1.0, 0.1 * Q(M) / B)
```
A manifest matching the baseline scores about `0.1`; ten times better caps at
`1.0`. The `k`-th-smallest is a genuine quantile: pushing up your best scenarios
does nothing once they are above the tail, so the score only responds to lifting
the totals of your **weakest** scenarios.

## Constraints
- `70 <= N <= 120`, `S = 40`, `k = 8`.
- Payoffs, weights, volumes are positive integers; instance file `<= 5 MB`.
- Time limit 5s, memory 512 MB. Deterministic: same manifest ⇒ same score.

## Example (worked score, illustrative — NOT the graded instance)
Say `S = 5`, `k = 2`. You load a set whose scenario totals are
`T = [60, 12, 55, 14, 58]`. Sorted: `[12, 14, 55, 58, 60]`; the 2nd smallest is
`Q = 14`. If instead you had hedged into a flatter set with totals
`[40, 33, 41, 35, 39]`, sorted `[33, 35, 39, 40, 41]`, then `Q = 35` — a far
better 20th-percentile even though its best scenario (41) is much lower than the
first set's best (60). The tail, not the peak, is what is graded.
