# Dragon-Kiln Firing Plan: Exact Glaze Rendezvous

## Problem
A dragon-kiln is a long, sealed tunnel kiln. Its cross-section along the firing
direction is modeled as `N` cells in a row, numbered `0..N-1`. Some cells hold
**burner mouths**; other cells hold **target zones** that must all reach the
glaze-maturation temperature `H` at **exactly step `T`** — not before, not
after, just present in the reading taken at step `T`.

Every step, heat in the sealed kiln redistributes by a fixed rule (insulated
ends — heat never leaves the kiln, only moves sideways):
```
h'[i] = h[i]/2 + h[i-1]/4 + h[i+1]/4     (out-of-range neighbour == h[i] itself)
```
starting from `h = 0` everywhere at step `0`.

You control a set of **firings**. A firing is `(burner b, start s, duration d,
power p)`: at every step `t` with `s <= t < s+d`, `p` units of heat are added
to burner `b`'s cell **before** that step's redistribution is applied. Firings
must complete inside the window: `0 <= s`, `1 <= d`, `s + d <= T`. A firing's
**fuel cost** is `p*d + F0` (an integer setup surcharge `F0` per firing, on top
of the power-time product) — fuel is counted exactly, never estimated.

Multiple firings' heat simply adds (superposition): a cell's final
temperature is the exact sum of every firing's contribution, so overlapping
firings from different burners jointly credit the same target.

## Input (stdin)
```
N T H Pmax F0 numBurners numTargets
b_1 b_2 ... b_numBurners        (burner cell positions)
x_1 x_2 ... x_numTargets        (target cell positions)
```
All values are integers. `1 <= N <= 40`, `1 <= T <= 40`, `1 <= Pmax <= 20`,
`1 <= F0 <= 20`, `1 <= numBurners, numTargets <= 8`.

## Output (stdout)
```
K
b_1 s_1 d_1 p_1
...
b_K s_K d_K p_K
```
`K` is the number of firings (`0 <= K <= 300`). Each `b_i` is a **0-indexed**
burner index (into the input burner list, not a cell position). All fields
are integers.

## Feasibility
Simulate the redistribution rule exactly (integer/rational — no rounding) for
`T` steps with all `K` firings applied. The output is feasible iff:
- every `b_i` is in `[0, numBurners)`;
- `0 <= s_i <= T-1`, `1 <= d_i <= T - s_i`, `1 <= p_i <= Pmax`;
- the final temperature of **every** target cell is `>= H`.

Any parse error, out-of-range field, non-finite/non-integer token, or a
target below `H` at step `T` scores `0`.

## Objective
Minimize total fuel `F = sum(p_i*d_i + F0)`.

## Scoring
Let `B` be the fuel of the checker's own baseline: fire, at full power `Pmax`
for the entire window (`s=0, d=T`), the nearest burner to each target,
deduplicating burners shared by several targets. With your feasible fuel `F`:
```
Ratio = min(1, 0.1 * B / F)
```
Using exactly the baseline construction scores `0.1`. Halving fuel doubles
the ratio.

## Constraints
- `1 <= N,T <= 40`, `1 <= numBurners,numTargets <= 8`, `1 <= Pmax,F0 <= 20`.
- Deterministic integer-exact scoring; no timing, no randomness.

## Example
`N=5, T=3, H=1, Pmax=4, F0=1`, one burner at cell `2`, one target at cell `2`.
Firing `(b=0, s=0, d=1, p=4)` injects `4` at cell `2` at step `0`, then three
redistribution steps run. `h[2]` after step 3 comfortably exceeds `1`, so this
is feasible with fuel `4*1+1=5`. The baseline (`s=0,d=3,p=4`) costs
`4*3+1=13`, so this submission scores `min(1, 0.1*13/5) = 0.26`. (Illustrative
numbers only — real instances use larger, kernel-shaped grids.)
