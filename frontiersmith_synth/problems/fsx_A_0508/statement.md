# Ridge Sculptor: Staged Pours Under the Angle of Repose

You sculpt a 1-D ridge out of sand. There are `N` cells in a row (indices `0..N-1`),
each with an integer height, all starting at `0`. Walls at both ends: grains never
leave the row. You are given a **target profile** `t[0..N-1]` and must approximate it
by pouring grains in exactly `K` ordered **stages**.

At stage `k` you output a cell `c_k` and an integer grain count `g_k` (`0 <= g_k <= G`).
The `g_k` grains land on cell `c_k`; then the pile **settles** by an angle-of-repose rule
before the next stage begins.

## Settling (deterministic, abelian)

While some adjacent pair violates the slope limit, a grain slides from the higher cell
to its lower neighbour:

- while `h[i] - h[i+1] > S` for some `i`, move grains right; while `h[i+1] - h[i] > S`,
  move grains left — until `|h[i] - h[i+1]| <= S` for every adjacent pair.

The stable state is **unique** (independent of the order of slides), and settling only
adds mass locally, so heights **never decrease**: a poured grain is permanent, and later
pours flow downhill over everything already settled.

## Output (stdout)

Exactly `K` lines, line `k` = `c_k g_k` (two integers). Any other token count, an index
outside `0..N-1`, a grain count outside `0..G`, or a non-integer/`nan`/`inf` token makes
the submission infeasible (score 0).

## Objective (minimise)

Let `h^(k)` be the settled profile after stage `k`. Define

```
shortfall = sum_i max(0, t[i] - h^(K)[i])                      # final undershoot
integ_over = sum_{k=1..K} sum_i max(0, h^(k)[i] - t[i])        # overshoot, every stage
cost = shortfall + L * integ_over
```

`L`, the overshoot weight, is given in the input. Overshoot is charged at **every** stage
it exists: because heights never fall, material poured above target early is paid for again
and again. You cannot remove grains, so you cannot undo an overshoot.

The reason a sharp target is hard: where the target drops from a plateau of height `H` to
a notch, `H - 1 > S`, so the settled pile **cannot** hold that wall — building the plateau
forces runoff into the notch, an overshoot no strategy can avoid. Matching the target
cell-by-cell as early as possible therefore over-pours and pays the flood many times; the
edge comes from keeping the pile under target and shaping the plateaus late, reserving the
notches for the runoff you cannot prevent.

## Scoring

The checker replays your pours, computes `cost`, forms an internal baseline `B` (the
do-nothing cost `sum_i t[i]`), and reports `Ratio = min(1000, 100 * B / cost) / 1000`
(higher is better; do-nothing ≈ `0.1`).

## Input (stdin)

```
N K S L G
t[0] t[1] ... t[N-1]
```

## Constraints

`4 <= N <= 40`, `8 <= K <= 25`, `1 <= S <= 2`, `1 <= L <= 4`, `0 <= t[i]`. One instance
per test; 10 tests of increasing size. Time limit 5 s, memory 512 MB.

## Example

Input:
```
9 9 1 3 10
1 5 5 5 1 5 5 5 1
```
Two plateaus of height 5 separated by notches at height 1, with `S = 1` so each wall of
height 4 exceeds the slope limit. A submission is 9 lines like `2 4` / `6 4` / … . Pouring
onto a plateau raises it but spills a grain into the neighbouring notch (an overshoot of
about `H - S - 1 = 3` there); doing that in stage 1 pays `3` overshoot for all 9 stages,
whereas deferring it to the last stages pays it far fewer times — the same final shape,
a very different `cost`.
