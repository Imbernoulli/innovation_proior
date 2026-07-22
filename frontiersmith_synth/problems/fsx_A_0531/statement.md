# The Bookbinder's Pocket Fold

A bookbinder must fold a long paper map down to fit a pocket **without cracking the
sheet**. The map is a strip of **N** unit cells in a row, cell `0` on the left to
cell `N-1` on the right. Cell `c` has a **thickness** `h[c] >= 1` (laminated cells
are thicker). Between cell `c` and cell `c+1` lies **crease `c`** (`0 <= c <= N-2`).
Some creases are **reinforced** (a stitched spine): bending one costs extra stress `W`.

You fold the map with a sequence of **simple folds** and must end with a footprint of
width at most **T**. Score is lower total stress; the harness normalises it (below).

## Fold rules

At any moment the paper occupies `w` footprint positions ("slots"), left to right;
initially `w = N` and slot `j` holds just cell `j`. A **simple fold** names a fold
line `k` with `1 <= k <= w-1` (between slot `k-1` and slot `k`). The **shorter side**
of the line is flipped over onto the other side and its slots stack, in mirror image,
onto the aligned slots of the fixed side (ties fold the left side):

- if `k <= w-k`, slots `0..k-1` reflect onto slots `k..2k-1`, new width `w-k`;
- otherwise slots `k..w-1` reflect onto slots `2k-1-j`, new width `k`.

Stacked slots simply merge (their cell sets and thicknesses combine). A cell never
leaves the strip; folds only reduce the width.

**Stress of one fold** = (total thickness carried on the flipped side) + `W * b`,
where `b` = the number of **reinforced** creases *bent by this fold*. Crease `c` is
bent by a fold at line `k` iff cells `c` and `c+1` currently sit on opposite sides of
line `k` (one in a slot `< k`, the other in a slot `>= k`).

The key subtlety: whenever a side is flipped, **all** of its accumulated layers are
carried through the fold. Fold a thick pile late and often and you pay for it every
time; keep piles thin (or keep the thick region on the *fixed* side) and you pay less.

## Feasibility

Your fold sequence is **feasible** iff every line `k` is in range `1..w-1` at the
moment it is applied and, after all folds, the footprint width is `<= T`. Anything
else (missing folds, out-of-range line, final width `> T`, non-integer token) scores 0.

## Objective

Minimise the **total stress** `F` = the sum of the per-fold stresses over your whole
schedule.

## Scoring

Let `B` be the stress of the accordion baseline (fold one slot at a time from the
left, `k = 1`, repeated `N - T` times). For a feasible schedule with total stress `F`,

```
Ratio = min(1, 0.075 * B / F)
```

so reproducing the baseline scores `~0.075` and a much cheaper schedule climbs toward
`1.0`. You want `F` far below `B`.

## Input (stdin)

```
N T R W
h[0] h[1] ... h[N-1]
c_1 c_2 ... c_R        (the R reinforced crease indices; blank line if R = 0)
```

## Output (stdout)

```
M
k_1 k_2 ... k_M        (the M fold lines, in order; omit the line if M = 0)
```

## Constraints

`2 <= N <= 20`, `1 <= T < N`, `0 <= R <= N-2`, `1 <= h[c] <= 20`, `W >= 0`.
Time limit 5s, memory 512 MB.

## Example

Input:
```
4 1 0 0
1 1 1 1

```
The accordion baseline folds `k=1` three times: it flips a slot of thickness 1, then 2,
then 3, for stress `1+2+3 = 6`, so `B = 6`. The schedule `2 1` folds the middle first
(flip 2 layers, stress 2), then folds the two double-slabs together (flip 2 layers,
stress 2): total `F = 4`, giving `Ratio = 0.075*6/4 = 0.1125`. Because the map here is
uniform there is little room to do better; on maps with a heavy centre the *order* of
folds changes `F` dramatically.
