# Chemist's Scale: Signed Weighing Design

## Problem
A chemist has `n` candidate reference masses of unknown weight and an old-style two-pan
balance. In one *weighing session* she may place some of the `n` references on the
**left pan** (coefficient `+1`), some on the **right pan** (coefficient `-1`), and leave
the rest off the scale (coefficient `0`) — using **exactly `k`** of the `n` references
each session (the balance's mechanism only tolerates exactly `k` loaded arms). She runs
`n` independent sessions and records an `n x n` design matrix `W`, row `i` holding the
`{-1,0,1}` coefficients used in session `i`.

If the sessions' readings are to be inverted into `n` *independent, equally-precise*
mass estimates, the **session-covariance matrix** `W W^T` must be as close as possible
to `k * I` (the identity scaled by `k`): the diagonal is automatically `k` (each row has
exactly `k` entries of magnitude 1, so `(WW^T)_{ii} = k` always), but every **off-diagonal**
entry `(WW^T)_{ij}` (`i != j`) measures unwanted correlation between session `i` and
session `j` — nonzero means sessions `i` and `j` can't be un-mixed cleanly. The chemist
wants to choose the `n x n` sign/support pattern `W` that minimizes total leftover
correlation.

Design matrices with `W W^T = k I` **exactly** are called *weighing matrices* `W(n,k)`;
they are known to exist for only special `(n,k)` pairs, so for most inputs perfection is
unreachable and you must minimize the defect instead.

## Input (stdin)
```
n k
```
`n` is the number of references / sessions, `k` is the fixed number of loaded arms per
session (`1 <= k < n <= 200`).

## Output (stdout)
Print `n` lines, each with `n` space-separated integers from `{-1, 0, 1}`: row `i` is the
coefficient vector used in session `i`.

## Feasibility
An output is valid iff **all** hold:
- exactly `n` lines are printed, each with exactly `n` integer tokens (all finite);
- every token lies in `{-1, 0, 1}`;
- every row has **exactly `k`** nonzero entries.
Any violation scores `Ratio: 0.0`.

## Objective
Let `W` be the submitted `n x n` matrix and `G = W W^T`. Minimize the **defect**
```
F = sum_{i != j} |G_{i,j}|
```
(the diagonal never contributes, since it is forced to `k` by feasibility). `F = 0` means
a perfect weighing matrix.

## Scoring
The checker builds its own reference design `B`: session `i` loads references
`i, i+1, ..., i+k-1` (indices mod `n`) all on the left pan (`+1`) — the "obvious" sliding
window, whose defect we call `B` (always `> 0` for `1 <= k < n`). With minimization
normalization and a raised cap (so a perfect `F = 0` still leaves headroom above the best
attainable score):
```
sc = min(880.0, 100.0 * B / max(1e-9, F))
Ratio = sc / 1000.0
```
Reproducing the sliding-window design scores `Ratio = 0.1`; any design with
`F <= B/8.8` (about `11.4x` less defect) saturates the `0.88` cap.

## Constraints
- `4 <= n <= 200`, `3 <= k <= n - 1`.
- Time limit 5s, memory 512MB.

## Example
`n = 4, k = 3`. Submitting
```
0 1 1 1
1 0 1 -1
1 -1 0 1
1 1 -1 0
```
gives `G = W W^T` with diagonal `3,3,3,3` and every off-diagonal entry `0`, so `F = 0`
(a perfect weighing matrix `W(4,3)`). The sliding-window baseline `B` for `n=4,k=3` has
defect `24`, so this submission scores `Ratio = min(880, 2400)/1000 = 0.880`. The
sliding-window design itself scores `Ratio = 0.100`.
