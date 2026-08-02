# Dual-Channel Relay Mesh: Suppressing Monochromatic Cliques

## Problem
A relay mesh has `n` nodes, numbered `0..n-1`. Every pair of nodes shares a direct radio
link, and each link must be assigned one of two frequency **channels**, `0` or `1` (this is
a full 2-coloring of the edges of the complete graph `K_n`). A set of `k` nodes is a
**failure clique** if all `C(k,2)` links among them use the *same* channel: a single jammed
band then takes out every link in that cluster at once. Your job is to choose a channel for
every link so that failure cliques of size `k` are as scarce as possible.

## Input (stdin)
One line with two integers:
```
n k
```
`n` = number of relay nodes, `k` = the failure-clique size to suppress.

## Output (stdout)
`n` lines, each with `n` integers (`0` or `1`): the channel matrix `M`. `M[i][j]` is the
channel of the link between nodes `i` and `j`. Diagonal entries are ignored. The matrix
**must be symmetric**: `M[i][j] == M[j][i]` for every `i != j`.

## Feasibility
- Exactly `n` lines of `n` tokens each, every token exactly `0` or `1` (no floats, no
  `nan`/`inf`, no other characters).
- `M[i][j] == M[j][i]` for all `i != j`.

Any violation scores `Ratio: 0.0`.

## Objective (maximize suppression)
Let `V` = the exact number of `k`-node subsets that are failure cliques (same channel on
every internal link, counted over BOTH channels). Fewer failure cliques is better, i.e. you
want `V` as small as possible.

## Scoring
The checker also builds its own **baseline** channel matrix `B`-construction internally:
partition the `n` nodes into consecutive groups of size `k-1` (so no single group alone can
contain a `k`-clique); every link inside a group gets channel `0`; a link between two
different groups `gi`, `gj` gets channel `(gi + gj) mod 2`. This rule ignores clique counts
entirely -- it is not random and it is not the intended construction, just a cheap
deterministic reference. Let `B` be that construction's own failure-clique count `V`. With
`V` your feasible count,
```
sc    = min(1000, 100 * B / max(1, V))
Ratio = sc / 1000
```
so matching the baseline scores `0.1`, and cutting the failure-clique count to a tenth of
the baseline caps the score at `1.0`.

## Constraints
`18 <= n <= 45`, `k = 4`. (Every 2-coloring of `K_18` or larger contains at least one
monochromatic `K_4` -- a classical fact -- so `V >= 1` always; there is no way to reach a
perfect zero-violation matrix on these instances.) Runs in well under the time limit for
these sizes.

## Example
For `n = 6`, `k = 3` (illustrative shape only -- the graded instances use `k = 4`),
coloring link `(i,j)` by channel `1` iff `|i-j| mod 6` is in `{1, 2}` else `0` gives a
5-cycle-like structure with a handful of monochromatic triangles; a worse matrix (e.g. every
link on channel `0`) gives `V = C(6,3) = 20`, the maximum possible, and scores far below the
baseline. On the real instances, a matrix that reproduces the baseline group-parity rule
above scores exactly `Ratio = 0.1`; a matrix with a third as many failure cliques as the
baseline scores `Ratio = 0.3`.
