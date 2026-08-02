# Row Placement Under Channel Congestion and Timing Slack

## Problem
A chip has `n` standard cells that must be placed into the `n` slots of a
single layout row (a bijection: each cell gets a distinct slot `0..n-1`).
Between consecutive slots `g` and `g+1` runs a routing channel with a fixed
integer **capacity** `capacity[g]` — the number of wires that may physically
cross that gap.

The netlist is a set of **nets**; a net is a group of `k >= 2` cell indices
that must be electrically connected. Once cells are placed, a net's route is
its **bounding interval**: if its terminal cells land on slots with minimum
`lo` and maximum `hi`, the net is wired straight through every channel gap
`g` with `lo <= g < hi`, contributing exactly `hi - lo` to that net's
**span** (its wirelength) and one unit of usage to each of those `hi-lo`
channels. Some nets are flagged **timing-critical**; each such net carries an
explicit **slack** bound, and its span must not exceed that bound (a longer
route means the signal cannot reach its destination within a clock period).

Minimizing every net's span independently pulls its cells close together —
but many nets sharing the same cluster of slots pile their usage onto the
same few channels, which can blow straight through the channel capacities
even though the *total* wire used is small. A layout must balance "keep
wires short" against "keep the channels that matter unclogged," and it must
never let a timing-critical net drift long while doing so.

## Input (stdin)
```
n_cells n_nets
capacity[0] capacity[1] ... capacity[n_cells-2]
k crit slack c_1 c_2 ... c_k        (one line per net, n_nets lines)
```
`crit` is `1` if the net is timing-critical (then `slack >= 0` bounds its
span) or `0` (then `slack` is `-1`, unused). `c_1..c_k` are distinct cell
indices in `[0, n_cells-1]`. Constraints: `8 <= n_cells <= 26`,
`n_cells <= n_nets <= 2*n_cells + 5`, `2 <= k <= n_cells`.

## Output (stdout)
Exactly `n_cells` integers: `pos[0] pos[1] ... pos[n_cells-1]`, where
`pos[i]` is the slot assigned to cell `i`. This must be a permutation of
`0..n_cells-1` (whitespace/newlines free).

## Feasibility
1. Output parses as exactly `n_cells` integers forming a permutation of
   `0..n_cells-1`; otherwise the run scores 0.
2. Every timing-critical net's span (`max(pos[c]) - min(pos[c])` over its
   terminals) must not exceed its `slack`; otherwise the run scores 0.
3. For every channel gap `g`, the number of nets whose interval covers `g`
   must not exceed `capacity[g]`; otherwise the run scores 0.

## Objective
Minimize the **total wirelength** `F = sum` over all nets of their span.

## Scoring
Let `B` be the total wirelength of the fixed **identity placement**
(`pos[i] = i` for all `i`) — always a valid reference value the checker
computes directly from the input (it need not itself be feasible for your
specific submission, it is just the normalizer). With your feasible `F`:
```
Ratio = min(1.0, 0.1 * B / F)
```
Matching `F = B` scores `0.1`; a `10x` reduction in wirelength caps the
score at `1.0`. An infeasible output scores `0.0`.

## Constraints
- `8 <= n_cells <= 26`, time limit 5s, memory 512MB.
- All capacities/slacks are non-negative integers fixed in the input; the
  instance always admits at least one feasible layout.
- Deterministic integer arithmetic only; no randomness in scoring.

## Example (worked, small)
`n_cells=4`, nets: `{0,1}` critical slack `1`, `{2,3}`, `{0,3}`. Identity
placement `pos=[0,1,2,3]` gives spans `1, 1, 3` so `B = 5`. Placing
`pos = [0,1,3,2]` gives spans `1` (net `{0,1}`), `1` (net `{2,3}`, slots
`3,2`), `2` (net `{0,3}`, slots `0,2`) so `F = 4`, `Ratio = min(1, 0.1*5/4)
= 0.125`. If channel capacities were tight enough that squeezing cell `3`
next to cell `0` overflowed a gap, that placement would instead score `0`.
