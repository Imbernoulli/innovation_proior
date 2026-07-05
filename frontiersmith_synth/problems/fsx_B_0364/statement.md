# Seabed Splice Router: Minimum-SWAP Channel Routing on a Cable Mesh

## Problem
A deep-sea observatory maintains a network of `n` seabed **junction boxes** wired by
**cables**. Each box currently holds exactly one **signal channel** (a token); the boxes
and cables form a fixed, sparse graph (a rectangular mesh plus a few long-haul trunk
cables).

A maintenance crew must perform an ordered **splice schedule** of `q` operations. Splice
`i` names two channels `(a, b)` that have to be spliced together. A splice can only be
performed while its two channels sit on **directly-cabled** (adjacent) boxes.

To bring distant channels together the crew performs **SWAPs**: choosing one cable
`(u, v)`, they exchange the channels resting on boxes `u` and `v`. Each SWAP is one unit
of work. Splices must be executed **in the given order**.

Emit a routing — an interleaving of SWAPs and splice executions — that carries out the
entire schedule using **as few SWAPs as possible**. This is exactly qubit-routing /
token-swapping on a fixed coupling map; the true minimum is NP-hard and unknown.

## Input (stdin)
```
n e q
```
then `e` lines each `u v` (an undirected cable between boxes `u` and `v`, `0 <= u,v < n`);
then one line of `n` integers `p[0] p[1] ... p[n-1]` — `p[s]` is the channel initially on
box `s` (a permutation of `0..n-1`); then `q` lines each `a b` — splice `i` (1-based, in
order) joins channels `a` and `b`.

## Output (stdout)
A whitespace-separated list of **moves**, each one of:
```
S u v      # SWAP: exchange the channels on boxes u and v; (u,v) must be a cable
G i        # execute splice i (1-based); i must be the NEXT pending splice, and its
           # two channels must currently rest on adjacent (cabled) boxes
```
You may place moves on one or many lines. The tokens `nan`/`inf` are rejected.

## Feasibility
The checker tracks the live channel-to-box mapping while replaying your moves. The output
is feasible iff: every `S u v` names a real cable and valid boxes; every `G i` targets the
next pending splice (indices `1,2,...,q` in order) with its two channels on adjacent boxes;
and all `q` splices are executed. Any illegal swap, out-of-order or non-adjacent splice,
missing splice, malformed token, or non-finite value scores **0**. Because every splice is
applied to its true channel pair, in order, at adjacency, a feasible routing is certified
**functionally equivalent** to the original schedule.

## Objective
Minimize `F`, the total number of `S` (SWAP) moves.

## Scoring
The checker builds its own baseline `B`: the naive **reset-to-home** router that, for each
splice, routes the first channel to its partner along a shortest cable path (`d-1` swaps
for home distance `d`) and then undoes those swaps to restore the home placement — cost
`2*(d-1)` per splice, summed over the schedule. With your swap count `F`,
```
Ratio = min(1, 0.1 * B / F)
```
The reset-to-home baseline scores `0.1`. Halving the swap count doubles the ratio; reaching
a tenth of `B` caps at `1.0`. The minimum-SWAP routing is unknown, so headroom remains.

## Constraints
- Graph is connected; `1 <= q`; boxes are `0..n-1`.
- Deterministic integer scoring; no timing, no randomness in the score.

## Example
Suppose the checker computes `B = 40` for some instance. A routing using `F = 20` SWAPs
scores `min(1, 0.1 * 40 / 20) = 0.2`; one using `F = 10` scores `0.4`. The reset-to-home
baseline (`F = 40`) scores exactly `0.1`.
