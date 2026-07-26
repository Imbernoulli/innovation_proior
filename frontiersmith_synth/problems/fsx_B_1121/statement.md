# Aligned Road Coloring for Fast Synchronization

## Problem

You are given a directed, strongly connected road network on `n` intersections
(vertices), numbered `0..n-1`. Every intersection has exactly `m` outgoing
road segments ("slots"), numbered `0..m-1` in a fixed order; each slot leads
to some intersection (self-loops and repeated targets are allowed). The road
segments themselves are fixed and given to you -- your job is to **paint
each intersection's `m` slots with the `m` colors `0..m-1`**, one color per
slot, no color repeated at an intersection (i.e. at every vertex the coloring
is a bijection from colors to slots).

A coloring turns the network into a deterministic finite automaton: state =
intersection, alphabet = colors, and applying color `a` at intersection `v`
moves you along whichever slot at `v` was painted `a`. A coloring is
**synchronizing** if some finite sequence of colors (a "reset word"), applied
starting from *every* intersection simultaneously, drives all of them to the
very same intersection. (Concretely: track the *set* of currently-possible
locations, starting from all `n` of them; applying one color to the whole
set replaces it with the set of colors' targets; the word is a reset word
once this set has size 1.)

Not every coloring is synchronizing -- and among the ones that are, reset
words can be short or long. Your goal: choose a synchronizing coloring whose
**shortest reset word is as short as possible**.

## Input (stdin)

```
n m
t[0][0] t[0][1] ... t[0][m-1]
...
t[n-1][0] ... t[n-1][m-1]
```
`t[v][i]` is the destination of vertex `v`'s slot `i` (`0 <= t[v][i] < n`).

## Output (stdout)

`n` lines, each with `m` integers: a permutation of `0..m-1`. The `i`-th
number on line `v` is the color assigned to slot `i` of vertex `v`.

## Feasibility

Each output line must be a permutation of `0..m-1` (exactly `m` tokens,
finite integers, each value in `0..m-1` used exactly once). Any violation
scores `0.0`. Additionally, the resulting automaton must be synchronizing;
if no reset word exists, the coloring scores `0.0` regardless of format
validity.

## Scoring

Let `F` be the length of your coloring's shortest reset word (found by the
checker via an exact breadth-first search over reachable location-sets --
always tractable at this scale). The checker also builds one fixed reference
coloring of its own from the raw road network (not shown to you) and scores
its shortest reset word `B > 0` the same way. Since shorter is better:

```
ratio = min(1000, 100 * B / F) / 1000
```

A coloring matching the reference's quality scores around `0.1`; a
meaningfully shorter reset word scores higher, capped at `1.0`.

## Constraints

`6 <= n <= 11`, `m = 4`, time limit 5s, memory 512MB.

## Example

Suppose `n=3`, `m=2`, with slots `t[0]=(1,0)`, `t[1]=(2,0)`, `t[2]=(0,1)`.
Output
```
1 0
1 0
1 0
```
means vertex 0's slot 0 (-> vertex 1) is colored 1 and slot 1 (-> vertex 0,
a self-loop) is colored 0; similarly for 1 and 2. Under color 0: vertex 0
stays at 0, vertex 1 goes to 0, vertex 2 goes to 1. Applying color 0 twice to
the full set `{0,1,2}` gives `{0,1,2} -> {0,1} -> {0}` -- a reset word of
length 2. If the checker's own reference construction on this graph had
shortest reset word `B=4`, this submission would score
`min(1000, 100*4/2)/1000 = 0.2`.

## Why this is open-ended

Many colorings are feasible; among the synchronizing ones, reset-word length
varies enormously depending on which physical edges you align with which
color. There is no known closed-form optimum for general instances -- the
shortest reset word is only ever found by search, and the best coloring
uses the graph's specific structure, not just any spanning-tree-shaped
recipe.
