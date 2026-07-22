# Colotomic Grid: Phasing a Nested Gong Cycle

## Problem
A gamelan-style **colotomic cycle** has length `N` beats (`0..N-1`, wrapping). It is played
by two groups of instruments:

- A **structural chain** of `D` instruments, from slowest to fastest, where instrument `i`
  plays exactly `k_i` onsets per cycle and `k_1 | k_2 | ... | k_D | N` (each layer's onset
  count divides the next). The colotomic rule: **every onset of chain layer `i` must also be
  an onset of chain layer `i+1`** (the slower gong always strikes on a beat the faster
  instrument also marks).
- `R` **free** instruments, each playing exactly `f_j` onsets per cycle (`f_j | N`), with no
  nesting requirement of their own.

At most `cap` of these `D+R` instruments may sound on the same beat, cycle-wide.

You must output the full onset set of every instrument. An instrument's **evenness** score
rewards onsets that are as close as possible to equally spaced (this only depends on the
*shape* of its gap pattern, not on which beat it starts at — rotating a perfectly-spaced
pattern is still perfectly spaced). The **texture** score rewards the union of *all* onsets
spreading its density fairly evenly across the whole cycle, rather than clumping into a few
regions.

## Input (stdin)
```
N D R cap
k_1 k_2 ... k_D
f_1 f_2 ... f_R
```

## Output (stdout)
`D+R` lines, in the same order as the input (chain layers first, then free instruments).
Line `i` holds the `k_i` (or `f_j`) onset positions of that instrument, space-separated
integers in `[0, N)`.

## Feasibility
- Instrument `i` must print exactly its required onset count; all values in `[0, N)`,
  pairwise distinct within that instrument.
- Nesting: for every `i < D`, the onset set of chain layer `i` must be a subset of layer
  `i+1`'s onset set.
- Cap: for every beat `t`, at most `cap` of the `D+R` instruments may include `t`.
Any violation scores `Ratio: 0.0`.

## Objective
For instrument `i` with onset multiset gaps `g_1..g_k` (circular, mean `mu = N/k`):
`evenness_i = mu^2 / (mu^2 + Var(g))`, where `Var` is the population variance — `1.0` for
perfectly equal gaps, smaller as gaps get ragged. Let `U` be the union of every instrument's
onsets (duplicates merged). Split the cycle into 24 equal arcs and let `H` be the Shannon
entropy (base 2) of how `U`'s points fall into those arcs, normalized by `log2(min(24,|U|))`.
Maximize:
```
F = 0.9 * average_i(evenness_i) + 0.1 * H
```

## Scoring
The checker also builds its own simple reference `B`: each chain layer gets onsets
`{0,...,k_i-1}` (nested prefixes, badly bunched); each free instrument gets its own block of
consecutive integers parked at an evenly spaced anchor around the cycle (always collision-free,
still badly bunched). Then:
```
Ratio = min(1000, 100 * F / max(1e-9, B)) / 1000.0
```
Reproducing the reference scores `0.1`; doing `10x` better caps at `1.0`.

## Constraints
- `60 <= N <= 500`, `2 <= D <= 3`, `1 <= R <= 7`, all `k_i`, `f_j` divide `N` and are `>= 3`.
- `cap >= D` always (the nested chain alone forces `D` simultaneous onsets at its coarsest
  beats).
- Time limit 5s, memory 256m.

## Example
`N=72, D=2, R=1, cap=3`, chain `k = (3, 6)`, free `f = (4)`. One valid, high-scoring answer:
```
0 24 48
0 12 24 36 48 60
9 27 45 63
```
Layer 1 (`{0,24,48}`) is a subset of layer 2 (`{0,12,...,60}`): nesting holds. The free
instrument's onsets `{9,27,45,63}` never touch layer 2's onsets, so the beat-0 pileup (both
chain layers, count 2) never grows past `cap=3`. Every instrument's gaps are perfectly equal,
so `average evenness = 1.0`; the merged 10 points (6 from the chain's union, 4 from the free
instrument, all distinct) also spread cleanly across the 24 arcs, so
`H = 1.0`, giving `F = 1.0`. The checker's own reference `B` (prefixes `0,1,2` / `0..5` plus a
bunched free block) evaluates to about `0.302`, so this answer scores
`Ratio = min(1000, 100*1.0/0.302)/1000 = 0.331`.
