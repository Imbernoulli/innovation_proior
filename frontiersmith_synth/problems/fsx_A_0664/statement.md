# Gray-Adjacent State Encoding

## Problem
You are given a complete deterministic transition table on `N` states (numbered
`0..N-1`) and `M = 4` input symbols (numbered `0..3`): `next[u][s]` is the state
reached from state `u` on symbol `s`. `N` is guaranteed to be a power of two; let
`B = log2(N)`.

Before this automaton can be built as two-level (sum-of-products) combinational logic,
every state must be given a distinct **B-bit codeword**. Since `N = 2^B`, your codeword
assignment must use *every* codeword exactly once (a bijection `states -> {0,1}^B`).

Once codewords are fixed, the transition function splits into `B` independent
**next-state-bit functions**: bit `i` of the next state's codeword is a Boolean function
`phi_i` of `B + 2` Boolean inputs — the `B` bits of the current state's codeword plus the
2 bits of the input symbol. The checker rewrites each `phi_i` in a canonical minimized
sum-of-products form (a deterministic prime-implicant / greedy-cover minimizer) and counts
the total number of **literals** (one occurrence of a variable or its negation in a
product term) summed over all `B` functions. Fewer literals means cheaper logic.

The same transition RELATION, re-labelled through a different bijection, can need
wildly different literal counts — this is the classical *state-assignment problem* from
digital design. States whose transition BEHAVIOR is alike (they respond similarly across
symbols, directly or through where their successors eventually lead) should end up at low
Hamming distance from each other, so the minimizer's product terms merge into fewer,
larger cubes. Numbering states in the order the input happens to list them (or any other
encoding that ignores this behavioral adjacency) typically leaves the logic dense — the
input order carries no information about which states behave alike.

## Input (stdin)
```
N K
```
(`K = 2` always, `M = 2^K = 4`), then `N` lines, each with `M` integers: line `u` gives
`next[u][0] next[u][1] next[u][2] next[u][3]`.

`4 <= N <= 32`, `N` a power of two.

## Output (stdout)
`N` lines. Line `u` (0-indexed) is a length-`B` string over `{0,1}`: the codeword
assigned to state `u`.

## Feasibility
The output must have exactly `N` lines, each of length exactly `B`, each character `0`
or `1`, and all `N` codewords pairwise distinct (so they exactly cover `{0,1}^B`). Any
violation scores `Ratio: 0.0`.

## Objective
Minimize `F`, the total literal count across the `B` minimized next-state-bit functions.

## Scoring
Let `B0` be `F` computed under the naive identity encoding (`codeword(u) = binary(u)`,
which the checker also builds and scores internally). With `F` your literal count,
```
Ratio = min(1, 0.1 * B0 / F)
```
A 10x reduction in literals versus the naive baseline already reaches the score cap.

## Example (illustrative form only — a hand-built toy, unrelated to the hidden structure
of the real test data)
`N=4, K=2, B=2`:
```
next[0] = 0 3 1 2
next[1] = 1 2 0 3
next[2] = 2 1 3 0
next[3] = 3 0 2 1
```
The naive identity encoding (`0->00, 1->01, 2->10, 3->11`) gives `F = 16` literals
(`Ratio = 0.1000`, matching the baseline by construction). But state 0's behavior is
really governed by two INDEPENDENT one-bit toggles hidden behind the labelling: relabel
`0->01, 1->11, 2->10, 3->00` (i.e. treat state `u`'s codeword high bit as "cluster" and
low bit as "element", each flipped by only one symbol bit) and every next-state-bit
function collapses to depend on just 2 of the 4 variables, giving `F = 8` literals —
`Ratio = min(1, 0.1*16/8) = 0.2000`, twice the naive score. On the real test data the
hidden structure is not this simple and the state ids are shuffled so that it is not
visible from the input order at all — it must be discovered from the transition table
itself.

## Constraints
- `4 <= N <= 32`, `N` a power of two, `M = 4`.
- Time limit 5s, memory 512MB per test case.
