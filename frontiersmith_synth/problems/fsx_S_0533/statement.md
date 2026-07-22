# Collapse the Automaton: Synchronizing a Hidden Cerny Machine

You are given a deterministic finite automaton (DFA): `n` states labelled `0..n-1` and `m`
input symbols labelled `0..m-1`. Reading a symbol moves the machine from a state to another
state according to a transition table `delta`, where `delta[s][i]` is the state you reach from
state `i` on symbol `s`.

A **reset word** (a.k.a. synchronizing word) is a sequence of symbols `w = s_1 s_2 ... s_L`
such that, no matter which state the machine starts in, after reading `w` it always ends in
**the same single state**. Formally: start with the set of *all* states active; applying a
symbol `s` replaces the active set `S` by `{ delta[s][i] : i in S }`; after applying all of
`w`, exactly one state must remain active.

This automaton is guaranteed to be synchronizable. Your job: **output a reset word that is as
short as possible.**

## Input (stdin)

```
n m
delta[0][0] delta[0][1] ... delta[0][n-1]     # row for symbol 0
delta[1][0] delta[1][1] ... delta[1][n-1]     # row for symbol 1
...                                            # m rows total
```
All transitions are integers in `[0, n-1]`. Sizes satisfy `2 <= n <= 40`, `1 <= m <= 8`.

## Output (stdout)

The reset word: whitespace-separated symbol indices, each in `[0, m-1]`. The word length `L`
must satisfy `1 <= L <= 200000`. Example: `1 0 0 1 0` is the word of length 5 that applies
symbol 1, then 0, 0, 1, 0.

## Feasibility

Your output is rejected (score `0`) unless every token is an integer in `[0, m-1]`, the length
is in `[1, 200000]`, and simulating the word from the set of all `n` states collapses it to
exactly one state.

## Objective

Minimize `F`, the number of symbols in your reset word.

## Scoring

The checker builds its own **baseline** reset word `B` (a valid but deliberately long
construction) and reports

```
ratio = min(1.0, 0.1 * B / F)
```

So a word matching the baseline length scores about `0.1`, and shorter words score higher; the
ratio saturates only for a word ten times shorter than the baseline. Your per-case score is
this ratio; the total is the mean over all test cases.

## Why this is hard (and what to notice)

Two natural attacks disappoint:

* **Powerset BFS.** Searching the `2^n` subsets for a true shortest reset word is exact but
  explodes long before these sizes.
* **Pairwise merging.** The textbook polynomial heuristic repeatedly picks two currently-active
  states and appends a shortest word that merges them (found by a backward breadth-first search
  over pairs). It always produces a valid reset word, but it is *blind to the automaton's global
  structure* and on these instances it lands well above the true optimum.

These transition tables are not random. They hide an **algebraic skeleton**: one symbol acts as
a permutation that cycles all `n` states in a hidden order, and one symbol acts as a single
*contraction* that fuses exactly one state into its neighbour along that cycle (leaving every
other state fixed). The labels and symbol order are scrambled, so you must **recover** this
skeleton from the table. Once you see the cycle and where the contraction bites, you can drive
the whole active set around the cycle and pinch it down one state at a time, producing a reset
word of near-quadratic length — far shorter than what pairwise merging gives. Reading the
planted permutation/contraction algebra, not grinding pairwise merges, is the intended insight.

## Example (worked scoring — illustrative sizes only)

Suppose `n = 5` and a submitted word has length `F = 20`, while the checker's baseline word has
length `B = 60`. Then `ratio = min(1.0, 0.1 * 60 / 20) = 0.30`. A competitor who found a length-`16`
word would score `0.1 * 60 / 16 = 0.375`. (These numbers are illustrative and are **not** the
values used by any test case.)
