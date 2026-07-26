# Covering Every DFA Transition with the Fewest Symbols

You are given a complete deterministic finite automaton (DFA): `n` states, an
alphabet of `k` symbols, and a start state `s0`. Every `(state, symbol)` pair has
exactly one outgoing transition (the transition table is a total function). You
must write a **test suite** — a set of input strings — such that when every string
is run from `s0` it exercises every one of the `n*k` transitions at least once.
Minimise the **total length** (total number of symbols) of your test suite.

## Input (stdin)

```
n k s0
sym_1 sym_2 ... sym_k
row_0            (k integers: target state for symbol_1..symbol_k from state 0)
row_1
...
row_{n-1}
```
`sym_i` are single lowercase letters (the alphabet, fixed order used by every row).
`row_i[j]` is the state reached from state `i` reading `sym_j`. `4 <= n <= 15`,
`2 <= k <= 3`.

## Output (stdout)

```
m
str_1
str_2
...
str_m
```
`m >= 0` is the number of test strings; each `str_i` is a (possibly empty) string
over the alphabet, written on its own line with no separators between symbols
(e.g. `abba`). Every string is run **from `s0`** independently.

## Feasibility

- `m` must parse as a non-negative integer, and exactly `m` further lines must
  follow (an empty line is a valid empty string).
- Every character of every string must be one of the `k` alphabet symbols.
- Total output size is bounded (at most 200000 lines, at most 2,000,000 symbols
  total) — any violation, or any malformed line, scores `0`.
- Replaying every string from `s0` must, over their union, exercise **all** `n*k`
  transitions at least once. If even one transition is never exercised, the
  suite is infeasible and scores `0`.

## Objective & Scoring

Let `F` be the total length (sum of `len(str_i)`) of your suite. The checker also
builds an internal baseline `B`: for **every transition independently**, the
shortest path (in symbols) from `s0` to that transition's source state, followed
by that transition's symbol, emitted as its **own separate string** (this always
covers everything, but restarts from `s0` for every single transition). Your score is

```
Ratio = min(1.0, 0.1 * B / F)
```

so the per-transition baseline scores `~0.1`; you must do meaningfully better to
climb, and the score saturates only once you are roughly `10x` shorter than it.

## What makes it hard

Covering each transition with its own start-to-source path is correct but wildly
redundant: it walks the same prefixes over and over and never reuses a string's
tail as the head of the next requirement. The task is really a **single covering
walk**: a Chinese-postman-style edge tour of the transition graph. If every
state's in-degree equals its out-degree the whole graph already admits an Euler
walk visiting every transition exactly once. When it does not (some states are
'hubs' with lopsided in/out-degree from the way other rows route into them), you
must duplicate a minimum-cost set of existing paths to rebalance the graph before
an Euler walk exists — and that duplication should be routed via **shortest
paths between the right pairs of imbalanced states**, not wherever a walk
happens to get stuck. A single continuous string built by chasing the *nearest*
uncovered transition, state-index by state-index, easily makes bad long-range
detours that a globally-planned rebalancing avoids.

## Example (worked, small illustrative DFA — not from the real generator)

`n=2, k=2, s0=0`, symbols `a b`; row 0 = `1 0` (a->1, b->0); row 1 = `0 1` (a->0, b->1).
The single string `"abab"` run from state 0 visits: `0 -a-> 1` (covers `(0,a)`),
`1 -b-> 1` (covers `(1,b)`), `1 -a-> 0` (covers `(1,a)`), `0 -b-> 0` (covers
`(0,b)`) — all 4 transitions in one string of length 4, so `F = 4`. The checker's
per-transition baseline needs `(0,a)`:1, `(0,b)`:1, `(1,a)`: dist(0->1)+1=2,
`(1,b)`: dist(0->1)+1=2, so `B = 6`, giving `Ratio = min(1, 0.1*6/4) = 0.15`.

## Constraints

Time limit 5s, memory 512MB. `4 <= n <= 15`, `2 <= k <= 3`. Scoring is fully
deterministic given the input.
