# Privacy-Preserving Sensor Activation Codes

## Problem

A building operator wants to publish activation codes for private sensor queries. Each
codeword is a subset of sensors that will be activated together for one query type. To
avoid leaking too much about any one sensor, every codeword must use exactly the same
number of sensors and any two codewords may overlap only a limited amount. To make the
codes resistant to coalition attacks, no codeword may be hidden inside the union of two
other codewords.

In combinatorial terms, construct as many hyperedges as possible on `n` vertices. Every
hyperedge has size exactly `w`, every pair of hyperedges intersects in at most `lambda`
vertices, and the family is **2-cover-free**: for every three distinct hyperedges
`A, B, C`, it is never true that `A subseteq B union C`.

There is no known closed-form optimum for these finite cover-free constant-weight
families. Good constructions must balance local packing, pairwise overlap, and repair of
cover-free violations.

## Input (stdin)

```
n w lambda d R salt
```

`d` is always `2` in the generated tests. `R` is an output row cap, and `salt` is a
public deterministic salt that solvers may use to seed their own construction heuristics.

## Output (stdout)

Print a binary code:

```
m
row_1
row_2
...
row_m
```

`m` is the number of codewords. Each `row_i` must be a binary string of length `n`; a
`1` at position `j` means sensor `j` is active in that codeword. Output no labels,
comments, or extra tokens.

## Feasibility

The artifact is rejected and scores `0` unless all of the following hold:

- `0 <= m <= R`, and the output has exactly `1 + m` whitespace-separated tokens.
- Every row is a length-`n` string over `{0,1}` with exactly `w` ones.
- All rows are distinct.
- Every pair of rows has Hamming intersection at most `lambda`.
- For every row `A` and every two distinct other rows `B, C`, `A` has at least one `1`
  outside `B union C`.

Malformed integers, floats, `nan`, `inf`, wrong token counts, oversized output, duplicate
rows, and any violated constraint all score `0`.

## Objective (maximize)

Let `F = m`, the number of feasible activation codes submitted. Maximize `F`.

## Scoring

The checker builds its own deterministic reference construction `B` using a small
first-fit candidate stream. The score is

```
sc    = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```

Reproducing the reference construction scores about `0.1`; finding larger cover-free
families scores higher. The hardest instances leave substantial headroom below the cap.

## Constraints

- Generated tests satisfy `20 <= n <= 38`, `10 <= w <= 14`, `5 <= lambda <= 7`,
  `d = 2`, and `R <= 170`.
- Scoring is deterministic and uses exact integer bit operations only.

## Example (worked score)

For illustration, suppose `n = 6`, `w = 3`, `lambda = 1`, and the checker reference has
`B = 2`. The rows

```
2
111000
000111
```

are feasible: both have weight `3`, their pairwise overlap is `0`, and the cover-free
condition is vacuous for only two rows. The objective is `F = 2`, so the score is
`Ratio = min(1000, 100 * 2 / 2) / 1000 = 0.1`.
