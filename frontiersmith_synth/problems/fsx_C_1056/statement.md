# One Keyboard, Three Quarrelling Languages: Envelope-Optimal Keyslot Assignment

## Problem
A hardware vendor is designing ONE physical keyboard that must serve `K = 3`
writing communities at once. The keyboard has `N` key **slots** laid out on a
staggered 3-row grid (given as 2-D coordinates). There are `N` distinct
**symbols** (letters/glyphs), and you must choose a **bijection** (a
permutation) from symbols to slots — the keyboard's layout.

Each language `k` supplies its own `N x N` **digraph frequency table**
`freq_k[i][j]`: how often symbol `i` is immediately followed by symbol `j` in
that language's corpus. Typing a digraph costs the Euclidean **travel
distance** between the two symbols' assigned slots. A language's total
**finger-travel cost** under your layout is the frequency-weighted sum of
these travel distances over all digraphs. The three languages do NOT agree on
what matters: they may share some common ground, but each also has its own
idiosyncratic high-frequency digraphs the others barely use — and their
corpora can be wildly different sizes.

To make languages comparable, each language's cost is divided by that
language's own **random-layout baseline** (the closed-form expected cost
under a uniformly random layout, given ITS OWN digraph table). This
normalization is scale-free: it does not care how large a language's raw
corpus is, only how good the layout is relative to what that language would
get from doing nothing clever. You are scored by the **envelope** — the
WORST (maximum) normalized cost across the three languages. A layout that
looks great on average but abandons one language scores badly; you must
serve all three.

## Input (stdin)
```
N K
x_0 y_0
...
x_{N-1} y_{N-1}
freq_0 row 0 (N ints)
...
freq_0 row N-1
freq_1 row 0
...
freq_{K-1} row N-1
```
`(x_i, y_i)` are slot `i`'s coordinates. `freq_k[i][j]` (row i, column j of
language k's block) is a nonnegative integer digraph count; `K = 3` always.

## Output (stdout)
Print `N` whitespace-separated integers: token `i` is the slot index (in
`[0, N-1]`) assigned to symbol `i`. Must be a permutation of `0..N-1`.

## Feasibility
Rejected (score `0`) if: the output does not contain exactly `N` tokens; any
token is not a finite integer in `[0, N-1]`; or the tokens are not all
distinct (not a bijection).

## Objective (minimize)
For language `k`, `TC_k = sum_{i != j} freq_k[i][j] * dist(slot(i), slot(j))`
where `dist` is Euclidean distance. Let `S_k = sum_{i!=j} freq_k[i][j]` and
`T_avg` = the average travel distance over all ordered pairs of distinct
slots (fixed by the grid). The random-layout baseline is `base_k = T_avg *
S_k`, and the normalized cost is `NC_k = TC_k / base_k`. The graded objective
is `F = max_k NC_k`.

## Scoring
Let `B` = `F` evaluated at the **identity** assignment (symbol `i` to slot
`i`) — the checker's own trivial construction. Then
```
sc    = min(1000, 100 * B / F)
Ratio = sc / 1000
```
so the identity layout always scores exactly `0.1`; a layout with a 10x
better envelope caps at `1.0`. Fully deterministic exact arithmetic (up to
double-precision Euclidean distance).

## Constraints
- `6 <= N <= 28`, `K = 3` fixed.
- All `freq_k[i][j] >= 0`, integers; `freq_k[i][i] = 0`.
- Grid coordinates are fixed real numbers with no two slots coincident.

## Example (illustrative FORM only — smaller than real instances, not the
planted three-way structure)
`N=4, K=2` (only for illustration), unit-square slots `(0,0),(1,0),(0,1),(1,1)`.
Language A: only `freq_A[0][3]=100`. Language B: only `freq_B[1][2]=8`.
`T_avg = 1.138071`, so `base_A = 113.807`, `base_B = 9.1046`.

Identity assignment puts symbols `0,3` on a diagonal (`dist=sqrt(2)`) and
`1,2` on the other diagonal (`dist=sqrt(2)`), so `NC_A = NC_B = 1.2427 = B`
(as always, `Ratio=0.1` here by construction).

Submitting `0 2 3 1` (symbol0->slot0, symbol1->slot2, symbol2->slot3,
symbol3->slot1) puts BOTH digraph pairs on adjacent (`dist=1`) slots:
`TC_A=100`, `NC_A=0.8787`; `TC_B=8`, `NC_B=0.8787`; `F=0.8787`. Then
`sc = min(1000, 100*1.2427/0.8787) = 141.42`, `Ratio = 0.14142`.
