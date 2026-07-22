# Canyon Rewrite: Expand Before You Can Collapse

## Problem

You are given a sequence of *n* symbols (each symbol an integer label from 1
to *k*), a positive integer **cost** per symbol label, and two families of
local rewrite rules you may apply to the sequence, in any order, up to a
total budget of *r* applications:

- **Expand** a single symbol `v` at position `i` into an ordered pair `x y`,
  replacing that one symbol with two — but ONLY if `v` has an expand rule
  `v -> x y` (some symbols have none; no symbol has more than one).
- **Collapse** an adjacent pair `x y` at positions `i, i+1` into a single
  symbol `z` — but ONLY if the ordered pair `(x, y)` has a collapse rule
  `x y -> z` (most pairs have none).

Every rule that exists for this instance is listed explicitly in the input —
look it up, don't guess. A rule application touches only the symbols it
consumes and produces; it never affects the rest of the sequence beyond
shifting positions by the local insertion/deletion. The **cost of a
sequence** is the sum of the costs of its symbols.

An application is not required to make the immediate cost better: its local
cost change equals cost(produced) − cost(consumed), which can be positive
(worse) or negative (better). Some collapse rules only become legal after an
earlier expand has changed which symbols are adjacent — an expand can be
locally cost-INCREASING yet be the only way to reach a much cheaper sequence
a few steps later.

Your goal: choose a sequence of at most *r* rule applications (in order) so
the FINAL sequence's total cost is as small as possible.

## Input (stdin)

```
n k r
s_1 s_2 ... s_n              (initial sequence, 1 <= s_i <= k)
cost_1 cost_2 ... cost_k     (cost_v = cost of symbol label v, positive)
m_e
v x y                        (m_e lines: symbol v expands to ordered pair x,y)
m_c
x y z                        (m_c lines: adjacent pair x,y collapses to symbol z)
```
All values fit in signed 64-bit integers; `n` up to a few thousand.

## Output (stdout)

```
m
op_1 pos_1
...
op_m pos_m
```
`m` (0 <= m <= r) is how many rule applications you make. Each following
line is `E pos` (expand the symbol at 1-indexed position `pos`) or `C pos`
(collapse the adjacent pair at 1-indexed positions `pos, pos+1`). Positions
refer to the sequence as it stands after all previously listed applications
(length changes +1 on expand, −1 on collapse). `m = 0` (just print `0`) is
always legal and means "do nothing."

## Feasibility

Score 0 if: the move-count line is missing or not a single integer in
`[0, r]`; the number of move lines disagrees with it; any move line is not
exactly `<op> <pos>`; `pos` is out of range at that point; the referenced
symbol has no matching expand rule (`E`) or the referenced pair has no
matching collapse rule (`C`); or any parsed value is not finite.

## Objective and Scoring

Let `B` be the cost of the ORIGINAL sequence (the do-nothing cost) and `F`
the cost of your FINAL sequence after your moves. The checker reports
`Ratio: 100 * B / F` (capped at 1.0, i.e. reaching 1/10th of `B` already
caps the score). Smaller `F` is better; doing nothing scores low but never
zero.

## Constraints

- 1 <= n <= 6000, 1 <= k <= 20, 0 <= r <= 2n.
- 1 <= cost_v <= 1000.
- Each symbol has at most one expand rule; each ordered pair has at most one
  collapse rule.
- Time limit: 5 seconds. Memory: 512 MB.

## Example (illustrative shape only — not the actual rule table)

Suppose symbol 1 costs 50 and expands to `2 3` (costs 20, 35). Symbol 3
followed by symbol 4 (cost 40) collapses to symbol 5 (cost 2). Sequence
`1 4` costs `50+40=90`. Expanding position 1 gives `2 3 4`, cost
`20+35+40=95` — WORSE. But now positions 2,3 form the pair `3 4`, which
collapses to `5`: the sequence becomes `2 5`, cost `20+2=22`. Two moves took
the cost from 90, briefly UP to 95, then down to 22 — far better than any
single move alone could reach.
