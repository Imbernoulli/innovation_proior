# Cancel-and-Share: Minimum-XOR Straight-Line Programs for GF(2) Linear Maps

## Problem
You are given a target linear map over GF(2): an `m x n` binary matrix `M`. For an input
vector `x` in GF(2)^n, the map produces `M x`, i.e. output row `i` is the XOR of the inputs
selected by the 1s of row `i`.

Build a **linear straight-line program** of two-input XOR gates that computes all `m`
outputs, and make it use **as few gates as possible**.

## Input (stdin)
```
m n
<m rows, each n space-separated bits (0/1)>   # row i of M
```
`1 <= m, n <= 40`. Every row has weight >= 2.

## Output (stdout)
A circuit over **nodes**. Nodes `0..n-1` are the inputs `x_0..x_{n-1}`. Then declare `G`
gates; gate `k` (0-indexed) is node `n+k`.
```
G
a_0 b_0        # gate 0 = node[a_0] XOR node[b_0]
a_1 b_1
...
a_{G-1} b_{G-1}
o_0 o_1 ... o_{m-1}   # node index realising each output row (in order)
```
Each operand of gate `k` must be a **strictly earlier** node (index `< n+k`); operands may
be equal (`a==b` yields the zero vector). Every `o_i` is any node index in `[0, n+G)`.

## Feasibility
The circuit is valid iff, evaluated as GF(2) vectors, `node[o_i]` equals row `i` of `M` for
every `i` (exact equivalence for all `2^n` inputs). Any parse error, out-of-range index,
forward reference, wrong output, or non-integer token scores **0**.

## Objective
Minimise the gate count `F = G`.

## Scoring
Let `B = sum_i (weight(row_i) - 1)` be the cost of folding each row independently with no
reuse (the checker's internal baseline). With `F` your gate count, the score is
```
Ratio = min(1.0, 0.1 * B / F)
```
so the independent-fold baseline scores `0.1`, and driving the gate count down raises the
score (capped at `1.0`). Fewer gates is strictly better.

## What makes it hard
Folding each row on its own wastes work; reusing shared partial sums helps, **but** any
shared intermediate you reuse for a row can only contain bits that row actually needs. When
rows are dense (most bits set), that restriction is severe. A different regime opens up if
you allow a partial sum to include bits it will later **cancel** (`x XOR x = 0`): for a dense
row it can be cheaper to build one big shared sum and XOR away the few bits it should not
contain than to add up the many bits it should. Cancellation-free sharing cannot reach those
circuits. Balancing when to add and when to cancel-and-subtract, across all rows, is the game.

## Example
Input `M` = one row `1 1 1` (`n=3`). Independent fold: `x_0^x_1` then `^x_2` = 2 gates,
`B=2`, `Ratio` = `0.1*2/2 = 0.1`. A submission using 1 gate would score `0.1*2/1 = 0.2`.
(For a single weight-3 row 2 gates is forced; the savings appear when many dense rows share
a common sum.)

## Constraints
Time limit 5 s, memory 512 MB. Deterministic scoring only.
