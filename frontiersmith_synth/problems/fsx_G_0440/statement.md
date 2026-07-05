# XOR-Count of a GF(2) Linear Layer: Minimal 2-Input XOR Straight-Line Program

## Problem
A block cipher's diffusion layer applies a fixed **GF(2)-linear map** `y = M x`, where
`M` is an `m x n` matrix over the field GF(2) (arithmetic mod 2). In hardware the map
is realized as a circuit of **2-input XOR gates**. Your job is to compute the whole map
with **as few XOR gates as possible** — the classic *XOR-count* / linear straight-line
program minimization problem.

You are given `M`. Emit a straight-line program: a sequence of 2-input XOR gates over
the `n` input bits, then a choice of which computed signal realizes each of the `m`
output bits. The checker verifies the program computes **exactly** `y = M x` for every
input, then counts the gates. Fewer gates is better.

## Input (stdin)
```
m n
row_0          (n entries, each 0 or 1)
row_1
...
row_{m-1}
```
`M[j][i]` is entry `i` of row `j`. Output bit `j` equals the GF(2) sum (XOR) of the
input bits `x_i` for which `M[j][i] = 1`.

## Output (stdout)
```
G
a_1 b_1
a_2 b_2
...
a_G b_G
o_0 o_1 ... o_{m-1}
```
- Signals are indexed by integers. **Inputs** `x_0..x_{n-1}` are signals `0..n-1`.
- `G` is the number of XOR gates. Gate `t` (for `t = 1..G`) defines a new signal with
  index `n + t - 1`, whose value is `signal[a_t] XOR signal[b_t]`. Both `a_t` and `b_t`
  must reference **strictly earlier** signals, i.e. `0 <= a_t, b_t < n + t - 1`.
- The final line lists, for each output `j = 0..m-1`, the index `o_j` of the signal
  equal to row `j` of `M`. Any valid signal index `0 <= o_j < n + G` may be used
  (an input may be reused directly, and one signal may serve several outputs).

## Feasibility
A submission is rejected (score 0) if any token is non-integer / non-finite, the token
count is wrong, a gate references a not-yet-defined signal, an output index is out of
range, or **any** output signal fails to equal its target row of `M` exactly.

## Objective
**Minimize** `G`, the number of XOR gates.

## Scoring
Let `B = sum_j (popcount(row_j) - 1)` be the naive per-row cost (build each output by
chaining its set bits with no sharing). The score is
```
Ratio = min(1, 0.1 * B / G)
```
Reproducing the naive baseline (`G = B`) scores `0.1`; halving the gate count scores
`0.2`; a 10x reduction caps at `1.0`. The true optimum is NP-hard and unknown, so
headroom is genuine — no polynomial method is known to reach it.

## Constraints
- `10 <= m, n <= 30` across the difficulty ladder; each row has popcount `>= 3`.
- `0 <= G <= 500000`. Scoring is exact integer GF(2) arithmetic — fully deterministic.

## Example
For `M` with rows `y_0 = x_0 ^ x_1 ^ x_2` and `y_1 = x_1 ^ x_2 ^ x_3` (`m=2, n=4`),
one valid program shares the common `x_1 ^ x_2`:
```
3
1 2
0 4
3 4
5 6
```
Signal 4 = `x_1^x_2`, signal 5 = `x_0^(x_1^x_2) = y_0`, signal 6 = `x_3^(x_1^x_2) = y_1`;
outputs `o_0 = 5`, `o_1 = 6`. This uses `G = 3` gates. Here `B = (3-1)+(3-1) = 4`, so
`Ratio = min(1, 0.1 * 4 / 3) = 0.1333`. The naive program would use `G = 4` gates and
score `0.1`.
