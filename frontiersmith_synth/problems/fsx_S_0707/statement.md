# Leaked Cipher Layer: Minimal XOR Rewiring

## Problem

A cipher's linear diffusion layer over GF(2) has leaked as a dense `m x m`
0/1 matrix `M`: it maps an `m`-bit input `x` to an `m`-bit output `y = M x`
(all arithmetic mod 2). You must **rewire** this layer as a straight-line
program of two-input XOR gates, using as few gates as possible, so it can be
re-implemented cheaply.

You do not see how `M` was built. It only LOOKS dense and unstructured; the
matrix may (or may not, on any given row) share hidden structure across rows
that a careful synthesizer can exploit to save gates -- or may not, in which
case there is nothing to exploit and a plain per-row expansion is close to
optimal. Either way your program must compute `M x` **exactly**, for every
`x`, using only XOR gates.

## Input (stdin)

```
m
row_1
row_2
...
row_m
```
`row_i` is a string of `m` characters `0`/`1`: `M[i][j]` is character `j`
(1-indexed) of `row_i`.

## Output (stdout)

```
K
a_1 b_1
a_2 b_2
...
a_K b_K
out_1 out_2 ... out_m
```
- Lines are numbered `1..m` for the inputs `x_1..x_m`, then `m+1..m+K` for
  the `K` gates you define, in order.
- Line `a_t b_t` (the `t`-th gate, defining line `m+t`) means
  `line[m+t] := line[a_t] XOR line[b_t]`. Both `a_t` and `b_t` must reference
  an **earlier** line (an input, or a gate you already defined) -- no forward
  references, no cycles.
- `out_i` is the line index whose value equals `y_i` for every input `x`.

## Feasibility

Your program is checked by simulating it once, bit-parallel, against all `m`
standard basis inputs simultaneously; because XOR gates are GF(2)-linear,
matching every `out_i` against row `i` of `M` on that single simulation
certifies `y = M x` for **every** `x in {0,1}^m` at once. Any malformed
line (bad indices, forward reference, wrong token count, non-integer token)
or any row that doesn't match `M` exactly scores 0 -- there is no partial
credit for an incorrect circuit.

## Objective

Minimize `K`, the number of XOR gates.

## Scoring

The checker builds its own baseline `B`: the cost of computing every output
row **independently**, chaining together that row's own set input bits with
no sharing across rows at all (`B = sum_i max(0, weight(row_i) - 1)`). Your
score is
```
Ratio = min(1, 0.1 * B / K)
```
So matching the independent-chain baseline scores `0.1`; using a shared
sub-circuit to cut `K` to a tenth of `B` caps the score at `1.0`. Any
malformed or incorrect program scores `0`.

## Constraints

`16 <= m <= 100` across the 10 test cases (increasing size). Time limit 5s,
memory 512MB. `K` is bounded by a generous cap; absurdly large or malformed
outputs are rejected.

## Example (worked, illustrative shape only -- not a real test case)

Suppose `m = 4` and row 1 of `M` is `1101` (i.e. `y_1 = x_1 XOR x_2 XOR x_4`).
An independent-chain solver would spend 2 gates on this row alone: line 5
`:= x_1 XOR x_2`, line 6 `:= line5 XOR x_4`, `out_1 = 6`. If some OTHER row,
say row 3, happens to equal `1101` too (or differs from it by only one or two
bits), a smarter program reuses line 6 (or line 6 XOR one correction bit)
for row 3 instead of re-deriving it from scratch -- one shared gate serving
two outputs. The real test cases plant this kind of cross-row reuse
opportunity at a much larger, harder-to-see scale; finding it (rather than
just pairing up columns you happen to notice) is what separates a strong
program from a merely competent one.
