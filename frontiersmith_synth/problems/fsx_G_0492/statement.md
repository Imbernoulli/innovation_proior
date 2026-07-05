# Nonlinear S-box Design: Minimizing the Walsh Peak of an n-bit Permutation

## Problem
A block-cipher **S-box** of width `n` is a bijection `S : {0,1}^n -> {0,1}^n`,
given as a lookup table `S(0), S(1), ..., S(2^n - 1)`. The single most important
cryptographic quality of an S-box is its **nonlinearity**: how far, in Hamming
distance, its output bits stay from *every* affine function. A low-nonlinearity
S-box is fatally vulnerable to linear cryptanalysis.

Nonlinearity is governed by the **Walsh (Walsh-Hadamard) transform**. For masks
`a, b in {0,1}^n` write the bit dot-product `a . x = XOR_i (a_i AND x_i)` and
define
```
W_S(a, b) = SUM_{x in {0,1}^n} (-1)^{ (b . S(x)) XOR (a . x) } .
```
The **linearity** of `S` is the largest Walsh peak over all *output* masks:
```
L(S) = max over b != 0, over all a  of  |W_S(a, b)| .
```
The nonlinearity is `NL(S) = 2^{n-1} - L(S)/2`, so **maximizing nonlinearity is
exactly minimizing the Walsh peak `L(S)`**. Your job: design the permutation
whose Walsh peak is as small as possible.

## Input (stdin)
```
n
```
A single integer `n` (the S-box width).

## Output (stdout)
Print `2^n` integers `S(0), S(1), ..., S(2^n - 1)`, whitespace-separated
(one per line is fine). Value `S(x)` is the image of input `x`. The table must
be a **permutation** of `{0, 1, ..., 2^n - 1}`.

## Feasibility
An output is valid iff **all** hold:
- it contains exactly `2^n` integers;
- every integer lies in `[0, 2^n - 1]`;
- the integers are pairwise distinct (i.e. the table is a bijection).

Any violation -- wrong count, out-of-range, repeated value, or a non-integer /
non-finite token -- scores `Ratio: 0.0`.

## Objective
Minimize the Walsh peak `L(S)` (equivalently, maximize nonlinearity
`NL(S) = 2^{n-1} - L(S)/2`).

## Scoring
The checker builds its own trivial reference: the **identity** permutation
`S(x) = x`. Being affine, each of its nonzero output bits is a linear function
with a single full-height Walsh peak, so `L(identity) = 2^n`. This is the
baseline `B = 2^n`. With minimization normalization:
```
sc    = min(1000.0, 100.0 * B / max(1e-9, L(S)))
Ratio = sc / 1000.0
```
Reproducing the identity scores `Ratio = 0.1`; driving the Walsh peak down by a
factor of `10` caps at `1.0`. Because `L(S) >= 2^{n/2+1}` for a permutation, the
score has genuine headroom and no reachable saturation for the tested widths.

## Constraints
- `4 <= n <= 8` across the test ladder (`16 <= 2^n <= 256`).
- The checker recomputes the full Walsh spectrum; it is `O(n * 4^n)` and
  deterministic.
- Time limit 5s, memory 512m.

## Example
Take `n = 4` (worked score, illustrative only). The identity table
`0 1 2 ... 15` has Walsh peak `L = 16 = B`, giving
`sc = 100 * 16 / 16 = 100`, i.e. `Ratio = 0.100`.

The multiplicative-inverse S-box over `GF(2^4)` (with `0 -> 0`) achieves the
optimal `L = 8` (`NL = 4`), giving `sc = 100 * 16 / 8 = 200`, i.e.
`Ratio = 0.200`. A random permutation typically lands in between (e.g.
`L = 12`, `Ratio ~ 0.133`). No affine table can score above `0.1`, and pushing
the Walsh peak below the algebraic construction -- an open question for most `n`
-- is what earns the top of the scale.
