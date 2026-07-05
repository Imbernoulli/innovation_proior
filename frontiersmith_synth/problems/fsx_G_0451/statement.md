# Minimize GF(2) Multiplications for a GF(2^k) Field Multiplier

## Problem
Finite-field arithmetic over GF(2^k) is the workhorse of elliptic-curve and error-correcting-code
hardware. A *bit-parallel* multiplier computes the product of two field elements as a fixed Boolean
circuit over the two primitive gates of GF(2):

- **AND** (`a & b`) = multiplication of two bits in GF(2), and
- **XOR** (`a ^ b`) = addition of two bits in GF(2).

The dominant, and genuinely hard-to-minimize, cost of such a multiplier is the number of **AND gates**
— the count of scalar GF(2) *multiplications* (its *multiplicative* / *bilinear* complexity). Schoolbook
uses `k*k` multiplications; Karatsuba-style and interpolation schemes use fewer, but the true minimum
for a given `k` is **open**. Your job: synthesize a correct multiplier using **as few AND gates as
possible**.

Field elements are represented in the polynomial basis. Element `a` is the bit vector
`(a_0, a_1, ..., a_{k-1})` standing for `a_0 + a_1 x + ... + a_{k-1} x^{k-1}`. The product is taken
modulo a fixed irreducible polynomial `f(x) = m_0 + m_1 x + ... + m_{k-1} x^{k-1} + x^k` given in the
input, so the result `c = a * b mod f` is again a length-`k` bit vector `(c_0, ..., c_{k-1})`. All
arithmetic is over GF(2).

## Input (stdin)
```
k
m_0 m_1 ... m_k        (k+1 coefficients of the modulus f, m_k = 1)
```

## Output (stdout): a GF(2) straight-line program
Wires are numbered. Wires `0 .. k-1` are the input bits `a_0 .. a_{k-1}`; wires `k .. 2k-1` are the
input bits `b_0 .. b_{k-1}`. Then you emit `G` gates; gate number `t` (0-based) produces wire
`2k + t`. Print:
```
G
<gate 0>
<gate 1>
...
<gate G-1>
o_0 o_1 ... o_{k-1}
```
Each gate line is `OP i j` where `OP` is `AND` or `XOR` and `i`, `j` are indices of **already-defined**
wires (`0 <= i, j < 2k + t`). The last line lists the `k` wire indices that carry the output bits
`c_0 .. c_{k-1}` (each in `0 .. 2k+G-1`; an output may point at an input or a gate wire).

## Feasibility
The circuit must compute the multiplication map **exactly**: for every one of the `2^(2k)` input
assignments `(a, b)`, wire `o_i` must equal bit `i` of `a * b mod f`. The checker verifies this by
comparing full Boolean truth tables (an exact proof). Any parse error, forward/undefined wire
reference, wrong token count, out-of-range index, or `G > 4000` is infeasible. A circuit that does not
compute the exact map scores `0`.

## Objective (minimize)
Minimize `F` = the number of **AND gates** (GF(2) multiplications). XOR gates are unrestricted (you may
use as many as you need) but are not part of the score.

## Scoring
Let `B = k*k` be the schoolbook multiplication count. A feasible circuit scores
```
ratio = min(1.0, 0.1 * B / F)
```
so the schoolbook baseline scores `0.1`, and driving the multiplication count down raises the score
(a 10x reduction would cap at `1.0`, which is unreachable here — the multiplicative complexity of
GF(2^k) multiplication is far above `k*k / 10`). Infeasible output scores `0`. Non-finite or malformed
tokens are rejected.

## Constraints
- `3 <= k <= 9`; `f` is irreducible of degree `k` (fixed per test).
- `0 <= G <= 4000`.

## Example (worked score)
Take `k = 4`, `f = x^4 + x + 1`. Schoolbook uses `B = 16` AND gates and scores `0.1`. A one-level
Karatsuba multiplier computes the degree-6 product with `2*2^2 + 2^2 = 12` AND gates, then reduces mod
`f` using only XORs; the multiplication count drops to `F = 12`, scoring `min(1, 0.1*16/12) = 0.1333`.
A fully recursive Karatsuba multiplier reaches `F = 9`, scoring `min(1, 0.1*16/9) = 0.1778`. Fewer
multiplications, higher score.
