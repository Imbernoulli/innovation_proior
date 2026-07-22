# Doubling Rent: Leveled Circuits for Palindromic Polynomials

## Problem

In leveled homomorphic encryption, additions on ciphertexts are essentially free,
but every scalar **multiplication** consumes one level of the ciphertext's noise
budget -- and provisioning enough levels to survive a computation of multiplicative
*depth* `L` costs resources that grow like `2^L`. Two circuits that use the same
number of multiplications can therefore have wildly different rent depending only
on how those multiplications are *chained*.

You are given a single-variable polynomial `p(x) = a_0 + a_1 x + ... + a_d x^d`
with **palindromic** coefficients: `a_i = a_{d-i}` for every `i` (planted symmetry
you should exploit, not incidental). You must output a straight-line arithmetic
circuit ("program") that computes `p(x)` **exactly**, as a formal polynomial
identity in `x`.

## Input (stdin)

```
d
a_0 a_1 ... a_d
```
`d` is even, `4 <= d <= 32`. The `d+1` integers satisfy `a_i = a_{d-i}`, `a_0 != 0`
(so `p` has exact degree `d`), and `|a_i| <= 40`.

## Output (stdout)

```
L
<instruction 1>
...
<instruction L>
```
`L` is the number of instructions (`1 <= L <= 3000`). Instruction `k`
(`1 <= k <= L`) produces "wire `k`"; wire `0` is predefined to hold `x`. Each
instruction is one of:
```
C v         wire := the integer constant v      (|v| <= 1000000)
A i j       wire := wire_i + wire_j
S i j       wire := wire_i - wire_j
M i j       wire := wire_i * wire_j              (this is a MULTIPLICATION)
```
Operands `i, j` must reference an **earlier** wire (`0 <= i, j < k`; no cycles,
no forward references). The circuit's result is the value of wire `L`.

## Feasibility

The checker evaluates your circuit as an exact big-integer program at several
sample values of `x` (enough consecutive small integers to pin down any
degree-`<=d` candidate, plus a few large, widely spaced integers to catch a
higher-degree impostor) and compares against `p(x)` evaluated directly. **Any**
mismatch, parse error, out-of-range wire reference, or instruction/constant
outside the stated bounds scores `Ratio: 0.0`.

## Objective

Let `mult_count` = number of `M` instructions, and `mult_depth` = the length of
the longest chain of *dependent* multiplications (an `A`/`S` node's depth is the
max of its operands' depths, unchanged; an `M` node's depth is the max of its
operands' depths **plus one** -- additions never cost a level). Minimize:
```
F = mult_count * 2^mult_depth
```
The provably minimal `mult_count` for a generic degree-`d` polynomial is `d`,
achieved by Horner's rule -- but Horner's rule is a single sequential chain, so
`mult_depth = d` too, and `F` is exponential in `d`. Under this cost model
Horner's rule is the *worst* viable circuit for anything but tiny `d`; you must
restructure for depth, and the palindrome symmetry is exactly what lets you do
that without paying in multiplication count.

## Scoring

The checker computes `L(F) = ln(mult_count) + mult_depth * ln(2)` and normalizes
it between two analytic reference points: `L_base` (the "rebuild every power
from scratch, no sharing" naive scheme: `mult_count = d(d+1)/2`, `depth = d`)
and `L_floor` (an **unreachable** joint bound combining the true minimal count
`d` with the information-theoretic minimal depth `ceil(log2(d+1))` -- no real
circuit hits both at once). Then:
```
frac  = clip((L_base - L(F)) / (L_base - L_floor), 0, 1)
Ratio = 0.1 + 0.75 * frac
```
Lower `F` (fewer multiplications, and especially shallower chains, since depth
enters exponentially) gives a higher `Ratio`, capped at `0.85`.

## Constraints
Time limit 5s, memory 512MB, 10 test cases of increasing `d` (4 up to 32).

## Example (worked score, not one of the actual tests)
`d=2`, coefficients `3 5 3` (i.e. `p(x) = 3 + 5x + 3x^2`, palindromic since
`a_0=a_2=3`). A Horner circuit: `M`-chain `v=3; v=v*x+5; v=v*x+3` uses
`mult_count=2`, `mult_depth=2`. A paired circuit computes `x^2` (1 mult, depth 1),
then `3*(1+x^2)` (1 mult, depth 2) plus `5*x` (1 mult, depth 1), summed --
`mult_count=3`, `mult_depth=2`: at this tiny size the two are close, but the gap
widens sharply as `d` grows because `mult_depth` for Horner grows linearly while
a power-ladder construction keeps it logarithmic.
