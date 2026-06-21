# Context — Splitting a Number into More Than Two Parts

## Research question

Two integers `x` and `y`, each `n` digits long in a fixed base, must be multiplied exactly,
and the cost is measured by the asymptotic count of elementary digit operations as `n` grows.
The schoolbook method forms every digit-against-digit partial product and costs `Θ(n²)`.
The recent divide-and-conquer split-in-half method cuts each operand into two halves and, by
sharing one product across the two cross terms, needs only three half-size multiplications,
giving `Θ(n^{log₂ 3}) ≈ Θ(n^{1.585})` — already below the `n²` that was long believed to be a
floor.

The question is whether splitting operands into more than two parts, combined with appropriate
algebraic recombination, can reduce the asymptotic exponent further, and if so, how to carry it
out with ordinary positional integer arithmetic.

## Background

**The cost model.** Operands are written in a fixed radix `b`; an operation is an
add/subtract/multiply of single digits, plus the shifts (multiplication by powers of the
radix) that place a partial result. For two `n`-digit operands the input occupies `~2n`
symbols and the product `~2n+1`, so any method needs `Ω(n)` operations just to read input and
write output. The interesting quantity is `M(n)`, the least number of operations sufficient to
multiply, and how far above the trivial `Ω(n)` it sits.

**Schoolbook multiplication and its `n²`.** For every digit `xᵢ` and digit `yⱼ` form the
one-digit product `xᵢ·yⱼ`, place it at position `i+j`, and sum with carries: `n²` single-digit
products and `O(n²)` additions. Every pair of digits genuinely interacts in the product, which
made `n²` feel forced.

**The half-split method (the immediate ancestor, Karatsuba–Ofman 1962).** Cut each operand at
the middle, `x = x₁Bᵐ + x₀`, `y = y₁Bᵐ + y₀` with `m = n/2`. Then
```
x·y = x₁y₁·B²ᵐ + (x₁y₀ + x₀y₁)·Bᵐ + x₀y₀ .
```
The naive reading needs four half-size products, giving `T(n) = 4T(n/2) + O(n) = Θ(n²)`.
The saving is that the middle coefficient needs only the *sum* `x₁y₀ + x₀y₁`, and that sum is
recovered from one extra product of the half-sums minus the two corner products already in hand:
```
x₁y₀ + x₀y₁ = (x₁+x₀)(y₁+y₀) − x₁y₁ − x₀y₀ .
```
Three half-size products, `T(n) = 3T(n/2) + O(n) = Θ(n^{log₂ 3})`.

**Limbs and the digit convolution.** Cutting `x` into limbs writes `x = Σ_{i} x_i (Bᵐ)^i`;
multiplying two such expansions produces, at each output place, the sum `Σ_{i+j=ℓ} x_i y_j` of
limb-against-limb products before the carries are settled. The half-split above is the two-limb
instance of this: its three output quantities `x₁y₁, x₁y₀+x₀y₁, x₀y₀` are precisely those
place-sums.

**Polynomial facts on the table.** A polynomial of degree `d` is uniquely determined by its
values at any `d+1` distinct points; recovering the coefficients from those values is
interpolation, the inverse of evaluation. The linear system relating coefficients to values at
points `s₀,…,s_d` has the Vandermonde matrix `[s_i^j]`, whose determinant `∏_{i<j}(s_j−s_i)` is
nonzero precisely when the points are distinct, so the system is solvable. Over the integers
the solution is integral when the values come from an integer-coefficient polynomial, even
though intermediate fractions appear, and Gaussian elimination can be arranged to keep every
intermediate entry an integer.

**The squaring/multiplication equivalence and residue systems.** Multiplication and squaring
are interchangeable up to a constant via `xy = ¼[(x+y)² − (x−y)²]`. A residue number system
(represent a number by residues mod the first `k` primes) makes multiplication componentwise
and cheap, but conversion to and from positional form, and even comparing two numbers, is
expensive, so it does not lower `M(n)` for positional input/output.

## Baselines

**Schoolbook long multiplication.** Core idea: form every digit-by-digit product `xᵢyⱼ` and
accumulate at place `i+j`. Cost `Θ(n²)` single-digit products and additions. Exact and simple.
In the polynomial view it is the full convolution of the two limb polynomials — `k²` coefficient
products for a `k`-limb split.

**Half-split with three products (Karatsuba–Ofman 1962).** Core idea: two-limb polynomial
split, compute the three coefficients with three sub-products by sharing one product across the
two cross terms. Algorithm:
```
z₂ = x₁y₁,  z₀ = x₀y₀,  z₁ = (x₁+x₀)(y₁+y₀) − z₂ − z₀,
x·y = z₂B²ᵐ + z₁Bᵐ + z₀ .
```
Cost `T(n) = 3T(n/2) + O(n) = Θ(n^{log₂ 3}) ≈ n^{1.585}`.

**Naive `k`-part split (the `k²` convolution).** Core idea: cut each operand into `k` limbs,
form the limb polynomials `p, q` of degree `k−1`, and compute the `2k−1` coefficients of
`p·q` directly by convolution. Algorithm: `c_ℓ = Σ_{i+j=ℓ} x_i y_j`, then evaluate at `Bᵐ`.
Cost: `k²` sub-products of size `n/k`, so `T(n) = k²·T(n/k) + O(n) = Θ(n²)` for every fixed
`k` (since `log_k k² = 2`).

## Evaluation settings

The yardstick is the asymptotic count of elementary digit operations as a function of operand
length `n`, against the schoolbook reference `Θ(n²)` and the half-split reference
`Θ(n^{1.585})`. Operands are `n`-digit integers in a fixed radix; correctness is exact,
checked against the true product for all inputs; the regime of interest is large `n`, where the
exponent dominates constant factors. The analytic instrument is the recursion-tree /
master-theorem solution of `T(n) = a·T(n/k) + O(n)`: with branching `a` and shrink factor `k`
and linear combine work, the leaf term `n^{log_k a}` dominates whenever `log_k a > 1`, fixing
the growth rate. A second, finer measure is parallel depth (the count of dependent steps); the
combine work per level is `O(n)` (additions, subtractions, shifts, and the solution of a fixed
linear system in the split parameter).

## Code framework

The primitives already available: arbitrary-size integers, the schoolbook product as a base
operation, integer floor-division and remainder to cut an operand into limbs, shifting by
multiplying by powers of the radix, and exact integer linear algebra for a fixed, small system.
The scaffold is a recursive routine with a base case, a fixed multi-limb split, and one empty
algebraic slot to be filled in.

```python
BASE = 10
THRESHOLD = 3   # operands with few enough limbs are multiplied directly

def multiply_candidate(x, y, base=BASE):
    if base <= 1:
        raise ValueError("base must be greater than 1")

    if x < 0 or y < 0:
        sign = -1 if (x < 0) ^ (y < 0) else 1
        return sign * multiply_candidate(abs(x), abs(y), base)

    # base case: small operands -> direct multiply (O(1))
    if x < base ** THRESHOLD or y < base ** THRESHOLD:
        return x * y

    # choose a limb size so each operand has at most three limbs in base**m
    n = max(len(str(x)), len(str(y)))
    m = n // 3 + 1
    B = base ** m

    # cut each operand into limbs: x = sum x_i * B**i  (the limb polynomial p)
    x0, x1, x2 = (x % B), (x // B) % B, (x // (B * B))
    y0, y1, y2 = (y % B), (y // B) % B, (y // (B * B))

    # TODO: combine the limbs into the product and return it.
    return ...  # TODO
```
