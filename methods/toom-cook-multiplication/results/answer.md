# Toom–Cook Multiplication — k-Way Splitting via Evaluate / Pointwise-Multiply / Interpolate

## Problem

Multiply two `n`-digit integers exactly, faster than the half-split method
(`Θ(n^{log₂ 3}) ≈ n^{1.585}`). Toom–Cook generalizes the half-split from `k = 2` parts to an
arbitrary `k`, achieving `Θ(n^{log_k(2k-1)})`, an exponent that decreases with `k` and tends to
`1` as `k → ∞` (so multiplication runs in `n^{1+ε}` for any `ε > 0`). The `k = 3` case
(Toom-3) runs in `Θ(n^{log₃ 5}) ≈ n^{1.465}`.

## Key idea

A number cut into `k` limbs is a degree-`(k-1)` polynomial sampled at the limb base:
```
x = Σᵢ x_i Bⁱ = p(B),   p(t) = x_{k-1}t^{k-1} + … + x₁t + x₀ ,
```
and likewise `y = q(B)`. Then `x·y = (p·q)(B)`, so integer multiplication is **polynomial
multiplication** of `p, q` followed by one evaluation at `t = B` (a shift-and-add, `O(n)`).

The product polynomial `r = p·q` has degree `2k-2`, so it is determined by its values at any
`2k-1` distinct points. Each value is a single number product:
```
r(s) = p(s)·q(s) .
```
So instead of the convolution (`k²` coefficient multiplications), compute `r` by:
1. **Evaluate** `p` and `q` at `2k-1` points (cheap linear combinations of the limbs);
2. **Pointwise multiply** `r(s) = p(s)q(s)` — the only recursive multiplications, `2k-1` of them,
   each on operands of size `n/k`;
3. **Interpolate** the `2k-1` coefficients of `r` from its `2k-1` values (solve a Vandermonde
   system; exact over the integers);
4. **Recompose** `x·y = r(B) = Σ r_ℓ Bˡ`.

Recurrence and complexity:
```
T(n) = (2k-1)·T(n/k) + O(n)  ⇒  T(n) = Θ(n^{log_k(2k-1)}) .
```
`k = 2` gives `log₂ 3 = 1.585` (the half-split, points `{0, ∞, 1}`); `k = 3` gives
`log₃ 5 = 1.465`; the exponent `log_k(2k-1) → 1` as `k → ∞`.

## Toom-3 algorithm

Split into 3 limbs: `p(t) = x₂t² + x₁t + x₀`, `q(t) = y₂t² + y₁t + y₀`, product `r` degree 4
(coefficients `r₀..r₄`), five points `{0, 1, -1, -2, ∞}`.

**Evaluate** (`p`; `q` identical):
```
p(0)=x₀,  p(1)=x₀+x₁+x₂,  p(-1)=x₀-x₁+x₂,  p(-2)=x₀-2x₁+4x₂,  p(∞)=x₂ .
```
**Pointwise multiply** (the five recursive products):
```
v₀=p(0)q(0),  v₁=p(1)q(1),  v₋₁=p(-1)q(-1),  v₋₂=p(-2)q(-2),  v∞=p(∞)q(∞) .
```
**Interpolate** `r₀..r₄` (exact; every `/3`, `/2` lands on an integer):
```
r₀ = v₀
r₄ = v∞
r₃ = (v₋₂ − v₁) / 3
r₁ = (v₁ − v₋₁) / 2
r₂ = v₋₁ − v₀
r₃ = (r₂ − r₃) / 2 + 2·v∞
r₂ = r₂ + r₁ − v∞
r₁ = r₁ − r₃
```
(Inverse of the Vandermonde matrix of `{0,1,-1,-2,∞}`; the freebie points `0` and `∞` give
`r₀` and `r₄` directly.)
**Recompose**: `x·y = r₄B⁴ + r₃B³ + r₂B² + r₁B + r₀`, `B = baseᵐ` (shifts + adds; negative or
oversized `r_ℓ` are absorbed by big-integer carries/borrows).

## Why exact

`r = p·q` has integer coefficients, so the Vandermonde system over `ℤ` has an integer solution.
Run as Gaussian elimination, each step uses `P(s_i) − P(s_j) = (s_i − s_j)·Q(s_i)` for monic
`P, Q`, factoring out the point-difference cleanly — no remainder. (For limb polynomials modulo
`b`, this needs `b` prime so the ring is an integral domain and the system has a unique
solution.) The small points `{0, 1, -1, -2, ∞}` keep evaluations to adds/shifts and the
interpolation constants tiny.

## Worked example

`x = 123456`, `y = 654321`, base `10`, `m = 2`, `B = 100`:
high-to-low limbs `x = (12, 34, 56)`, so `x₀=56, x₁=34, x₂=12`; `y = (65, 43, 21)`, so `y₀=21, y₁=43, y₂=65`.
```
p(0)=56,  p(1)=102,  p(-1)=34,  p(-2)=36,  p(∞)=12
q(0)=21,  q(1)=129,  q(-1)=43,  q(-2)=195, q(∞)=65
v₀=1176, v₁=13158, v₋₁=1462, v₋₂=7020, v∞=780
```
Run the interpolation sequence: `r₀=1176`; `r₄=780`; `r₃=(7020−13158)/3=−2046`;
`r₁=(13158−1462)/2=5848`; `r₂=1462−1176=286`; `r₃=(286−(−2046))/2+2·780=1166+1560=2726`;
`r₂=286+5848−780=5354`; `r₁=5848−2726=3122`. So `(r₀,r₁,r₂,r₃,r₄)=(1176,3122,5354,2726,780)`.
```
x·y = 780·100⁴ + 2726·100³ + 5354·100² + 3122·100 + 1176
    = 78000000000 + 2726000000 + 53540000 + 312200 + 1176
    = 80779853376 = 123456·654321 .  ✓
```

## Code

```python
THRESHOLD = 3   # operands with at most a few base digits multiply directly

def exact_div(value, divisor):
    quotient, remainder = divmod(value, divisor)
    if remainder:
        raise ArithmeticError("interpolation division was not exact")
    return quotient

def toom3(x, y, base=10):
    if base <= 1:
        raise ValueError("base must be greater than 1")

    if x < 0 or y < 0:
        sign = -1 if (x < 0) ^ (y < 0) else 1
        return sign * toom3(abs(x), abs(y), base)

    # base case: small enough that the direct product is O(1)
    if x < base ** THRESHOLD or y < base ** THRESHOLD:
        return x * y

    # limb size m so each operand has at most 3 limbs in B = base**m
    n = max(len(str(x)), len(str(y)))
    m = n // 3 + 1
    B = base ** m

    # cut into 3 limbs (the polynomials p, q); integer floor-div/remainder only
    x0, x1, x2 = x % B, (x // B) % B, x // (B * B)
    y0, y1, y2 = y % B, (y // B) % B, y // (B * B)

    # evaluate p, q at 0, 1, -1, -2, inf  (adds + small shifts, no multiply)
    px1, py1 = x0 + x1 + x2,        y0 + y1 + y2
    pxm1, pym1 = x0 - x1 + x2,      y0 - y1 + y2
    pxm2, pym2 = x0 - 2*x1 + 4*x2,  y0 - 2*y1 + 4*y2

    # the FIVE recursive multiplications (5 instead of 9)
    v0   = toom3(x0,   y0,   base)
    v1   = toom3(px1,  py1,  base)
    vm1  = toom3(pxm1, pym1, base)
    vm2  = toom3(pxm2, pym2, base)
    vinf = toom3(x2,   y2,   base)

    # interpolation (exact: r = p*q has integer coefficients)
    r0 = v0
    r4 = vinf
    r3 = exact_div(vm2 - v1, 3)
    r1 = exact_div(v1 - vm1, 2)
    r2 = vm1 - v0
    r3 = exact_div(r2 - r3, 2) + 2 * r4
    r2 = r2 + r1 - r4
    r1 = r1 - r3
    coeffs = [r0, r1, r2, r3, r4]

    # recompose r(B) = sum r_i * B^i  (shifts + adds, O(n))
    result = 0
    for c in reversed(coeffs):
        result = result * B + c
    return result


if __name__ == "__main__":
    import random
    for _ in range(5000):
        a = random.randint(-10 ** random.randint(0, 40), 10 ** random.randint(0, 40))
        b = random.randint(-10 ** random.randint(0, 40), 10 ** random.randint(0, 40))
        assert toom3(a, b) == a * b
    assert toom3(123456, 654321) == 80779853376
    print("all toom3 tests passed")
```
