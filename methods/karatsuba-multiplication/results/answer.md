# Karatsuba Multiplication — Breaking the n² Barrier

## Problem

Multiply two `n`-digit integers exactly. Schoolbook long multiplication forms a separate product for each digit-pair and costs `Θ(n²)` single-digit multiplications, and `n²` was widely conjectured to be a genuine lower bound. Karatsuba multiplication does it in `o(n²)` — specifically `Θ(n^{log₂ 3}) ≈ Θ(n^{1.585})` — by a divide-and-conquer split that needs only **three** half-size multiplications per level instead of four.

## Key idea

Split each operand at a `Bᵐ` boundary (`m ≈ n/2`):
```
a = a₁·Bᵐ + a₂ ,   b = b₁·Bᵐ + b₂ .
```
The product expands as
```
a·b = a₁b₁·B²ᵐ + (a₁b₂ + a₂b₁)·Bᵐ + a₂b₂ .
```
The naive reading needs four sub-multiplications (`a₁b₁`, `a₁b₂`, `a₂b₁`, `a₂b₂`), giving `T(n) = 4T(n/2) + O(n) = Θ(n²)` — no gain over schoolbook, because the exponent is `log₂ 4 = 2`.

The leap: the middle coefficient needs only the **sum** `a₁b₂ + a₂b₁`, not the two cross products separately. That sum is hidden inside the single product of the half-sums,
```
(a₁ + a₂)(b₁ + b₂) = a₁b₁ + (a₁b₂ + a₂b₁) + a₂b₂ ,
```
so the cross sum is recovered by subtracting the two corner products that are already being computed:
```
a₁b₂ + a₂b₁ = (a₁ + a₂)(b₁ + b₂) − a₁b₁ − a₂b₂ .
```

## Algorithm

Three multiplications per split:
```
z₂ = a₁·b₁
z₀ = a₂·b₂
z₁ = (a₁ + a₂)(b₁ + b₂) − z₂ − z₀     (= a₁b₂ + a₂b₁)

a·b = z₂·B²ᵐ + z₁·Bᵐ + z₀ .
```
Everything except the three recursive multiplications (the additions, subtractions, and shifts by powers of `B`) is `O(n)`.

## Squaring form

Because `a·b = ¼[(a+b)² − (a−b)²]`, the method can equivalently be stated for squaring a single `n`-digit number `a = aₕ·Bᵐ + aₗ`, using `2aₕaₗ = (aₕ+aₗ)² − aₕ² − aₗ²`:
```
a² = aₕ²·B²ᵐ + [(aₕ+aₗ)² − aₕ² − aₗ²]·Bᵐ + aₗ² ,
```
three squarings of `m`-digit numbers — `aₕ²`, `aₗ²`, `(aₕ+aₗ)²`. The sum `aₕ+aₗ` can be `(m+1)`-digit; writing
```
aₕ+aₗ = 2a₃ + ε,    ε ∈ {0,1}
(aₕ+aₗ)² = 4a₃² + 4a₃ε + ε²
```
reduces it back to an `m`-digit square plus shifts and adds, so the recursion halves cleanly.

## Complexity

```
T(n) = 3·T(n/2) + O(n).
```
The recursion tree has `log₂ n` levels; level `i` has `3ⁱ` nodes doing `O(n/2ⁱ)` work, contributing `O((3/2)ⁱ·n)`. The branching factor `3/2 > 1` means the leaves dominate: there are `3^{log₂ n} = n^{log₂ 3}` leaves, so
```
T(n) = Θ(n^{log₂ 3}) ≈ Θ(n^{1.585}),
```
strictly sub-quadratic. (In master-theorem terms, `a = 3`, `b = 2`, combine exponent `c = 1 < log₂ 3`, so the leaf term `n^{log_b a}` wins.)

## Worked example

`1234 × 4321`, `B = 10`, `m = 2`: `a₁=12, a₂=34, b₁=43, b₂=21`.
```
z₂ = 12·43 = 516
z₀ = 34·21 = 714
(a₁+a₂)(b₁+b₂) = (12+34)(43+21) = 46·64 = 2944
z₁ = 2944 − 516 − 714 = 1714
a·b = 516·10⁴ + 1714·10² + 714 = 5 332 114.   ✓
```

## Code

```python
BASE = 10

def karatsuba(x, y):
    # base case: a single-digit operand — multiply directly (O(1))
    if x < BASE or y < BASE:
        return x * y

    # split point: half the digit-length of the longer operand
    n = max(len(str(x)), len(str(y)))
    m = n // 2
    split = BASE ** m

    # cut each number into high/low halves at the B^m boundary.
    # integer floor-division + remainder (divmod) — true division would
    # turn the operands into floats and never reach the base case.
    high1, low1 = divmod(x, split)   # x = high1 * 10^m + low1
    high2, low2 = divmod(y, split)   # y = high2 * 10^m + low2

    # the THREE recursive multiplications
    z2 = karatsuba(high1, high2)                # a1 * b1
    z0 = karatsuba(low1, low2)                  # a2 * b2
    z3 = karatsuba(high1 + low1, high2 + low2)  # (a1+a2)(b1+b2)

    # middle coefficient: cross sum recovered from the product-of-sums
    # minus the two corner products already computed
    z1 = z3 - z2 - z0                           # = a1*b2 + a2*b1

    # recombine: x*y = z2 * B^(2m) + z1 * B^m + z0   (shifts + adds, O(n))
    return z2 * BASE ** (2 * m) + z1 * BASE ** m + z0
```

Quick check:
```python
assert karatsuba(1234, 4321) == 1234 * 4321 == 5332114
assert karatsuba(31415926, 27182818) == 31415926 * 27182818
```
