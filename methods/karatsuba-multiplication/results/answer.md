# Karatsuba Multiplication — Breaking the n² Barrier

## Problem

Multiply two `n`-digit integers exactly. Schoolbook long multiplication forms a separate product for each digit-pair and costs `Θ(n²)` single-digit multiplications. Karatsuba multiplication does it in `o(n²)` — specifically `Θ(n^{log₂ 3}) ≈ Θ(n^{1.585})` — by a divide-and-conquer split that needs only **three** half-size multiplications per level instead of four.

## Key idea

Split each operand at a `Bᵐ` boundary (`m ≈ n/2`):
```
x = x₁·Bᵐ + x₀ ,   y = y₁·Bᵐ + y₀ .
```
The product expands as
```
x·y = x₁y₁·B²ᵐ + (x₁y₀ + x₀y₁)·Bᵐ + x₀y₀ .
```
The naive reading needs four sub-multiplications (`x₁y₁`, `x₁y₀`, `x₀y₁`, `x₀y₀`), which gives `T(n) = 4T(n/2) + O(n) = Θ(n²)` — no gain over schoolbook.

The leap: the middle coefficient needs only the **sum** `x₁y₀ + x₀y₁`, not the two cross products separately. That sum is hidden inside the single product of the half-sums:
```
(x₁ + x₀)(y₁ + y₀) = x₁y₁ + (x₁y₀ + x₀y₁) + x₀y₀ ,
```
so the cross sum is recovered by subtracting the two corner products that are already being computed:
```
x₁y₀ + x₀y₁ = (x₁ + x₀)(y₁ + y₀) − x₁y₁ − x₀y₀ .
```

## Algorithm

Three multiplications per split:
```
z₀ = x₀·y₀
z₂ = x₁·y₁
z₃ = (x₁ + x₀)(y₁ + y₀)
z₁ = z₃ − z₂ − z₀          (= x₁y₀ + x₀y₁)

x·y = z₂·B²ᵐ + z₁·Bᵐ + z₀ .
```
Everything except the three recursive multiplications (the additions, subtractions, and shifts by powers of `B`) is `O(n)`.

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

`1234 × 4321`, `B = 10`, `m = 2`: `x₁=12, x₀=34, y₁=43, y₀=21`.
```
z₂ = 12·43 = 516
z₀ = 34·21 = 714
z₃ = (12+34)(43+21) = 46·64 = 2944
z₁ = 2944 − 516 − 714 = 1714
x·y = 516·10⁴ + 1714·10² + 714 = 5 332 114.   ✓
```

## Code

```python
def karatsuba(x, y):
    # base case: a single-digit operand — multiply directly (O(1))
    if x < 10 or y < 10:
        return x * y

    # split point: half the digit-length of the longer operand
    n = max(len(str(x)), len(str(y)))
    m = n // 2
    split = 10 ** m

    # cut each number into high/low halves at the B^m boundary.
    # integer floor-division + remainder (divmod) — true division would
    # turn the operands into floats and never reach the base case.
    high1, low1 = divmod(x, split)   # x = high1 * 10^m + low1
    high2, low2 = divmod(y, split)   # y = high2 * 10^m + low2

    # the THREE recursive multiplications
    z0 = karatsuba(low1, low2)                  # x0 * y0
    z2 = karatsuba(high1, high2)                # x1 * y1
    z3 = karatsuba(high1 + low1, high2 + low2)  # (x1+x0)(y1+y0)

    # middle coefficient: cross sum recovered from the product-of-sums
    # minus the two corner products already computed
    z1 = z3 - z2 - z0                           # = x1*y0 + x0*y1

    # recombine: x*y = z2 * B^(2m) + z1 * B^m + z0   (shifts + adds, O(n))
    return z2 * 10 ** (2 * m) + z1 * 10 ** m + z0
```

Quick check:
```python
assert karatsuba(1234, 4321) == 1234 * 4321 == 5332114
assert karatsuba(31415926, 27182818) == 31415926 * 27182818
```
