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

Single self-contained C++17 program: it reads two non-negative big integers (whitespace-separated, arbitrary length) from `stdin` and prints their exact product to `stdout`. Operands are carried as little-endian base-10 digit vectors so the method works far past machine-word width; the recursion is the three-multiplication split, with the split point `m` taken as half the longer operand's digit length.

```cpp
// Karatsuba multiplication. Reads two non-negative big integers (whitespace-
// separated, arbitrary length) from stdin and prints their exact product.
#include <bits/stdc++.h>
using namespace std;

// A big number is held little-endian, one base-10 digit per vector slot.
using Big = vector<int>;          // digits[0] is the units place

static const int BASE = 10;

Big from_string(const string& s) {
    Big d;
    for (int i = (int)s.size() - 1; i >= 0; --i) d.push_back(s[i] - '0');
    if (d.empty()) d.push_back(0);
    return d;
}

void trim(Big& a) {                // drop leading (high-order) zeros
    while (a.size() > 1 && a.back() == 0) a.pop_back();
}

// add: a + b
Big add(const Big& a, const Big& b) {
    Big c;
    int carry = 0;
    for (size_t i = 0; i < a.size() || i < b.size() || carry; ++i) {
        int s = carry;
        if (i < a.size()) s += a[i];
        if (i < b.size()) s += b[i];
        c.push_back(s % BASE);
        carry = s / BASE;
    }
    return c;
}

// sub: a - b, assuming a >= b (used only where the algebra guarantees it)
Big sub(const Big& a, const Big& b) {
    Big c;
    int borrow = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        int s = a[i] - borrow - (i < b.size() ? b[i] : 0);
        if (s < 0) { s += BASE; borrow = 1; } else borrow = 0;
        c.push_back(s);
    }
    trim(c);
    return c;
}

// shift: multiply by BASE^k (append k low-order zeros)
Big shift(const Big& a, size_t k) {
    if (a.size() == 1 && a[0] == 0) return a;   // 0 stays 0
    Big c(k, 0);
    c.insert(c.end(), a.begin(), a.end());
    return c;
}

bool is_zero(const Big& a) { return a.size() == 1 && a[0] == 0; }

// Karatsuba: three half-size multiplications instead of four.
//   z2 = x1*y1,  z0 = x2*y2,  z1 = (x1+x2)(y1+y2) - z2 - z0
//   x*y = z2*B^(2m) + z1*B^m + z0
Big karatsuba(const Big& x, const Big& y) {
    // base case: a single-digit operand -> multiply digit-by-number, O(len)
    if (x.size() == 1 || y.size() == 1) {
        long long mul = (x.size() == 1) ? x[0] : y[0];
        const Big& big = (x.size() == 1) ? y : x;
        Big c;
        long long carry = 0;
        for (size_t i = 0; i < big.size(); ++i) {
            long long s = (long long)big[i] * mul + carry;
            c.push_back((int)(s % BASE));
            carry = s / BASE;
        }
        while (carry) { c.push_back((int)(carry % BASE)); carry /= BASE; }
        if (c.empty()) c.push_back(0);
        trim(c);
        return c;
    }

    // split both operands at m = half the length of the longer one
    size_t m = max(x.size(), y.size()) / 2;
    size_t mx = min(m, x.size()), my = min(m, y.size());

    Big x2(x.begin(), x.begin() + mx);   // low  half of x
    Big x1(x.begin() + mx, x.end());     // high half of x
    Big y2(y.begin(), y.begin() + my);   // low  half of y
    Big y1(y.begin() + my, y.end());     // high half of y
    if (x1.empty()) x1.push_back(0);
    if (x2.empty()) x2.push_back(0);
    if (y1.empty()) y1.push_back(0);
    if (y2.empty()) y2.push_back(0);
    trim(x1); trim(x2); trim(y1); trim(y2);

    Big z2 = karatsuba(x1, y1);                       // a1*b1   (high)
    Big z0 = karatsuba(x2, y2);                       // a2*b2   (low)
    Big z3 = karatsuba(add(x1, x2), add(y1, y2));     // (a1+a2)(b1+b2)
    Big z1 = sub(sub(z3, z2), z0);                    // = a1*b2 + a2*b1

    // recombine: z2*B^(2m) + z1*B^m + z0
    Big result = add(add(shift(z2, 2 * m), shift(z1, m)), z0);
    trim(result);
    return result;
}

string to_string_big(const Big& a) {
    string s;
    for (int i = (int)a.size() - 1; i >= 0; --i) s += char('0' + a[i]);
    return s;
}

int main() {
    string sa, sb;
    if (!(cin >> sa >> sb)) return 0;
    Big a = from_string(sa), b = from_string(sb);
    trim(a); trim(b);
    Big prod = (is_zero(a) || is_zero(b)) ? Big{0} : karatsuba(a, b);
    cout << to_string_big(prod) << "\n";
    return 0;
}
```

Quick check: `1234 4321` on stdin prints `5332114`, and `31415926 27182818` prints `853973398759468` (`= 31415926 × 27182818`); a sweep of thousands of random pairs up to several hundred digits each agrees with the exact product.
