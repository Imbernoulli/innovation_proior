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

Single-file C++17 program: reads two (possibly signed, arbitrarily long) integers
`x` and `y` from stdin (whitespace-separated) and prints their exact product `x*y`
to stdout. The big-integer arithmetic is self-contained (base `10^9` limbs); the
limb base `B = (10^9)^m` is realized by limb-window slicing and shifts.

```cpp
// Toom-3 (Toom-Cook, k=3) recursive big-integer multiplication.
// Reads two (possibly signed, arbitrarily long) integers x and y, one per line
// (or whitespace-separated) from stdin, prints their exact product x*y to stdout.
//
// Split each operand into 3 base-B limbs -> degree-2 polynomials p,q with
// p(B)=x, q(B)=y. The product r=p*q has degree 4, so it is fixed by its values
// at 5 points {0,1,-1,-2,inf}. Evaluate p,q there, multiply pointwise (5
// recursive products instead of 9), interpolate r0..r4, recompose r(B).
// T(n) = 5*T(n/3) + O(n) = Theta(n^{log_3 5}) ~ Theta(n^1.465).

#include <bits/stdc++.h>
using namespace std;

// Arbitrary-precision signed integer in base 10^9 (little-endian limbs).
struct Big {
    static const uint32_t BASE = 1000000000u; // 10^9
    int sign = 1;            // +1 or -1; zero has sign +1 and empty mag
    vector<uint32_t> mag;    // little-endian, no trailing zero limbs

    Big() {}
    Big(long long v) {
        if (v < 0) { sign = -1; v = -v; }
        while (v) { mag.push_back((uint32_t)(v % BASE)); v /= BASE; }
    }

    bool isZero() const { return mag.empty(); }
    void trim() { while (!mag.empty() && mag.back() == 0) mag.pop_back(); if (mag.empty()) sign = 1; }

    // unsigned compare of magnitudes: -1,0,1
    static int cmpMag(const Big& a, const Big& b) {
        if (a.mag.size() != b.mag.size()) return a.mag.size() < b.mag.size() ? -1 : 1;
        for (size_t i = a.mag.size(); i-- > 0;)
            if (a.mag[i] != b.mag[i]) return a.mag[i] < b.mag[i] ? -1 : 1;
        return 0;
    }
};

// magnitude add: |a|+|b|
static Big addMag(const Big& a, const Big& b) {
    Big r; r.mag.resize(max(a.mag.size(), b.mag.size()) + 1, 0);
    uint64_t carry = 0;
    for (size_t i = 0; i < r.mag.size(); i++) {
        uint64_t s = carry;
        if (i < a.mag.size()) s += a.mag[i];
        if (i < b.mag.size()) s += b.mag[i];
        r.mag[i] = (uint32_t)(s % Big::BASE);
        carry = s / Big::BASE;
    }
    r.trim();
    return r;
}

// magnitude subtract: |a|-|b|, requires |a| >= |b|
static Big subMag(const Big& a, const Big& b) {
    Big r; r.mag.resize(a.mag.size(), 0);
    int64_t borrow = 0;
    for (size_t i = 0; i < a.mag.size(); i++) {
        int64_t s = (int64_t)a.mag[i] - borrow - (i < b.mag.size() ? (int64_t)b.mag[i] : 0);
        if (s < 0) { s += Big::BASE; borrow = 1; } else borrow = 0;
        r.mag[i] = (uint32_t)s;
    }
    r.trim();
    return r;
}

static Big add(const Big& a, const Big& b);

// signed subtract a-b
static Big sub(const Big& a, const Big& b) {
    Big nb = b; nb.sign = -nb.sign;
    return add(a, nb);
}

// signed add
static Big add(const Big& a, const Big& b) {
    if (a.isZero()) return b;
    if (b.isZero()) return a;
    if (a.sign == b.sign) { Big r = addMag(a, b); r.sign = a.sign; r.trim(); return r; }
    int c = Big::cmpMag(a, b);
    if (c == 0) return Big();
    if (c > 0) { Big r = subMag(a, b); r.sign = a.sign; r.trim(); return r; }
    Big r = subMag(b, a); r.sign = b.sign; r.trim(); return r;
}

// multiply magnitude by small non-negative int (fits in uint32 factor)
static Big mulSmallMag(const Big& a, uint32_t k) {
    Big r;
    if (k == 0 || a.isZero()) return r;
    r.mag.resize(a.mag.size() + 1, 0);
    uint64_t carry = 0;
    for (size_t i = 0; i < a.mag.size(); i++) {
        uint64_t s = (uint64_t)a.mag[i] * k + carry;
        r.mag[i] = (uint32_t)(s % Big::BASE);
        carry = s / Big::BASE;
    }
    r.mag[a.mag.size()] = (uint32_t)carry;
    r.trim();
    return r;
}

// signed multiply by small signed int
static Big mulSmall(const Big& a, long long k) {
    if (k == 0) return Big();
    int s = (k < 0) ? -1 : 1;
    Big r = mulSmallMag(a, (uint32_t)llabs(k));
    r.sign = a.sign * s; r.trim();
    return r;
}

// exact division of signed Big by small positive int (remainder must be 0)
static Big divExact(const Big& a, uint32_t d) {
    Big r; r.mag.resize(a.mag.size(), 0);
    uint64_t rem = 0;
    for (size_t i = a.mag.size(); i-- > 0;) {
        uint64_t cur = rem * Big::BASE + a.mag[i];
        r.mag[i] = (uint32_t)(cur / d);
        rem = cur % d;
    }
    // rem must be 0 for an exact division (guaranteed by Toom-3 interpolation)
    r.sign = a.sign; r.trim();
    return r;
}

// shift left by `limbs` base-10^9 limbs (multiply by BASE^limbs)
static Big shiftLimbs(const Big& a, size_t limbs) {
    if (a.isZero()) return a;
    Big r; r.mag.assign(limbs, 0);
    r.mag.insert(r.mag.end(), a.mag.begin(), a.mag.end());
    r.sign = a.sign; r.trim();
    return r;
}

// extract limbs [lo, hi) (a window of base-10^9 limbs) as a non-negative Big
static Big limbWindow(const Big& a, size_t lo, size_t hi) {
    Big r;
    if (lo >= a.mag.size()) return r;
    hi = min(hi, a.mag.size());
    r.mag.assign(a.mag.begin() + lo, a.mag.begin() + hi);
    r.trim();
    return r;
}

// total decimal-ish size proxy: number of base-10^9 limbs
static size_t limbCount(const Big& a) { return a.mag.size(); }

// schoolbook multiply of two non-negative magnitudes
static Big mulSchool(const Big& a, const Big& b) {
    Big r;
    if (a.isZero() || b.isZero()) return r;
    r.mag.assign(a.mag.size() + b.mag.size(), 0);
    for (size_t i = 0; i < a.mag.size(); i++) {
        uint64_t carry = 0;
        uint64_t ai = a.mag[i];
        for (size_t j = 0; j < b.mag.size(); j++) {
            uint64_t cur = r.mag[i + j] + ai * b.mag[j] + carry;
            r.mag[i + j] = (uint32_t)(cur % Big::BASE);
            carry = cur / Big::BASE;
        }
        size_t k = i + b.mag.size();
        while (carry) { uint64_t cur = r.mag[k] + carry; r.mag[k] = (uint32_t)(cur % Big::BASE); carry = cur / Big::BASE; k++; }
    }
    r.trim();
    return r;
}

static const size_t THRESHOLD = 8; // few-limb operands multiply directly

// Toom-3 multiplication of two signed Bigs.
static Big toom3(const Big& X, const Big& Y) {
    // sign handling: recurse on absolute values, fix sign at the end
    int outSign = X.sign * Y.sign;
    Big x = X; x.sign = 1;
    Big y = Y; y.sign = 1;

    // base case: small operands -> direct schoolbook product (O(1) in limbs)
    if (limbCount(x) < THRESHOLD || limbCount(y) < THRESHOLD) {
        Big r = mulSchool(x, y); r.sign = (r.isZero() ? 1 : outSign); r.trim();
        return r;
    }

    // limb size m (in base-10^9 limbs) so each operand has at most 3 limbs in B = (10^9)^m
    size_t n = max(limbCount(x), limbCount(y));
    size_t m = n / 3 + 1;
    // B = BASE^m, represented implicitly via limb windows of width m

    // cut into 3 limbs (the polynomials p, q): x = xb*B^2 + xa*B + xc, with
    // limbs xc,xa,xb (low,mid,high) and yc,ya,yb likewise
    Big xc = limbWindow(x, 0, m), xa = limbWindow(x, m, 2 * m), xb = limbWindow(x, 2 * m, 3 * m);
    Big yc = limbWindow(y, 0, m), ya = limbWindow(y, m, 2 * m), yb = limbWindow(y, 2 * m, 3 * m);

    // evaluate p, q at 0, 1, -1, -2, inf  (adds and small shifts, no multiply)
    Big px1 = add(add(xc, xa), xb);
    Big py1 = add(add(yc, ya), yb);
    Big pxm1 = add(sub(xc, xa), xb);
    Big pym1 = add(sub(yc, ya), yb);
    Big pxm2 = add(sub(xc, mulSmall(xa, 2)), mulSmall(xb, 4));
    Big pym2 = add(sub(yc, mulSmall(ya, 2)), mulSmall(yb, 4));

    // the FIVE recursive multiplications (5 instead of 9):
    Big v0 = toom3(xc, yc);     // r(0)   = x0*y0
    Big v1 = toom3(px1, py1);   // r(1)
    Big vm1 = toom3(pxm1, pym1);// r(-1)
    Big vm2 = toom3(pxm2, pym2);// r(-2)
    Big vinf = toom3(xb, yb);   // r(inf) = x2*y2

    // interpolation: recover r0..r4 from the five values.
    // every division (/3, /2) is exact because r = p*q has integer coefficients.
    Big r0 = v0;
    Big r4 = vinf;
    Big r3 = divExact(sub(vm2, v1), 3);
    Big r1 = divExact(sub(v1, vm1), 2);
    Big r2 = sub(vm1, v0);
    r3 = add(divExact(sub(r2, r3), 2), mulSmall(r4, 2));
    r2 = sub(add(r2, r1), r4);
    r1 = sub(r1, r3);

    // recompose r(B) = sum r_i * B^i  (shifts by i*m limbs + adds, O(n));
    // negative / oversized coefficients are absorbed by big-integer carries/borrows
    Big result = r0;
    result = add(result, shiftLimbs(r1, m));
    result = add(result, shiftLimbs(r2, 2 * m));
    result = add(result, shiftLimbs(r3, 3 * m));
    result = add(result, shiftLimbs(r4, 4 * m));

    result.sign = (result.isZero() ? 1 : outSign);
    result.trim();
    return result;
}

// parse a decimal string (optional leading +/-) into a Big
static Big parseBig(const string& s) {
    Big r;
    size_t i = 0; int sign = 1;
    if (i < s.size() && (s[i] == '+' || s[i] == '-')) { if (s[i] == '-') sign = -1; i++; }
    // strip leading zeros for clean magnitude, keep at least value
    string digits = s.substr(i);
    // build by processing 9 decimal digits at a time from the most significant end
    // simpler: accumulate via repeated *10^9 chunks from the front
    size_t start = 0;
    while (start < digits.size() && digits[start] == '0') start++;
    digits = digits.substr(start);
    if (digits.empty()) return r; // zero
    // process from most significant: group into 9-digit chunks aligned to the right
    int firstLen = (int)(digits.size() % 9);
    if (firstLen == 0) firstLen = 9;
    size_t pos = 0;
    // start with first chunk
    auto chunkVal = [&](size_t p, int len) -> uint32_t {
        uint32_t v = 0;
        for (int k = 0; k < len; k++) v = v * 10 + (uint32_t)(digits[p + k] - '0');
        return v;
    };
    r = Big((long long)chunkVal(pos, firstLen));
    pos += firstLen;
    while (pos < digits.size()) {
        r = mulSmallMag(r, Big::BASE); // *10^9
        uint32_t v = chunkVal(pos, 9);
        r = addMag(r, Big((long long)v));
        pos += 9;
    }
    r.sign = (r.isZero() ? 1 : sign);
    return r;
}

// convert Big to decimal string
static string toString(const Big& a) {
    if (a.isZero()) return "0";
    string s;
    for (size_t i = 0; i < a.mag.size(); i++) {
        char buf[16];
        if (i + 1 == a.mag.size()) snprintf(buf, sizeof(buf), "%u", a.mag[i]);
        else snprintf(buf, sizeof(buf), "%09u", a.mag[i]);
        s = string(buf) + s;
    }
    if (a.sign < 0) s = "-" + s;
    return s;
}

int main() {
    string sx, sy;
    if (!(cin >> sx >> sy)) return 0;
    Big x = parseBig(sx);
    Big y = parseBig(sy);
    Big prod = toom3(x, y);
    cout << toString(prod) << "\n";
    return 0;
}
```
