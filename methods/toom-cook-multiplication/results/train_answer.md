We need to multiply two $n$-digit integers exactly, and the only thing we count is the asymptotic number of elementary digit operations as $n$ grows. Schoolbook multiplication forms every digit-against-digit partial product and costs $\Theta(n^2)$; because every pair of digits genuinely interacts in the answer, that quadratic cost felt forced for a long time. The half-split method of Karatsuba and Ofman (1962) broke it: cutting each operand at the middle, $x = x_1 B^m + x_0$ and $y = y_1 B^m + y_0$ with $m = n/2$, the obvious expansion wants four half-size products, but the middle coefficient needs only the *sum* $x_1 y_0 + x_0 y_1$, and that sum is recovered as $(x_1+x_0)(y_1+y_0) - x_1 y_1 - x_0 y_0$ from one extra product minus the two corners already computed. Three half-size products give $T(n) = 3T(n/2) + O(n) = \Theta(n^{\log_2 3}) \approx n^{1.585}$. The trouble is that $\log_2 3$ is a strange place to stop. It is nothing but $\log_2$ of the number of sub-products, where the base $2$ is the number of parts; the whole exponent is $\log_{(\text{parts})}(\text{sub-products})$. That begs us to turn the knob: cut into $k > 2$ parts and see whether the exponent falls. But the naive $k$-part split warns us not to be optimistic — cutting into $k$ limbs and computing the convolution coefficients $c_\ell = \sum_{i+j=\ell} x_i y_j$ directly costs one sub-product per pair $(i,j)$, i.e. $k^2$ sub-multiplications, so $T(n) = k^2 T(n/k) + O(n)$ and $\log_k(k^2) = 2$, right back to $\Theta(n^2)$. More parts by themselves buy nothing. Any win must come from doing the convolution in fewer than $k^2$ sub-products, the way the half-split cut four down to three.

I propose Toom–Cook multiplication: split each operand into $k$ limbs, but compute the limb convolution by evaluation, pointwise multiplication, and interpolation rather than by brute force. The structural observation that unlocks everything is that a number cut into limbs is a polynomial sampled at the limb base. Writing $B = b^m$ and
$$p(t) = x_{k-1}t^{k-1} + \cdots + x_1 t + x_0, \qquad q(t) = y_{k-1}t^{k-1} + \cdots + y_1 t + y_0,$$
we have $x = p(B)$ and $y = q(B)$, so $x\cdot y = p(B)\,q(B) = (p\cdot q)(B)$. Integer multiplication has become *polynomial* multiplication of two degree-$(k-1)$ polynomials, followed by a single evaluation of the product at $t = B$, which is just shifting each coefficient left by $\ell\cdot m$ digits and adding — $O(n)$, free. All the cost lives in finding the coefficients of $r := p\cdot q$. Now, a polynomial is pinned down not only by its coefficients but equally by its *values*: a polynomial of degree $d$ is determined by its values at any $d+1$ distinct points. Our product $r$ has degree $2k-2$, so it is fixed by its values at $2k-1$ distinct points, and the value at a point $s$ is
$$r(s) = p(s)\cdot q(s),$$
a single number times a single number. This is the load-bearing move: evaluate $p$ and $q$ at $2k-1$ points (cheap linear combinations of the limbs, adds and small shifts, $O(n)$), do *one* multiplication $p(s)q(s)$ per point to obtain $r(s)$, and then interpolate the $2k-1$ coefficients of $r$ back from its $2k-1$ values by solving the Vandermonde linear system the points define. The product-of-two-numbers structure of $r(s)=p(s)q(s)$ is exactly what makes each value cost only one multiply, so the convolution that needed $k^2$ sub-products now needs $2k-1$ of them, each on operands of size $n/k$. The recurrence becomes
$$T(n) = (2k-1)\,T(n/k) + O(n) \;\Rightarrow\; T(n) = \Theta\!\big(n^{\log_k(2k-1)}\big).$$
Since $2k-1 < k^2$ for every $k \ge 2$, the exponent is always below $2$, and as $k\to\infty$, $\log_k(2k-1) = 1 + \log_k 2 - o(1) \to 1$, so multiplication runs in $n^{1+\varepsilon}$ for any $\varepsilon>0$. This answers whether $1.585$ is special: it is not, it is merely the $k=2$ rung of a ladder. And the half-split is precisely that rung — with points $\{0, \infty, 1\}$, $r(0)=x_0 y_0$ is the constant coefficient, $r(\infty)$ (the leading-coefficient "value at infinity", the limit of $r(t)/t^d$) is $x_1 y_1$, and the middle coefficient comes out of interpolation as $r(1) - r(\infty) - r(0) = (x_1+x_0)(y_1+y_0) - x_1 y_1 - x_0 y_0$, exactly Karatsuba's trick. The "share one product across the cross terms" idea is just what interpolation does at the middle point, and the $\infty$ point earns its keep by handing over the leading coefficient with no combination to form.

The case I build concretely is $k=3$ (Toom-3), giving $\Theta(n^{\log_3 5}) \approx n^{1.465}$. Cut into three limbs, $p(t)=x_2 t^2 + x_1 t + x_0$ and $q$ likewise; the product $r$ has degree $4$ and five coefficients $r_0,\dots,r_4$, so I need five points. I take $\{0, 1, -1, -2, \infty\}$, and the choice matters. The evaluations $p(s)$ must be cheap and, crucially, must not inflate the size of the operands I recurse on. At $s=1$ I just add the limbs; at $s=-1$ the alternating signs separate the even and odd parts cleanly; at $s=-2$ I get $x_0 - 2x_1 + 4x_2$, where the $2$ and $4$ are small shifts. Keeping the magnitudes small holds $p(s)$ to about one limb ($n/3$ digits) so the recursive products stay the right size; a point like $s=4$ would make $16x_2$ and bloat the operands. The two freebies $0$ and $\infty$ cost no multiplication at all and read off $r_0$ and $r_4$ directly. So I evaluate
$$p(0)=x_0,\quad p(1)=x_0+x_1+x_2,\quad p(-1)=x_0-x_1+x_2,\quad p(-2)=x_0-2x_1+4x_2,\quad p(\infty)=x_2,$$
identically for $q$, and form the five pointwise products $v_0,v_1,v_{-1},v_{-2},v_\infty$ — the only real multiplications, the five I recurse on, in place of the nine of the convolution. Recovering the coefficients is a linear system whose matrix is the Vandermonde matrix of the points; its determinant $\prod_{i<j}(s_j-s_i)$ over the finite points is nonzero because the points are distinct, so $r$ is uniquely recovered. I solve it the cheap way, eliminating with the freebies first: $r_0=v_0$ and $r_4=v_\infty$ come immediately; subtracting those, the $+1$ and $-1$ rows add and subtract to separate even from odd parts, giving $r_2$ and the sum $r_1+r_3$; the fifth point $-2$ supplies the one extra equation needed to split $r_1$ from $r_3$. Concretely the ordered sequence is
$$r_0=v_0,\quad r_4=v_\infty,\quad r_3=\frac{v_{-2}-v_1}{3},\quad r_1=\frac{v_1-v_{-1}}{2},\quad r_2=v_{-1}-v_0,$$
$$r_3=\frac{r_2-r_3}{2}+2v_\infty,\quad r_2=r_2+r_1-v_\infty,\quad r_1=r_1-r_3,$$
reusing partial results so only a single $/3$ and two $/2$ divisions appear. Each division lands on an exact integer — this is essential, and it is not luck. Because $r=p\cdot q$ has integer coefficients, the Vandermonde system over $\mathbb{Z}$ has an integer solution, and the elimination can be run so every intermediate stays integral: each step subtracts evaluations of a monic polynomial $P$ at two points, and $P(s_i)-P(s_j) = (s_i-s_j)\,Q(s_i)$ for a monic $Q$ of degree one less, so the point-difference factors out cleanly with no remainder. Those differences are exactly the small numbers $3$ and $2$ that show up as divisors. (For limb polynomials reduced modulo $b$ the same holds provided $b$ is prime, so the ring is an integral domain and the system stays uniquely solvable; over the ordinary integers we are always safe.) Finally I recompose $x\cdot y = r_4 B^4 + r_3 B^3 + r_2 B^2 + r_1 B + r_0$ by shifts and adds; some $r_\ell$ can be negative or exceed $B$, but evaluating the polynomial at $B$ with full-precision integer arithmetic carries and borrows correctly, so no manual normalization is needed. The recursion bottoms out by stripping signs and recursing on absolute values (the $-1$ and $-2$ evaluations can be negative), with a base case that multiplies directly once an operand is small, and a limb size $m = \lfloor n/3 \rfloor + 1$ chosen via integer floor-division and remainder so the limbs are genuine integers and the operands actually shrink toward the base case.

Concretely, the deliverable is a single self-contained C++17 program that reads the two (possibly signed, arbitrarily long) integers `x` and `y` from stdin, whitespace-separated, and writes their exact product `x*y` to stdout. The big-integer arithmetic is carried in-house, base `10^9` limbs little-endian; the limb base `B = (10^9)^m` never has to be materialized — it is realized by slicing limb windows of width `m` and shifting by `i·m` limbs — so the split, the five evaluations, the five recursive products, the exact interpolation, and the shift-and-add recomposition all run directly on the limb arrays.

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
