I propose the canonical name Schönhage–Strassen integer multiplication for this method. The goal is to multiply two very large integers exactly while using far fewer bit operations than the quadratic schoolbook approach, and to do it without any floating-point round-off that would jeopardize the last bit of the answer. The insight that makes this possible is that integer multiplication is really polynomial convolution in disguise. If I write an integer a as a sum of M-bit digits a_i times powers of 2^M, and do the same for b, then the product digits before carries are c_k equal to the sum over all i and j with i plus j equal to k of a_i times b_j. That is exactly the convolution of the two digit sequences. The remaining work, propagating carries, is only linear in the number of digits. So the hard part of multiplication is convolution, and the hard part of convolution can be diagonalized by a Fourier transform.

The convolution theorem tells me that the discrete Fourier transform turns convolution into pointwise multiplication. If I compute the DFT of each digit sequence, multiply the transformed values point by point, and apply the inverse DFT, I recover the convolution. The fast Fourier transform computes each DFT in N log N operations by recursively splitting even and odd indexed terms and recombining them with butterfly operations. The naive use of this idea runs the FFT over the complex numbers, where the root of unity is a transcendental complex exponential. That works in principle, but floating-point arithmetic introduces round-off in every twiddle factor, and guarding against that error requires extra precision and delicate analysis. I want exactness instead, so I need a setting where the root of unity is represented exactly.

The key observation is that the FFT does not actually need complex numbers. It only needs a commutative ring containing a primitive N-th root of unity omega, meaning omega raised to N is one and omega raised to r minus one is invertible for every r between zero and N, and it needs N itself to be invertible in the ring so that the inverse transform can divide by N. If I can find such a ring inside the integers modulo some carefully chosen number, every butterfly becomes an exact integer operation. The natural choice is a Fermat-type ring, the integers modulo 2^n plus one. In this ring, 2^n is congruent to minus one, so 2 raised to 2n is congruent to one, and 2 has order exactly 2n. If the transform length N divides 2n, then omega equal to 2 raised to 2n over N is a primitive N-th root of unity. Even better, omega is a power of two. Multiplying a residue by a power of two is just a bit shift, possibly followed by a reduction modulo 2^n plus one. That reduction is cheap because bits that spill past position n wrap back with a sign flip, exploiting the fact that 2^n is minus one. Every twiddle factor, every weight, and even the division by N in the inverse transform become shifts. The transform itself becomes multiplication-free.

There is still a subtlety. A plain length-N transform computes a cyclic convolution, which corresponds to multiplication modulo x^N minus one, meaning coefficient k plus N wraps around and adds to coefficient k. For recursive nesting it is more convenient to work modulo x^N plus one, the negacyclic convolution, because then the result at x equal to 2^M is a product modulo 2^{MN} plus one, which has the same Fermat shape as the ring itself. I can obtain the negacyclic convolution from the cyclic one by weighting the input digits. If I multiply digit i by theta raised to i before the transform, where theta equals 2 raised to n over N, then theta raised to N equals 2^n, which is minus one in the ring. The wraparound term at position k plus N picks up an extra factor of theta^N, which is minus one, so after unweighting by theta raised to minus k the wrapped contribution is subtracted rather than added. This converts the cyclic convolution into the negacyclic one. The weights and unweights are again powers of two, so they cost only shifts.

The algorithm now has a clean recursive shape. To multiply two numbers modulo 2^q plus one, I choose a transform length T and split each operand into T pieces of about half the ring width in bits. I weight the pieces, run the forward shift-only NTT, multiply the N resulting coefficient pairs pointwise modulo the current Fermat ring, run the inverse NTT, unweight, and reassemble with carries. The pointwise products are themselves multiplications modulo a smaller Fermat number, so they are smaller instances of the same problem. I recurse until the operands are small enough that a base-case multiplication is faster than setting up another transform. Choosing the inner size to be about the square root of the outer size makes the recursion depth logarithmic in the logarithm of the input size, because q shrinks to q^{1/2}, then q^{1/4}, and so on. Each level performs an N log N transform on numbers whose total bit length is proportional to the current input size, so each level costs order q log q. Multiplying by the logarithmic number of levels gives an overall bit complexity of order n log n log log n, all in exact integer arithmetic.

At the top level, where I want the full integer product rather than a residue, I zero-pad by filling only the lower half of the coefficient array and choosing the padded length so that the Fermat modulus exceeds the true product. The negacyclic wraparound then lands in zero coefficients and disappears, leaving the honest convolution followed by ordinary carry propagation. This is the complete Schönhage–Strassen pipeline: convolution through an exact number-theoretic transform over a Fermat ring, multiplication-free butterflies because the root of unity is a power of two, negacyclic weighting to make the subproblems self-similar, and recursion on those subproblems until they become small.

The following single-file C++17 program implements the core mechanism. It reads two integers a and b (decimal, with an optional leading minus) from standard input and writes their exact product to standard output. Since C++ has no native big integer, it carries a minimal magnitude type — base-2^32 limbs with shift, add, subtract, and a schoolbook base-case multiply — and on top of that implements arithmetic modulo 2^n plus one, a shift-based cyclic NTT, the negacyclic weighting that turns it into a Schönhage–Strassen style convolution, the recursive pointwise multiplication, and a top-level wrapper that zero-pads to obtain the exact full product. The implementation is not tuned for speed, but it faithfully executes the mathematical pipeline that gives the n log n log log n exact bound; compiling it with g++ -O2 -std=c++17 and checking it against ordinary big-integer products across many bit-lengths and sign combinations, including operands well past the recursion threshold, reproduces every product exactly.

```cpp
// Schoenhage-Strassen exact integer multiplication, single-file C++17.
// Reads two integers a and b (decimal, optional leading '-') from stdin,
// separated by whitespace/newline, and prints the exact product a*b to stdout.
//
// The product is computed by a Number-Theoretic Transform over the Fermat ring
// R = Z/(2^n+1)Z: there 2^n == -1, so 2 has order 2n and omega = 2^(2n/N),
// theta = 2^(n/N), 1/N are all powers of two => every twiddle/weight is a bit
// shift and the transform is multiplication-free.  Negacyclic weighting makes
// the N pointwise products smaller instances of the same "multiply mod 2^q+1"
// task, which we recurse on, giving O(n log n log log n) exact bit operations.
//
// A minimal big-integer type (base-2^32 limbs, with shifts/add/sub and a
// schoolbook base-case multiply) underpins the exact integer arithmetic.

#include <bits/stdc++.h>
using namespace std;

// ----------------------------- Big integer ---------------------------------
// Non-negative magnitude as base-2^32 little-endian limbs.  Sign handled at top.
struct Big {
    vector<uint32_t> d;                 // little-endian, no trailing zero limbs
    Big() {}
    Big(uint64_t v) { while (v) { d.push_back((uint32_t)v); v >>= 32; } }

    void trim() { while (!d.empty() && d.back() == 0) d.pop_back(); }
    bool isZero() const { return d.empty(); }
    size_t bitLength() const {
        if (d.empty()) return 0;
        size_t hi = d.size() - 1;
        int top = 0; uint32_t v = d[hi];
        while (v) { top++; v >>= 1; }
        return hi * 32 + top;
    }
};

static int cmp(const Big& a, const Big& b) {
    if (a.d.size() != b.d.size()) return a.d.size() < b.d.size() ? -1 : 1;
    for (size_t i = a.d.size(); i-- > 0;)
        if (a.d[i] != b.d[i]) return a.d[i] < b.d[i] ? -1 : 1;
    return 0;
}

static Big add(const Big& a, const Big& b) {
    Big r; r.d.resize(max(a.d.size(), b.d.size()) + 1, 0);
    uint64_t carry = 0;
    for (size_t i = 0; i < r.d.size(); i++) {
        uint64_t s = carry;
        if (i < a.d.size()) s += a.d[i];
        if (i < b.d.size()) s += b.d[i];
        r.d[i] = (uint32_t)s; carry = s >> 32;
    }
    r.trim(); return r;
}

// a - b, requires a >= b.
static Big sub(const Big& a, const Big& b) {
    Big r; r.d.resize(a.d.size(), 0);
    int64_t borrow = 0;
    for (size_t i = 0; i < a.d.size(); i++) {
        int64_t s = (int64_t)a.d[i] - borrow - (i < b.d.size() ? (int64_t)b.d[i] : 0);
        if (s < 0) { s += (int64_t)1 << 32; borrow = 1; } else borrow = 0;
        r.d[i] = (uint32_t)s;
    }
    r.trim(); return r;
}

// Left shift by 'bits'.
static Big shl(const Big& a, size_t bits) {
    if (a.isZero()) return Big();
    size_t limbShift = bits >> 5, bitShift = bits & 31;
    Big r; r.d.assign(a.d.size() + limbShift + 1, 0);
    for (size_t i = 0; i < a.d.size(); i++) {
        uint64_t v = (uint64_t)a.d[i] << bitShift;
        r.d[i + limbShift]     |= (uint32_t)v;
        r.d[i + limbShift + 1] |= (uint32_t)(v >> 32);
    }
    r.trim(); return r;
}

// Right shift by 'bits' (floor).
static Big shr(const Big& a, size_t bits) {
    size_t limbShift = bits >> 5, bitShift = bits & 31;
    if (limbShift >= a.d.size()) return Big();
    Big r; r.d.assign(a.d.size() - limbShift, 0);
    for (size_t i = 0; i < r.d.size(); i++) {
        uint64_t v = a.d[i + limbShift] >> bitShift;
        if (bitShift && i + limbShift + 1 < a.d.size())
            v |= (uint64_t)a.d[i + limbShift + 1] << (32 - bitShift);
        r.d[i] = (uint32_t)v;
    }
    r.trim(); return r;
}

// Low 'bits' bits of a (a mod 2^bits).
static Big lowBits(const Big& a, size_t bits) {
    size_t limbs = bits >> 5, extra = bits & 31;
    Big r;
    size_t take = min(a.d.size(), limbs + (extra ? 1 : 0));
    r.d.assign(a.d.begin(), a.d.begin() + take);
    if (extra && r.d.size() > limbs) r.d[limbs] &= (uint32_t)((1u << extra) - 1);
    r.trim(); return r;
}

// Schoolbook multiply (base case).
static Big mulSchool(const Big& a, const Big& b) {
    if (a.isZero() || b.isZero()) return Big();
    Big r; r.d.assign(a.d.size() + b.d.size(), 0);
    for (size_t i = 0; i < a.d.size(); i++) {
        uint64_t carry = 0, ai = a.d[i];
        for (size_t j = 0; j < b.d.size(); j++) {
            uint64_t cur = (uint64_t)r.d[i + j] + ai * b.d[j] + carry;
            r.d[i + j] = (uint32_t)cur; carry = cur >> 32;
        }
        size_t k = i + b.d.size();
        while (carry) { uint64_t cur = (uint64_t)r.d[k] + carry; r.d[k] = (uint32_t)cur; carry = cur >> 32; k++; }
    }
    r.trim(); return r;
}

// 2^n + 1 as a Big.
static Big fermat(size_t n) { Big r = shl(Big((uint64_t)1), n); return add(r, Big((uint64_t)1)); }

// a mod (2^n+1), for a >= 0.  Uses 2^n == -1: fold high blocks with alternating sign.
static Big reduceMod(const Big& a, size_t n) {
    Big mod = fermat(n);
    // Split a into n-bit blocks; block i contributes with sign (-1)^i since 2^n == -1.
    Big pos, neg;
    Big cur = a;
    int sign = 1;
    while (!cur.isZero()) {
        Big lo = lowBits(cur, n);
        if (sign > 0) pos = add(pos, lo); else neg = add(neg, lo);
        cur = shr(cur, n);
        sign = -sign;
    }
    // value = pos - neg (mod 2^n+1). Reduce each into [0, mod).
    auto modReduce = [&](Big x) -> Big {
        while (cmp(x, mod) >= 0) {
            Big lo = lowBits(x, n);
            Big hi = shr(x, n);
            // x = lo + 2^n*hi == lo - hi (mod). Compute lo - hi mod.
            if (cmp(lo, hi) >= 0) x = sub(lo, hi);
            else { x = sub(add(lo, mod), hi); }
        }
        return x;
    };
    pos = modReduce(pos);
    neg = modReduce(neg);
    Big res;
    if (cmp(pos, neg) >= 0) res = sub(pos, neg);
    else res = sub(add(pos, mod), neg);
    // final normalize
    while (cmp(res, mod) >= 0) res = sub(res, mod);
    return res;
}

// multiply by 2^exponent in R = Z/(2^n+1); exponent taken mod 2n (can be negative).
static Big shiftMod(const Big& x, long long exponent, size_t n) {
    long long m = (long long)(2 * n);
    long long e = exponent % m; if (e < 0) e += m;
    return reduceMod(shl(x, (size_t)e), n);
}

// ----------------------------- NTT over R ----------------------------------
// length-N cyclic NTT, omega = 2^(2n/N); every '* w^j' is a shift-and-fold.
static void ntt(vector<Big>& a, size_t n, bool inverse) {
    size_t N = a.size();
    // bit-reversal permutation
    size_t j = 0;
    for (size_t i = 1; i < N; i++) {
        size_t bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) swap(a[i], a[j]);
    }
    Big mod = fermat(n);
    for (size_t len = 2; len <= N; len <<= 1) {
        long long step = (long long)(2 * n) / (long long)len;
        if (inverse) step = -step;
        for (size_t start = 0; start < N; start += len) {
            long long tw = 0;
            for (size_t i = 0; i < len / 2; i++) {
                Big u = a[start + i];
                Big v = shiftMod(a[start + i + len / 2], tw, n);
                a[start + i]             = reduceMod(add(u, v), n);
                // u - v mod
                Big uv;
                if (cmp(u, v) >= 0) uv = sub(u, v);
                else uv = sub(add(u, mod), v);
                a[start + i + len / 2]   = reduceMod(uv, n);
                tw += step;
            }
        }
    }
    if (inverse) {
        // 1/N = 2^{-log2 N}
        size_t k = 0; while ((((size_t)1) << k) < N) k++;
        for (size_t i = 0; i < N; i++) a[i] = shiftMod(a[i], -(long long)k, n);
    }
}

// forward decls for recursion
static Big multiplyMod(const Big& a, const Big& b, size_t K, size_t M, size_t n);

static const size_t RECURSE_BITS = 1 << 14;

// pointwise product x*y mod (2^n+1), recursing for large n.
static Big pointwiseMul(const Big& x, const Big& y, size_t n) {
    if (n <= RECURSE_BITS) return reduceMod(mulSchool(x, y), n);
    Big two_n = shl(Big((uint64_t)1), n);
    if (cmp(x, two_n) == 0 || cmp(y, two_n) == 0) return reduceMod(mulSchool(x, y), n);
    size_t nb = 0; { size_t t = n; while (t) { nb++; t >>= 1; } } // bit length of n
    size_t k = max((size_t)2, nb / 2);
    size_t K = (size_t)1 << k;
    while (K > 2 && (n % K)) { k--; K = (size_t)1 << k; }
    if (n % K) return reduceMod(mulSchool(x, y), n);
    size_t M = n / K;
    size_t Kb = 0; { size_t t = K; while (t) { Kb++; t >>= 1; } } // bit length of K
    size_t np = 2 * M + Kb;
    np = ((np + K - 1) / K) * K;
    if (np >= n) return reduceMod(mulSchool(x, y), n);
    return reduceMod(multiplyMod(x, y, K, M, np), n);
}

// a*b mod (2^(K*M)+1): K=2^k pieces of M bits, ring 2^n+1; needs K|n, n>=2M+log2 K+1.
static Big multiplyMod(const Big& a_in, const Big& b_in, size_t K, size_t M, size_t n) {
    size_t out_n = K * M;
    Big out_mod = fermat(out_n);
    // reduce inputs mod 2^out_n + 1 (out_mod) so the residue '2^out_n' case is detectable
    auto modOut = [&](const Big& x) -> Big {
        Big v = x;
        while (cmp(v, out_mod) >= 0) {
            Big lo = lowBits(v, out_n);
            Big hi = shr(v, out_n);
            if (cmp(lo, hi) >= 0) v = sub(lo, hi);
            else v = sub(add(lo, out_mod), hi);
        }
        return v;
    };
    Big a = modOut(a_in), b = modOut(b_in);
    Big two_outn = shl(Big((uint64_t)1), out_n);
    if (cmp(a, two_outn) == 0 || cmp(b, two_outn) == 0) {
        return modOut(mulSchool(a, b));
    }
    vector<Big> A(K), B(K);
    for (size_t i = 0; i < K; i++) {
        A[i] = lowBits(shr(a, M * i), M);
        B[i] = lowBits(shr(b, M * i), M);
    }
    size_t theta_step = n / K;            // theta = 2^theta_step, theta^K = 2^n == -1
    for (size_t i = 0; i < K; i++) {
        A[i] = shiftMod(A[i], (long long)(theta_step * i), n);  // negacyclic weights
        B[i] = shiftMod(B[i], (long long)(theta_step * i), n);
    }
    ntt(A, n, false);
    ntt(B, n, false);
    vector<Big> C(K);
    for (size_t i = 0; i < K; i++) C[i] = pointwiseMul(A[i], B[i], n);  // recursive pointwise
    ntt(C, n, true);
    for (size_t i = 0; i < K; i++)
        C[i] = shiftMod(C[i], -(long long)(theta_step * i), n);         // unweight

    // reassembly: sum_i sign-adjusted C[i] << (M*i), as a signed value, then mod out_mod
    Big half = shl(Big((uint64_t)1), n - 1);  // 2^(n-1)
    Big mod_n = fermat(n);
    Big pos, neg;
    for (size_t i = 0; i < K; i++) {
        Big ci = C[i];
        bool negative = cmp(ci, half) > 0;     // coeff may be negative (read residue as signed)
        if (negative) {
            // ci -= (2^n+1)  =>  magnitude = mod_n - ci, contributes negatively
            Big mag = sub(mod_n, ci);
            neg = add(neg, shl(mag, M * i));
        } else {
            pos = add(pos, shl(ci, M * i));
        }
    }
    // value = pos - neg, reduced mod out_mod
    Big P = modOut(pos), Ng = modOut(neg);
    Big res;
    if (cmp(P, Ng) >= 0) res = sub(P, Ng);
    else res = sub(add(P, out_mod), Ng);
    return modOut(res);
}

// full product via zero-padded negacyclic transform (data fills lower K/2 pieces).
static Big multiplyMag(const Big& a, const Big& b) {
    if (a.isZero() || b.isZero()) return Big();
    size_t S = max(a.bitLength(), b.bitLength());
    // k = max(2, bitlen(bitlen(2S)) + 2)
    size_t twoS_bits = (2 * S);
    size_t blen = 0; { size_t t = twoS_bits; while (t) { blen++; t >>= 1; } } // bit length of 2S
    size_t blen2 = 0; { size_t t = blen; while (t) { blen2++; t >>= 1; } }    // bit length of that
    size_t k = max((size_t)2, blen2 + 2);
    size_t K = (size_t)1 << k;
    size_t M = (S + (K / 2) - 1) / (K / 2);
    size_t Kb = 0; { size_t t = K; while (t) { Kb++; t >>= 1; } } // bit length of K
    size_t np = 2 * M + Kb;
    np = ((np + K - 1) / K) * K;
    return multiplyMod(a, b, K, M, np);
}

// ----------------------------- decimal I/O ---------------------------------
static Big fromDecimal(const string& s) {
    Big r;
    for (char c : s) {
        if (c < '0' || c > '9') continue;
        // r = r*10 + (c-'0')
        Big t = mulSchool(r, Big((uint64_t)10));
        r = add(t, Big((uint64_t)(c - '0')));
    }
    return r;
}

static string toDecimal(Big a) {
    if (a.isZero()) return "0";
    // repeatedly divide by 1e9
    string out;
    const uint32_t BASE = 1000000000u;
    vector<uint32_t> chunks;
    while (!a.d.empty()) {
        uint64_t rem = 0;
        for (size_t i = a.d.size(); i-- > 0;) {
            uint64_t cur = (rem << 32) | a.d[i];
            a.d[i] = (uint32_t)(cur / BASE);
            rem = cur % BASE;
        }
        a.trim();
        chunks.push_back((uint32_t)rem);
    }
    char buf[16];
    out = to_string(chunks.back());
    for (size_t i = chunks.size() - 1; i-- > 0;) {
        snprintf(buf, sizeof(buf), "%09u", chunks[i]);
        out += buf;
    }
    return out;
}

int main() {
    string sa, sb;
    if (!(cin >> sa)) return 0;
    if (!(cin >> sb)) return 0;
    bool negA = !sa.empty() && sa[0] == '-';
    bool negB = !sb.empty() && sb[0] == '-';
    Big a = fromDecimal(sa);
    Big b = fromDecimal(sb);
    Big prod = multiplyMag(a, b);
    bool negR = (negA ^ negB) && !prod.isZero();
    if (negR) cout << '-';
    cout << toDecimal(prod) << '\n';
    return 0;
}
```

This program demonstrates why the Schönhage–Strassen algorithm is historically important: it was the first practical method to break the quadratic barrier for exact integer multiplication, reaching a worst-case bit complexity of order n log n log log n while keeping every arithmetic step exact. The modern view is that multiplication is convolution, convolution is diagonalized by Fourier transforms, and the FFT becomes both exact and cheap once the transform is performed over a ring where the root of unity is itself a shift, namely a power of two in the Fermat ring Z modulo 2^n plus one.
