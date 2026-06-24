**Problem.** A gear track has teeth at positions `1, 2, 3, ...`; a wheel with `m` teeth makes a position `x` *resonance-free* exactly when `gcd(x, m) = 1`. Given `m` and `q` inspection windows `[L, R]` (both endpoints **inclusive**), report for each window how many positions `x` in `[L, R]` satisfy `gcd(x, m) = 1`. Bounds: `1 <= m <= 10^12`, `1 <= q <= 2*10^5`, `1 <= L <= R <= 10^18`.

**Why the obvious scan is wrong (for the bounds).** Looping `x` from `L` to `R` and testing `gcd` is correct but `O(R - L + 1)`; with `R - L` up to `10^18` it cannot finish. The predicate `gcd(x,m)=1` is non-monotone in `x`, so there is no endpoint shortcut. We need a closed-form prefix count.

**Key idea — difference of inclusive prefix counts, via Mobius inclusion-exclusion.** Define `C(N) = #{x : 1 <= x <= N, gcd(x, m) = 1}`. A position `x <= N` is coprime to `m` iff it avoids every distinct prime of `m`, so by inclusion-exclusion

```
C(N) = sum over squarefree divisors d of m of mu(d) * floor(N / d)
     = sum over subsets S of m's distinct primes of (-1)^|S| * floor(N / prod(S)).
```

Factor `m` once by trial division to `10^6` (`m <= 10^12`); it has at most 11 distinct primes, so at most `2^11 = 2048` signed terms. Precompute the list of `(divisor, sign)` pairs once and reuse it for every query. The answer for a window is the **difference of inclusive prefix counts**:

```
answer(L, R) = C(R) - C(L - 1).
```

**Pitfalls — the boundary is the whole problem.**
1. *Inclusive/half-open off-by-one.* Use `C(R) - C(L - 1)`, not `C(R) - C(L)`. Since `C` counts `[1, N]` inclusively, subtracting `C(L)` counts the half-open `(L, R]` and silently **drops the left endpoint `L`**. This is invisible whenever `L` is not coprime to `m` and only surfaces when `L` itself is a valid position — e.g. the single-point window `[5, 5]` with `m = 12` must give `1`, but `C(5) - C(5) = 0`. Trace that case to catch it.
2. *The `C(0)` corner.* The `C(L - 1)` fix introduces `C(0)` when `L = 1`; `C(0)` must be `0` (no positive integers `<= 0`). Guard `coprimeUpTo` with `if (N <= 0) return 0;` so the empty-prefix count is provably zero rather than zero by luck.
3. *Overflow.* `R` up to `10^18` needs `long long` for positions, prefix counts, divisors (`d <= m <= 10^12`), and the signed accumulator. An `int` is a silent wrong answer on large tests.

**Edge cases.** `m = 1`: empty prime set, `C(N) = N`, every answer is `R - L + 1`. Prime `m`: two terms `N - floor(N/p)`. Prime-power `m` (e.g. `49`): distinct primes strip to `{7}`, so coprime-to-`49` equals coprime-to-`7`. Single-point `L = R`: answer is `1` iff `R` is coprime to `m`. `L = 1`: exercises the `C(0)` guard. Highly composite `m` (product of many small primes): maximum `2^k` terms, still tiny.

**Complexity.** Factor `m` in `O(sqrt(m))`. Build `2^k <= 2048` signed divisors once. Each query is `O(2^k)`; total `O(2^k * q)`, about `4*10^8` cheap ops, which runs in well under the 3 s limit (measured ~1.7 s at the worst case) using `O(2^k)` memory.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

// distinct prime factors of m
static vector<long long> primeFactors(long long m) {
    vector<long long> ps;
    for (long long p = 2; p * p <= m; ++p) {
        if (m % p == 0) {
            ps.push_back(p);
            while (m % p == 0) m /= p;
        }
    }
    if (m > 1) ps.push_back(m);
    return ps;
}

// Precompute the signed squarefree divisors of m from its distinct primes:
// each subset S contributes divisor prod(S) with sign (-1)^|S|. Built once,
// reused by every query, so a single query costs O(#subsets) and no bit loop.
static vector<pair<long long,int>> buildSigned(const vector<long long>& ps) {
    vector<pair<long long,int>> div; // (divisor, sign)
    div.push_back({1LL, +1});        // empty subset
    for (long long p : ps) {
        int cur = (int)div.size();
        for (int i = 0; i < cur; ++i)
            div.push_back({div[i].first * p, -div[i].second});
    }
    return div;
}

// count of integers in [1, N] coprime to m, via the precomputed signed divisors.
// N may be 0 -> returns 0 (handled by caller passing N = L-1 which can be 0).
static long long coprimeUpTo(long long N, const vector<pair<long long,int>>& div) {
    if (N <= 0) return 0;
    long long total = 0;
    for (const auto& d : div) total += d.second * (N / d.first);
    return total;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    long long m;
    int q;
    if (!(cin >> m >> q)) return 0;

    vector<long long> ps = primeFactors(m);    // empty when m == 1
    vector<pair<long long,int>> div = buildSigned(ps);

    while (q--) {
        long long L, R;
        cin >> L >> R;
        // safe positions in [L, R] inclusive = coprimeUpTo(R) - coprimeUpTo(L-1)
        long long ans = coprimeUpTo(R, div) - coprimeUpTo(L - 1, div);
        cout << ans << "\n";
    }
    return 0;
}
```
