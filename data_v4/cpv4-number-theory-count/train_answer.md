**Problem.** For `T` queries, each an integer `N` (`1 <= N <= 10^7`), count how many **distinct rational values** `a/b` the grid `1 <= a, b <= N` produces, where `(a,b)` and `(c,d)` denote the same value iff `a*d = b*c`. The naive `N*N` pair count overcounts wildly because `1/2 = 2/4 = 3/6` collapse. Read `T` then the `T` values from stdin; print one count per query. Example: `N = 3` gives `7` (the values `{1/3, 1/2, 2/3, 1/1, 3/2, 2/1, 3/1}`).

**Key idea — reduce to coprime pairs, then count with the summatory totient.** Every pair divides by `g = gcd(a,b)` to a unique lowest-terms representative `(p,q)` with `gcd(p,q)=1`, and two pairs are equal as values iff they share this representative. A reduced `(p,q)` with `1 <= p,q <= N` is itself in the grid, and every grid value reduces to such a pair, so the distinct values are **exactly** the coprime pairs in the `N x N` box:

```
C(N) = #{ (p, q) : 1 <= p, q <= N, gcd(p, q) = 1 }.
```

Split the square by the relation of `p` and `q`. The diagonal `p = q` is coprime only at `(1,1)` (one pair). Below the diagonal, for each `q` the coprime `p` in `[1, q-1]` number `phi(q)`, giving `sum_{q=2}^{N} phi(q)`; the above region is its mirror image with the same count. Hence, with `Phi(N) = sum_{q=1}^{N} phi(q)` (and `phi(1)=1`):

```
C(N) = 1 + 2 * sum_{q=2}^{N} phi(q) = 1 + 2*(Phi(N) - 1) = 2 * Phi(N) - 1.
```

Compute `phi` for all `q <= maxN` with a linear (Euler) sieve in `O(maxN)`, prefix-sum it into `Phi`, and answer each query with `2*Phi(N) - 1`. Sanity checks: `N=1 -> 2*1-1 = 1`; `N=2 -> 2*2-1 = 3`; `N=3 -> 2*4-1 = 7`.

**Pitfalls.**
1. *Double-counting the self-mirrored diagonal value `1/1`.* The tempting closed form `2 * Phi(N)` is wrong: it treats every coprime pair as half of a distinct `{(p,q),(q,p)}` couple, but `(1,1)` is its own mirror — swapping `p` and `q` fixes it. `Phi(N)` already includes `phi(1)=1` (the diagonal contribution), so doubling counts `1/1` twice. The `-1` removes exactly that over-count. The bug is glaring at `N = 1`, where `2*Phi(1) = 2` instead of the true `1`.
2. *Missing `break` in the linear sieve.* The sieve's correctness rests on striking each composite exactly once, by its smallest prime factor; you must `break` the instant `i % p == 0`. Without it, composites get struck more than once through the wrong factorization branch, silently corrupting some `phi` values (it can pass small hand checks by coincidence). After the multiplicative branch, `break`.
3. *Overflow — two places.* The answer reaches `~6.08 * 10^13` at `N = 10^7`, far past 32-bit; use `long long` for `Phi` and the answer. Also inside the sieve, compute `i * p` as `long long` before the `> maxN` guard, or `(int)i*p` can wrap negative, pass the bound, and write out of bounds.

**Edge cases.** `N = 1` -> `1` (only `1/1`; the case that exposes the off-by-one). `N = 2` -> `3` (`1/1, 1/2, 2/1`). Diagonal collisions `2/2, 3/3, ...` all collapse onto `1/1` and are counted once. `N = 10^7` -> `60792712854483`, needing 64-bit accumulators; the `int phi[]` plus `long long pref[]` arrays use ~120 MB and the run finishes in well under the 2 s limit. Empty input is guarded by `if (!(cin >> t)) return 0;`.

**Complexity.** `O(maxN)` time and memory for the single shared sieve and prefix sum, then `O(1)` per query; `T <= 5`.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int t;
    if (!(cin >> t)) return 0;
    vector<int> ns(t);
    int maxN = 1;
    for (int i = 0; i < t; i++) {
        cin >> ns[i];
        maxN = max(maxN, ns[i]);
    }

    // Linear sieve of Euler's totient up to maxN.
    vector<int> phi(maxN + 1);
    vector<int> primes;
    vector<char> isComp(maxN + 1, 0);
    phi[1] = 1;
    for (int i = 2; i <= maxN; i++) {
        if (!isComp[i]) {
            primes.push_back(i);
            phi[i] = i - 1;
        }
        for (int p : primes) {
            long long ip = (long long)i * p;
            if (ip > maxN) break;
            isComp[ip] = 1;
            if (i % p == 0) {
                phi[ip] = phi[i] * p;
                break;
            } else {
                phi[ip] = phi[i] * (p - 1);
            }
        }
    }

    // pref[k] = sum_{q=1}^{k} phi(q)
    vector<long long> pref(maxN + 1, 0);
    for (int i = 1; i <= maxN; i++) pref[i] = pref[i - 1] + phi[i];

    // Distinct value count = 2 * (sum_{q=1}^{N} phi(q)) - 1.
    // (1/1 counted once; each coprime p<q gives p/q and q/p.)
    for (int i = 0; i < t; i++) {
        int N = ns[i];
        long long ans = 2 * pref[N] - 1;
        cout << ans << "\n";
    }
    return 0;
}
```
