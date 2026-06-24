**Problem.** Cover a `1 x n` strip with bricks of length `1` and `2` laid end to end; paint each brick one of `K` colors so that any two touching bricks differ in color. Count the distinct (length, color) designs modulo `p`; the empty strip (`n = 0`) is one design. Read `n K p` from stdin, print the count mod `p`.

**Why the naive enumeration is too slow.** Listing every length-`1`/`2` layout and coloring each as `K * (K-1)^{t-1}` is correct but the number of layouts is `Fib(n+1)`, exponential in `n`. It is a fine *oracle* for tiny `n`, useless for `n = 2*10^5`. We need a linear recurrence.

**Key idea — a Fibonacci tiling DP with a non-uniform color factor.** Let `g[i]` be the number of valid colored designs of a length-`i` strip. A design's last brick is length `1` (sitting on a length-`i-1` strip) or length `2` (on a length-`i-2` strip), and these are disjoint, so `g[i]` is the sum of two contributions. Coloring the new last brick multiplies a factor onto the predecessor's count, and the factor is **not uniform**:

- factor `K` if the predecessor strip is **empty** (length `0`) — the new brick is the first brick, no left neighbour to avoid;
- factor `K - 1` otherwise — the new brick must differ from the one it touches.

All `K` colors are symmetric, so this scalar factor is independent of the predecessor's actual colors, which is what lets a single `g[i]` work without tracking the last color. With `g[0] = 1`:

- `c1 = (i-1 == 0) ? K : K-1`, contribution `c1 * g[i-1]` (length-`1` brick);
- if `i >= 2`, `c2 = (i-2 == 0) ? K : K-1`, contribution `c2 * g[i-2]` (length-`2` brick).

Answer: `g[n] mod p`.

**Correctness.** Each design has a unique last brick (the one covering position `n`), of a unique length, so the decomposition "design = (strip before last brick) + (last brick)" is a bijection — every design is counted exactly once, no double-count. The color factor is exact: the leading brick genuinely has `K` choices and every later brick exactly `K - 1` (any color but its left neighbour's). Hand check `n = 3, K = 3`: `g[1]=3, g[2]=2*3+3*1=9, g[3]=2*9+2*3=24`, matching `12+6+6=24` from direct coloring of layouts `1+1+1, 1+2, 2+1`.

**Pitfalls.**
1. *Off-by-one at the first brick.* A uniform `K-1` factor undercounts: the leading brick has `K` choices, not `K-1`. (`n=1` then prints `2` instead of `3`.)
2. *The length-2 first brick.* The `K` factor must be chosen per branch from *that branch's own predecessor*, not by special-casing `i == 1`. The length-`2` brick at `i = 2` sits on the empty strip and also deserves `K`; only-fixing `i==1` makes `n=2, K=3` print `8` instead of `9`.
3. *Overflow.* Products are `(<p) * (<p) <= 10^18`; use `long long` and reduce mod `p` after every multiply, with factors `Km, Km1` reduced once up front.

**Edge cases (all handled by the recurrence, no special cases):** `n = 0` -> `g[0] = 1 % p`; `n = 1` -> `K mod p`; `K = 1` -> `Km1 = 0` collapses every length-`>=3` strip to `0` (two touching same-color bricks are forced); `p = 1` -> every answer `0`.

**Complexity.** `O(n)` time, `O(n)` space (reducible to `O(1)`).

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    long long n, K, MOD;
    if (!(cin >> n >> K >> MOD)) return 0;

    // g[i] = number of valid colored tilings of a 1 x i strip using tiles of
    // length 1 and 2, each tile one of K colors, no two ADJACENT tiles equal.
    // Extending a tiling by one tile: K choices if it is the first tile
    // (predecessor is the empty strip), else K-1 choices (avoid neighbour).
    // g[0] = 1 (empty strip). For i >= 1, a length-1 tile sits on g[i-1],
    // a length-2 tile sits on g[i-2]; the K-vs-(K-1) factor depends only on
    // whether that predecessor strip is empty (length 0) or not.
    vector<long long> g(max<long long>(n + 1, 1));
    g[0] = 1 % MOD;
    long long Km = K % MOD, Km1 = ((K - 1) % MOD + MOD) % MOD;
    for (long long i = 1; i <= n; i++) {
        long long total = 0;
        // add a length-1 tile onto a strip of length i-1
        long long c1 = (i - 1 == 0) ? Km : Km1;     // first tile -> K, else K-1
        total = (total + c1 * g[i - 1]) % MOD;
        // add a length-2 tile onto a strip of length i-2 (needs i >= 2)
        if (i >= 2) {
            long long c2 = (i - 2 == 0) ? Km : Km1;  // first tile -> K, else K-1
            total = (total + c2 * g[i - 2]) % MOD;
        }
        g[i] = total;
    }

    cout << g[n] % MOD << "\n";
    return 0;
}
```
