**Problem.** For `q` queries, each giving `n a b`, count the integer widths `x` with `1 <= x <= n`
such that `a <= floor(n/x) <= b`. Constraints: `q <= 2*10^5`, `n, b <= 10^12`, `0 <= a <= b`. Print
one count per query. A per-query `O(n)` scan is far too slow, so each query must be answered in
`O(1)`.

**Key idea — turn the quotient band into an interval of widths.** `x -> floor(n/x)` is non-increasing
on `[1, n]`, so each one-sided quotient bound becomes a one-sided bound on `x`, and the valid `x` form
a single interval. The two endpoints have *opposite* inclusivity:

- `floor(n/x) >= a` iff `x <= floor(n/a)` (for `a >= 1`) — an **inclusive** upper cap `hi`. If
  `a <= 0`, the condition holds for every `x in [1, n]`, so `hi = n`.
- `floor(n/x) <= b` iff `n/x < b+1` iff `x > floor(n/(b+1))` — an **exclusive** lower bound
  `loExcl = floor(n/(b+1))`; valid `x` satisfy `x > loExcl`.

The valid widths are the integers in `( loExcl , hi ]`, so the answer is `max(0, hi - loExcl)`.

**Pitfalls.**
1. *The `b+1`, not `b`.* The lower side comes from the strict `n/x < b+1`, so the denominator is
   `b+1`. Writing `floor(n/b)` shifts the band by one and miscounts at the boundary. The single-value
   band `n=10, a=1, b=1` exposes it: correct is `floor(10/2)=5` giving `10-5=5`, but `floor(10/1)=10`
   gives `0`.
2. *Open vs closed.* The `a` side is inclusive (`x <= floor(n/a)`); the `b` side is exclusive
   (`x > floor(n/(b+1))`). The count of integers in `(loExcl, hi]` is `hi - loExcl` (no `+1`).
3. *Division by zero / `a = 0`.* When `a <= 0` do not compute `n/a`; set `hi = n` because every width
   already satisfies `floor(n/x) >= 0 >= a`.
4. *Overflow.* `n` and `b` reach `10^12` and `b+1` is formed, so use `long long`. `int` is a silent
   wrong answer.

**Edge cases.** `n = 1`; full band `[1, n]` -> `n` (since `floor(n/(n+1)) = 0` reaches `x = 1`); the
impossible `[0, 0]` -> `0` (the open lower bound makes `(n, n]` empty); a band entirely above the max
quotient, e.g. `n=6, [7,9]` -> `0` (`hi = floor(6/7) = 0`); degenerate `a > b` -> `0`; `b < 0` -> `0`.
All fall out of the same formula plus the early guards and the final `max(0, hi - loExcl)` clamp.

**Complexity.** `O(1)` per query, `O(q)` total time, `O(1)` extra space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

// Count x in [1, n] with floor(n/x) in [a, b], a<=b.
// floor(n/x) >= a  <=>  x <= floor(n/a)            (for a >= 1; if a <= 0 every x qualifies)
// floor(n/x) <= b  <=>  x >  floor(n/(b+1))        (since floor(n/x) <= b iff x > n/(b+1))
// So x ranges over ( floor(n/(b+1)) , floor(n/a) ], a left-open / right-closed interval.
long long solve(long long n, long long a, long long b) {
    if (n <= 0) return 0;
    if (a > b) return 0;
    if (b < 0) return 0;            // floor(n/x) >= 0 always; cannot be <= negative b
    // Upper index bound from the a-condition (floor(n/x) >= a).
    long long hi;
    if (a <= 0) {
        hi = n;                     // floor(n/x) >= a holds for all x in [1, n]
    } else {
        hi = n / a;                 // largest x with floor(n/x) >= a
    }
    if (hi > n) hi = n;
    // Lower exclusive bound from the b-condition (floor(n/x) <= b).
    long long loExcl = n / (b + 1); // floor(n/x) <= b iff x > floor(n/(b+1))
    if (loExcl < 0) loExcl = 0;
    // x in (loExcl, hi], also forced into [1, n] by construction.
    long long cnt = hi - loExcl;
    if (cnt < 0) cnt = 0;
    return cnt;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int q;
    if (!(cin >> q)) return 0;
    while (q--) {
        long long n, a, b;
        cin >> n >> a >> b;
        cout << solve(n, a, b) << "\n";
    }
    return 0;
}
```
