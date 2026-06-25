**Problem.** Build a length-`n` drum loop over `{H, R}` with exactly `h` hits and **no run of `K`
consecutive rests** (every maximal run of `R` has length `<= K-1`). Among all legal loops output the
**lexicographically smallest** under `H < R`, or `-1` if none exists. Read `n K h` from stdin
(`1 <= n <= 10^7`, `1 <= K <= n`, `0 <= h <= n`); print the pattern or `-1`.

**Key idea — feasibility as a gap-packing count, then a guarded greedy.** With `h` hits the timeline
splits into `h + 1` gaps (before the first hit, between hits, after the last), each holding at most
`K - 1` rests. So a legal loop exists iff

```
n - h <= (h + 1) * (K - 1)        (with 0 <= h <= n, K >= 1).
```

To get the lexicographically smallest loop, prefer `H` (the smaller symbol) at every beat, but only
when the *remaining suffix stays completable* — a hit spent too early can strand the trailing rests.
The completability test is the same bound applied to a suffix. Define `canFill(m, hh, c)` = "fill `m`
slots with `hh` hits, no `K`-run, given a standing run of `c` rests just before slot 0":

```
canFill(m, hh, c)  iff  0 <= hh <= m  AND  c <= K-1  AND  (m - hh) <= (hh+1)*(K-1) - c
```

(the first of the `hh+1` gaps is glued to the existing run `c`). Global feasibility is
`canFill(n, h, 0)`. Then sweep left to right: at a beat with `m` slots after it, `hitsLeft` hits, and
trailing run `c`, place `H` iff `hitsLeft >= 1 && canFill(m, hitsLeft-1, 0)` (a hit resets the run);
otherwise place `R`. Because the start state is feasible and every action keeps the suffix feasible,
the `R` branch never creates an illegal run.

**Pitfalls.**
1. *Off-by-one in the gap count.* There are `h + 1` gaps, not `h - 1` or `h`. The witness
   `n=3, K=2, h=1`: true capacity `(1+1)*1 = 2 >= 2`, so `RHR` is legal; `h*(K-1) = 1` would wrongly
   reject it.
2. *Standing-run guard.* `canFill` must reject `c > K-1` explicitly. Without it, a large hit budget
   inflates `(hh+1)*(K-1)` and masks an already-illegal run (e.g. `canFill(0, 2, 3)` with `K=3`
   wrongly reads as feasible), letting the greedy emit a `K`-run.
3. *32-bit overflow at scale.* `(hh+1)*(K-1)` reaches `~3*10^14`. A 32-bit multiply matches the brute
   force on every small test and then overflows: on the feasible `n=K=10^7, h=300` it wraps negative
   and prints `-1`. Use `long long` for all counts. (This is the "passed `n<=10`, scored 0" trap.)
4. *Output volume.* Up to `10^7` characters — build one `std::string` and write it once; per-character
   `cout` times out.

**Edge cases.** `n=1, K=1, h=1 -> H`; `n=1, K=1, h=0 -> -1`; `K=1` is feasible only when `h = n`
(no rest allowed); `h = 0` is feasible iff `n <= K-1` (`n=2,K=3 -> RR`, `n=3,K=3 -> -1`); `h = n` is
all `H`s; the tight boundary `n=8,K=3,h=2 -> RRHRRHRR` (gaps packed to exactly `K-1`), one beyond it
`n=9,K=3,h=2 -> -1`.

**Complexity.** `O(n)` time, `O(1)` extra state beyond the `O(n)` output buffer.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

// canFill(m, hh, c): can we fill m remaining slots with exactly hh hits (and m-hh rests),
// no run of K consecutive rests, given the run of rests ALREADY standing immediately
// before slot 0 has length c (0 <= c <= K-1)?
//
// hh hits split the m-hh rests into hh+1 gaps g_0,...,g_hh. The first gap g_0 is glued to
// the existing run c, so g_0 + c <= K-1; every other gap g_i <= K-1. Hence the maximum
// number of rests that fit is (K-1-c) + hh*(K-1) = (hh+1)*(K-1) - c. So feasibility is:
//   0 <= hh <= m  AND  c <= K-1  AND  (m - hh) <= (long long)(hh+1)*(K-1) - c
// Everything is 64-bit: (hh+1)*(K-1) can reach ~1e14.
static inline bool canFill(long long m, long long hh, long long c, long long K) {
    if (hh < 0 || hh > m) return false;
    if (c > K - 1) return false;
    long long capacity = (hh + 1) * (K - 1) - c; // max rests that fit
    return (m - hh) <= capacity;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    long long n, K, h;
    if (!(cin >> n >> K >> h)) return 0;

    // Global feasibility: fill all n slots with h hits, starting with no standing rest run.
    if (!canFill(n, h, 0, K)) {
        cout << "-1\n";
        return 0;
    }

    // Greedy: build lexicographically smallest pattern with 'H' < 'R'. Prefer 'H'; place it
    // whenever a hit is left and the remaining suffix stays completable. Otherwise place 'R'
    // (allowed only if it does not reach a run of K and the suffix stays completable).
    string out;
    out.reserve((size_t)n);
    long long hitsLeft = h;
    long long c = 0; // current trailing run of rests
    for (long long pos = 0; pos < n; pos++) {
        long long m = n - pos - 1; // slots remaining AFTER this one
        if (hitsLeft >= 1 && canFill(m, hitsLeft - 1, 0, K)) {
            out.push_back('H');
            hitsLeft -= 1;
            c = 0;
        } else {
            // place 'R'; invariant guarantees this branch is feasible
            out.push_back('R');
            c += 1;
        }
    }

    out.push_back('\n');
    cout << out;
    return 0;
}
```
