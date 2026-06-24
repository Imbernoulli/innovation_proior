**Problem.** `n` runners stand at integer positions `p[i]` on a circular track of circumference `L`
(`0 <= p[i] < L`, duplicates allowed). The circular distance between two runners is
`min(|p[i]-p[j]|, L-|p[i]-p[j]|)`. Count the unordered pairs whose circular distance is at most `D`.
Read `n L D` and the positions from stdin, print the count.

**Key idea — rewrite the metric, then two sorted sweeps.** With both positions in `[0, L)`, the raw
gap `d = |p[i]-p[j]|` lies in `[0, L-1]`, and

`min(d, L-d) <= D  <=>  d <= D` (short way) `or  d >= L-D` (long way, around the back).

Sort the positions. Each linear condition is a textbook two-pointer count:

- **Close** (`d <= D`): for each right endpoint `hi`, advance `lo` while `p[hi]-p[lo] > D`; add `hi-lo`.
- **Far** (`d >= L-D`): for each `hi`, advance `lo` while `p[hi]-p[lo] >= L-D`; the indices `[0..lo-1]`
  qualify, so add `lo`.

Each pair is counted once, at its larger index. The answer is the sum of the two sweeps — **but only
when the two conditions are disjoint.**

**Pitfalls.**
1. *Double-count at and above the diameter.* The short set `{d <= D}` and the far set `{d >= L-D}` are
   disjoint iff `D < L-D`, i.e. `2*D < L`. When `2*D >= L` they touch/overlap and *every* pair
   qualifies, so the answer is simply `n*(n-1)/2`. Special-case this **first**, with the guard `2*D >= L`
   (not `>`). A strict `>` routes the boundary case into split-and-add: e.g. `n=2, L=4, D=2, p=[0,2]`
   has the single pair counted once as short (`2 <= 2`) and once as far (`2 >= 2`), giving `2` instead
   of `1`.
2. *Far-side comparison must be `>=`.* The exact-boundary wrap pair (e.g. `p=[1,8]`, `L=10`, `D=3`:
   `d=7 = L-D`) is caught only by `>= L-D`; a `> L-D` misses it.
3. *Overflow.* The pair count reaches `~n*(n-1)/2 ~ 2*10^10` and `2*D` reaches `2*10^9`; use `long long`
   for the count, for `L`, for `D`, and widen `n*(n-1)` before multiplying. A 32-bit int is a silent
   wrong-answer on the large tests.

**Edge cases.** `n = 0` and `n = 1` -> `0` (no pairs). `D = 0` -> only coincident runners count (close
sweep on equal positions; far needs `d >= L`, impossible). `D = L` and any `2*D >= L` -> `n*(n-1)/2`.
All runners on one spot -> `C(n,2)`. All handled by the regime guard plus the two sweeps.

**Complexity.** `O(n log n)` for the sort, `O(n)` for the two sweeps; `O(1)` extra space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long L, D;
    if (!(cin >> n >> L >> D)) return 0;
    vector<long long> p(n);
    for (auto &x : p) cin >> x;
    sort(p.begin(), p.end());

    long long total = (long long)n * (n - 1) / 2;

    // If 2D >= L every unordered pair is within circular distance D.
    if (2 * D >= L) {
        cout << total << "\n";
        return 0;
    }

    // 2D < L: the "close" condition (d <= D) and the "far" condition
    // (d >= L-D) are disjoint, so count each with a separate sweep and add.
    long long ans = 0;

    // Close: number of pairs (i<j) with p[j]-p[i] <= D.
    {
        int lo = 0;
        for (int hi = 0; hi < n; hi++) {
            while (p[hi] - p[lo] > D) lo++;
            ans += (long long)(hi - lo); // pairs (lo..hi-1, hi)
        }
    }

    // Far: number of pairs (i<j) with p[j]-p[i] >= L-D.
    {
        long long thr = L - D;
        int lo = 0;
        for (int hi = 0; hi < n; hi++) {
            while (p[hi] - p[lo] >= thr) lo++;
            // indices [0 .. lo-1] satisfy p[hi]-p[i] >= thr
            ans += (long long)lo;
        }
    }

    cout << ans << "\n";
    return 0;
}
```
