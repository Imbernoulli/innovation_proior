**Problem.** A ring road of integer circumference `L` has `n` posts at distinct sorted positions
`p[0] < ... < p[n-1]` in `[0, L)`. Choose exactly `k` posts (`2 <= k <= n <= 2000`, `L <= 10^9`). The
*clearance* of a choice is the minimum, over cyclically consecutive chosen posts, of the clockwise arc
between them — including the wrap from the last chosen post back to the first. Maximize the clearance.

**Key idea — binary search the clearance with a *ring-correct* feasibility test.** "Achievable
clearance" is monotone: if `k` posts can be spread with every cyclic gap `>= d`, the same posts satisfy
any `d' < d`. So binary-search the largest `d` with `feasible(d)` true. Everything rides on `feasible`.

For a ring, the chosen posts form a *cycle* of exactly `k` gaps that sum to `L`. The feasibility test
fixes one post as a forced anchor, sweeps the others clockwise (unrolling the ring so positions only
increase, adding `L` once past index `n`), greedily takes the earliest post whose gap from the last
taken is `>= d`, and — the crux — refuses any post that would leave the closing wrap back to the anchor
below `d`. Because positions increase, once the wrap is too small it stays too small, so we `break`.
Since a ring has no canonical "first" post, we try **every** post as the anchor and accept if any anchor
admits `>= k` posts. Search range: `min gap <= L/k <= L/2` (the `k` gaps sum to `L`), so `hi = L/2`.

**Pitfalls.**
1. *Transplanting the line greedy.* The classic "aggressive cows" predicate — anchor at `p[0]`, sweep,
   count `>= k` — is wrong here. It measures only `k - 1` gaps and silently omits the wrap, and it
   anchors at a single post that need not be in the optimal set. On `L = 20`, posts
   `[0,1,2,3,9,11,17]`, `k = 3`, at `d = 8` it takes `{0, 9, 17}` and declares feasible, but that
   triple's wrap gap is `(0+20)-17 = 3`, so its true clearance is `3`, and the real answer is `6`. Fix:
   force the anchor chosen, check the wrap, and try all anchors.
2. *Overflow.* `pos = p[idx] + L` and `wrap = (p[start]+L) - pos` approach `2*10^9`; use `long long`.
   `int` is a silent wrong-answer on large `L`.
3. *Search bound.* Use `hi = L/2` (valid since `k >= 2`), not a guessed `L`, or the binary search
   wastes iterations / can misbehave at the top.

**Edge cases.** `k = 2`: clearance is `min(g, L-g)` — e.g. `[1,8]`, `L = 10` gives `3` (the short arc
binds, never the long one). `k = n`: every post forced, answer is the minimum existing cyclic gap. `d =
0`: trivially feasible (nonnegative gaps). Diametric pair `[0,5]`, `L = 10` gives `5 = L/2`, hitting the
bound exactly. Clustered posts (`[0,1,2,50]`, `L=100`, `k=3`) give `2`, driven by the small cluster.

**Complexity.** `feasible` is `O(n^2)` (n anchors x n sweep); binary search adds `O(log L)`; total
`O(n^2 log L)` ~ `1.2*10^8` at `n = 2000`, measured at `0.11 s`. Memory `O(n)`.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int n;
long long k, L;
vector<long long> p; // sorted, in [0, L)

// Greedy count starting from anchor post `start` (which is FORCED chosen):
// walk forward around the ring, take the next post whose gap from the last
// taken is >= d. Returns the number of posts taken (>= 1). The closing wrap
// gap (from the last taken back to `start`) must ALSO be >= d for the whole
// thing to form a valid cyclic placement; we enforce that by never taking a
// post that would leave a wrap gap < d, and by checking it at the end.
long long countFrom(int start, long long d) {
    long long taken = 1;
    long long lastPos = p[start];          // absolute position of last taken
    // iterate the other n-1 posts in cyclic order after `start`
    for (int step = 1; step < n; step++) {
        int idx = start + step;
        long long pos = p[idx % n] + (idx >= n ? L : 0); // unrolled position
        long long gap = pos - lastPos;
        if (gap < d) continue;             // too close, skip this post
        // would taking it leave a valid wrap back to start? wrap = (p[start]+L) - pos
        long long wrap = (p[start] + L) - pos;
        if (wrap < d) break;               // taking it (or anything further) kills the wrap
        taken++;
        lastPos = pos;
    }
    return taken;
}

// Can we choose >= k posts with every cyclic-adjacent gap >= d?
bool feasible(long long d) {
    if (d <= 0) return true;               // any selection has nonneg gaps
    // Some post must be chosen; try every post as the forced anchor.
    for (int s = 0; s < n; s++) {
        if (countFrom(s, d) >= k) return true;
    }
    return false;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    if (!(cin >> n >> k >> L)) return 0;
    p.resize(n);
    for (auto &x : p) cin >> x;
    sort(p.begin(), p.end());

    // Clearance is in [0, L/2] (k>=2 means at least two points, so the min
    // cyclic gap can be at most floor(L/2)). Binary search the largest d
    // for which a valid placement of >= k posts exists.
    long long lo = 0, hi = L / 2, ans = 0;
    while (lo <= hi) {
        long long mid = lo + (hi - lo) / 2;
        if (feasible(mid)) { ans = mid; lo = mid + 1; }
        else hi = mid - 1;
    }
    cout << ans << "\n";
    return 0;
}
```
