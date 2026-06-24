**Problem.** A single shared studio receives `n` booking requests; request `i` occupies the half-open interval `[s_i, f_i)` and pays profit `p_i > 0`. Accept any subset of pairwise non-overlapping requests (touching endpoints `f_i == s_j` are allowed) and maximize total profit; the empty schedule is allowed, so the answer is at least `0`. Read `n` and the `n` triples `s_i f_i p_i` from stdin, print the maximum profit.

**Why the obvious greedy is wrong.** This is *weighted* interval scheduling, and no single sort key is optimal. "Earliest finishing time" — the famous optimal rule for *maximizing the count* of non-overlapping intervals — fails once profits differ: on `[0,1) p=1` versus `[0,3) p=5` it grabs the short interval for `1`, blocking the `5`. "Largest profit first" fails too: on `[0,4) p=10` versus `[0,2) p=7` plus `[2,4) p=7` it takes the fat `10` and blocks the two lean intervals that together touch at `2` and pay `14`. Each greedy makes a local accept that commits the shared resource over a span whose global opportunity cost it never measures. Both are discarded.

**Key idea — sort by finish, then DP with binary search.** Sort the requests by finishing time ascending and relabel `0..n-1`. Let `best[i]` be the maximum profit using only the first `i` requests (the `i` smallest finishing times), with `best[0] = 0`. For request `i`:

- *Reject `i`*: keep `best[i]`.
- *Accept `i`*: collect `p_i`; the compatible earlier requests are exactly those finishing at or before `s_i`. Because finishing times are sorted, these form a prefix of length `j`, so accepting gives `best[j] + p_i`.

Hence `best[i+1] = max(best[i], best[j] + p_i)`, and the answer is `best[n]`. Find `j` — the count of earlier requests with `F[k] <= s_i` — by binary search on the sorted finishing-time array. This is `O(n log n)`.

**Pitfalls.**
1. *Touching endpoints.* Intervals are half-open, so a predecessor is compatible when it finishes `<= s_i`, not `< s_i`. A strict `F[k] < s_i` predicate drops valid touching predecessors: on `[0,2) p=20` plus `[2,4) p=6` it returns `20` instead of `26`. Use `F[k] <= s_i`.
2. *Overflow.* With `n` up to `2*10^5` and `p_i` up to `10^9`, the total reaches `~2*10^14`; use `long long` for the profit accumulator. An `int` is a silent wrong-answer on large tests.
3. *Index discipline.* The binary search returns the count of compatible predecessors `lo in [0,i]`; index `best[lo]`, where `best` has size `n+1` and `best[i]` means "first `i` requests."

**Edge cases (all handled by the recurrence + `best` initialized to 0):** `n = 0` -> `0`; a single request -> its profit; all requests mutually overlapping -> the single largest profit (every search yields `lo=0`, so `best[i+1]=max(best[i],p_i)`); a touching chain -> the full sum.

**Complexity.** `O(n log n)` time (one sort plus `n` binary searches), `O(n)` space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> profit 0
    vector<long long> s(n), f(n), p(n);
    for (int i = 0; i < n; i++) cin >> s[i] >> f[i] >> p[i];

    // Sort jobs by finishing time (the sweep order).
    vector<int> idx(n);
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&](int x, int y){ return f[x] < f[y]; });
    vector<long long> S(n), F(n), P(n);
    for (int i = 0; i < n; i++) { S[i] = s[idx[i]]; F[i] = f[idx[i]]; P[i] = p[idx[i]]; }

    // best[i] = max profit considering the first i jobs (in finish order).
    // For job i (0-based), find p = largest index j < i with F[j] <= S[i] (compatible),
    // via binary search on the sorted F array.
    vector<long long> best(n + 1, 0);
    for (int i = 0; i < n; i++) {
        // largest k in [0, i) with F[k] <= S[i]; count of such = j, so best[j] is reachable.
        int lo = 0, hi = i; // search in F[0..i-1]
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (F[mid] <= S[i]) lo = mid + 1;
            else hi = mid;
        }
        // lo = number of jobs among first i with F <= S[i]
        long long take = best[lo] + P[i];
        long long skip = best[i];
        best[i + 1] = max(take, skip);
    }

    cout << best[n] << "\n";
    return 0;
}
```
