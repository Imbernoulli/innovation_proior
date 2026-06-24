**Problem.** There are `n` gigs; gig `i` has a deadline `d[i]` (it must be finished on some day in
`1..d[i]`) and a payout `v[i]` (which may be negative or zero). You finish at most one gig per day and
may decline any gig. Maximize the total payout. Declining everything is allowed, so the answer is at
least `0`. Read `n`, the `n` deadlines, and the `n` payouts from stdin; print the maximum total payout.

**Key idea — greedy exchange with a "latest free day" structure.** Two exchange arguments drive the
whole solution:

1. *Sign reduction.* If an optimal plan accepts a gig with `v[i] <= 0`, deleting that gig leaves the
   schedule feasible and changes the total by `-v[i] >= 0` — never worse. So an optimal plan accepts
   **only strictly positive** gigs; filter to `v[i] > 0` first.
2. *Latest-day placement.* Process the positive gigs in **decreasing payout** order and assign each to
   the **latest still-free day at or before its deadline**; drop it if no such day exists. Parking a
   gig as late as legally allowed never blocks a placement a different choice would have kept open, so
   the greedy set is optimal.

Use a disjoint-set forest as the "latest free day" structure: `findFree(x)` returns the largest free
day index `<= x` (or `0` if none), and accepting day `s` sets `par[s] = s - 1` so later queries skip
it. Day `0` is the sentinel "no valid day."

**Correctness.** By argument (1) the optimal set uses only positive gigs. By the standard deadline-
scheduling exchange argument (2), among positive gigs the descending-payout / latest-free-day greedy
produces a maximum-weight schedulable set: any schedulable set the optimum uses can be transformed
into the greedy's set by a sequence of swaps that never lower the total, because a higher-payout gig
placed as late as possible dominates. Feasibility of the greedy's choices is exactly the Hall
condition (sorted accepted deadlines `d_j >= j`), which the latest-free-day assignment maintains.

**Pitfalls.**
1. *Sign / base case (the headline trap).* Accepting any gig you can physically fit, ignoring the
   payout's sign, will take a loss-making gig and can drive the total **below zero** — impossible,
   since declining everything yields `0`. A trace of `n=1, v=[-7]` returning `-7` exposes exactly
   this. Filter to `v[i] > 0`; then the empty instance, all-negative, and all-zero inputs all return
   `0` through the *same* code path (`order` is empty).
2. *Invalid-day sentinel.* A deadline of `0` means no day `>= 1` is `<= 0`, so the gig is unplaceable.
   Day `0` must be a sentinel: accept only when `findFree(cap) > 0` (strict), never `>= 0`. A `>= 0`
   guard would accept an unplaceable gig and write a negative index, corrupting the forest. A trace of
   `d=[0,1], v=[100,50]` (answer `50`, not `150`) catches it.
3. *Deadline cap and overflow.* Deadlines up to `10^9` would size a day array hopelessly large; cap
   each to `min(d[i], n)` (no schedule needs a day index `> n`). With `n` up to `2*10^5` and `|v[i]|`
   up to `10^9`, the total reaches `~2*10^14`; use `long long`, an `int` is a silent wrong-answer.

**Edge cases.** `n = 0` -> `0`; a single negative or zero gig -> `0`; all-negative / all-zero -> `0`;
a positive gig with deadline `0` -> declined; deadlines exceeding `n` -> capped harmlessly; ties on
the same early day -> total is order-independent because the tied payouts are equal.

**Complexity.** `O(n log n)` for the sort, `O(n alpha(n))` for the disjoint-set placement, `O(n)`
extra space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

// Disjoint-set "latest free day": find(x) returns the largest day index <= x
// that is still free (0 means no free day at or before x).
static vector<int> par;
int findFree(int x) {
    while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; }
    return x;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;                 // n = 0 (or empty input) -> answer 0
    vector<long long> d(n), v(n);
    for (int i = 0; i < n; i++) cin >> d[i];
    for (int i = 0; i < n; i++) cin >> v[i];

    // Only positive-payout gigs are ever worth scheduling; a gig with v <= 0
    // can always be declined for a strictly-not-worse total, so we skip it.
    // Among positive gigs, sort by payout descending and assign each to the
    // latest still-free day at or before its deadline (greedy exchange).
    vector<int> order;
    order.reserve(n);
    long long maxDay = 0;
    for (int i = 0; i < n; i++) {
        if (v[i] > 0) {
            order.push_back(i);
            // a deadline beyond n is useless: at most n gigs fit, so cap at n.
            long long cap = min(d[i], (long long)n);
            if (cap > maxDay) maxDay = cap;
        }
    }
    sort(order.begin(), order.end(), [&](int a, int b) { return v[a] > v[b]; });

    par.assign((size_t)maxDay + 1, 0);
    for (int day = 0; day <= (int)maxDay; day++) par[day] = day;

    long long answer = 0;
    for (int idx : order) {
        long long cap = min(d[idx], maxDay);   // latest day this gig may occupy
        if (cap <= 0) continue;                // deadline 0 -> no valid day
        int slot = findFree((int)cap);
        if (slot > 0) {                        // a free day exists
            answer += v[idx];
            par[slot] = slot - 1;              // mark day `slot` used
        }
    }

    cout << answer << "\n";                     // empty / all-nonpositive -> 0
    return 0;
}
```
