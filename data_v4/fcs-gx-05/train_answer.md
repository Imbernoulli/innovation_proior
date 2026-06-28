**Problem.** There are `T` slots (`1..T`), each usable by at most one request, and `n` requests where request `i` may only be served in a contiguous window of slots `[l_i, r_i]`. Decide whether all `n` requests can be served simultaneously — each in a distinct slot inside its own window. Print `YES` and one valid assignment (the slot for each request, in input order), or `NO`. Constraints: `n, T <= 2*10^5`.

**Why the obvious matching is the wrong representation.** This is a bipartite feasibility test: request `i` adjacent to every slot in `[l_i, r_i]`, answer `YES` iff a matching saturates all requests. Correct, but the graph has up to `n*T` edges — for `n = T = 2*10^5` with windows `[1, T]` that is `4*10^10` edges, impossible to even build. The general matcher discards the one fact that matters: each request's allowed slots form a contiguous **interval**.

**Key idea — earliest-deadline-first greedy + a Hall-condition feasibility test.** Process requests by **increasing right endpoint** `r_i` (deadline), ties by `l_i`; give each request the earliest still-free slot at or after `l_i`. If none exists within `[l_i, r_i]`, the instance is infeasible. EDF is optimal by an exchange argument: if it cannot place request `x`, then the requests with deadline `<= r_x` collectively demand more slots than exist in their common range — a violated Hall condition — so no assignment could have succeeded. Thus the greedy is a constructive feasibility test, not a heuristic.

To answer "earliest free slot `>= l`" in near-`O(1)` amortised, use a disjoint-set "next free slot" structure: `nxt[s]` points to the earliest free slot `>= s`. `find(l_i)` gives the slot; consuming slot `s` sets `nxt[s] = s + 1` so future finds skip it. Total time `O((n + T) * alpha)` plus the `O(n log n)` sort.

**Pitfalls.**
1. *Wrong sort key.* Sorting by `l_i` (release time) is wrong: on `[1,2], [1,1]` the flexible request grabs slot 1 and starves the rigid one, giving a false `NO`. Sort by `r_i` (deadline) — the request with less room must pick first.
2. *Missing sentinel.* Initialise `nxt` over `1..T+1`, not just `1..T`. If a range is saturated, `find` must land on the fixed point `T+1` (always `> r_i`) instead of walking into uninitialised memory and returning a bogus slot. A trace of `T=1` with two `[1,1]` requests exposes this: without the sentinel it falsely reports `YES`.

**Edge cases.** `n = 0` -> `YES` with an empty assignment line; `n = 1` trivial; `n > T` -> `NO` (a request hits the sentinel); point windows piling on one slot -> the second one fails; all windows `[1,T]` with `n = T` -> `YES` permutation. No special-casing needed — the sentinel handles `n > T` and saturation uniformly. Slot indices fit in `int`; there is no summation, so no overflow.

**Complexity.** `O(n log n + (n + T) * alpha(T))` time, `O(T)` space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

// Disjoint-set "next free slot" structure.
// nxt[s] points to the earliest slot >= s that is still free (or T+1 if none).
static int nxt[200005];

int find_free(int s) {
    // Path-compressed walk to the representative free slot.
    while (nxt[s] != s) {
        nxt[s] = nxt[nxt[s]];
        s = nxt[s];
    }
    return s;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long T;
    if (!(cin >> n >> T)) return 0;

    vector<int> l(n), r(n), order(n);
    for (int i = 0; i < n; i++) {
        cin >> l[i] >> r[i];
        order[i] = i;
    }

    // Earliest-deadline-first: process requests by increasing right endpoint,
    // breaking ties by increasing left endpoint.
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (r[a] != r[b]) return r[a] < r[b];
        return l[a] < l[b];
    });

    // Initialize the "next free slot" pointers for slots 1..T, plus a sentinel T+1.
    // (We only ever touch slots that appear as some l[i], so T may be huge; we
    //  index the DSU only up to the largest endpoint actually needed, but here
    //  endpoints are <= T and T fits in the array bound by the constraints.)
    int Ti = (int)T;
    for (int s = 1; s <= Ti + 1; s++) nxt[s] = s;

    vector<int> assign(n, -1);
    bool feasible = true;

    for (int idx : order) {
        int s = find_free(l[idx]);     // earliest free slot >= l[idx]
        if (s > r[idx]) {              // no free slot within [l, r]
            feasible = false;
            break;
        }
        assign[idx] = s;               // serve request idx in slot s
        nxt[s] = s + 1;                // slot s is now taken; point past it
    }

    if (!feasible) {
        cout << "NO\n";
    } else {
        cout << "YES\n";
        for (int i = 0; i < n; i++) {
            cout << assign[i] << (i + 1 < n ? ' ' : '\n');
        }
        if (n == 0) cout << '\n';
    }
    return 0;
}
```
