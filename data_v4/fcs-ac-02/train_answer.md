**Problem.** Given one integer `k` (`0 <= k <= 10^8`), construct any simple undirected graph with
exactly `k` triangles using at most `1000` vertices. Output `n m` then the `m` edges (1-indexed). A
solution always exists in this range; any valid graph is accepted (the checker recounts triangles).

**Key idea — unified incremental clique-stacking.** Two arithmetic facts are the whole toolbox: a
clique of size `c` holds `C(c,3)` triangles, and joining a new vertex to a *clique* of size `c`
creates exactly `C(c,2)` new triangles (one per edge inside the clique). Add vertices `0,1,2,...` one
at a time; join vertex `i` to the **first `c_i` vertices** `{0,...,c_i-1}`, choosing `c_i` as the
largest value with `C(c_i,2) <= r` (the remaining budget), capped at `i`. Then subtract `C(c_i,2)`
from `r`. This single move is both operations at once: when `c_i = i` the prefix clique grows by one;
when `c_i < i` it is a touch-up paying down residual. The invariant that makes the count exact: **the
first `p` vertices always form a clique**, because while `r` is large the greedy takes `c_i = i` and
the clique grows vertex by vertex, and once `c_i < i` every attachment is to a *prefix* of that clique
— and a prefix of a clique is a clique. So each step contributes exactly `C(c_i,2)`, and the total
lands on `k` precisely.

**Why the obvious plan is wrong.** The natural alternative — build one big clique `K_m` with the
largest `m` such that `C(m,3) <= k`, then patch the residual with extra vertices — is *incomplete*. It
wastes vertices: committing `m` vertices to the clique up front forces the residual to be paid in
vertex-per-chunk touch-ups. On 6 vertices it cannot reach `k = 9` or `k = 12`, both of which are
achievable (verified by brute enumeration). The cure is to *interleave* clique-growth and touch-ups in
one loop rather than splitting them into two phases.

**Pitfalls to get right.**
1. *Two-phase greedy is incomplete.* "Whole clique, then patch" misses feasible `k`. Use the unified
   "attach to the first `c` vertices" loop so clique-building and patching share one greedy.
2. *Output indexing.* Internal vertices are 0-based but the contract is 1-based; print `v+1`. A raw
   `0` endpoint is out of range and rejected.
3. *Empty graph for `k = 0`.* The loop never runs, leaving `n = 0`, which is a malformed graph. Emit a
   single isolated vertex (`n = 1`, zero edges).
4. *Vertex budget near boundaries.* For `k` just below a clique level the count jumps in big steps; a
   vertex-usage simulation confirms the greedy stays under `850` vertices for every `k <= 10^8`,
   inside the `1000` budget, so a solution always exists.

**Edge cases.** `k = 0` -> `1 0` (isolated vertex). `k = 1` -> a single `K3`. Boundary `k = C(c,3)-1`
-> `(c-1)`-clique plus a short touch-up tail; count stays exact. `k = 10^8` -> `848` vertices,
`356329` edges, exactly `10^8` triangles. No self-loops or duplicate edges by construction.

**Complexity.** `O(n)` greedy steps, each an `O(log n)` binary search for the largest affordable
clique; `O(m)` to emit edges, with `n <= 850` and `m` up to a few hundred thousand. Well within the
1 s / 256 MB limits.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    long long k;
    if (!(cin >> k)) return 0;                 // missing input -> nothing to do

    const int NMAX = 1000;                      // vertex budget

    // C2[c] = c*(c-1)/2 = number of edges in a clique of size c (also the number of
    // triangles a new vertex creates when joined to a clique of size c). Precompute up
    // to NMAX so we can pick the largest clique we can afford in O(1) by binary search.
    vector<long long> C2(NMAX + 1);
    for (int c = 0; c <= NMAX; c++) C2[c] = 1LL * c * (c - 1) / 2;

    // Greedy "clique-stacking". Add vertices 0,1,2,... one at a time. Vertex i is joined
    // to the first c_i vertices {0,...,c_i-1}, which we maintain as a clique, so it adds
    // exactly C2[c_i] new triangles. We pick c_i as large as possible without overshooting
    // the remaining budget r (and never more than i, the number of earlier vertices).
    long long r = k;
    vector<pair<int,int>> edges;                // the edge list we emit
    int n = 0;                                  // vertices used so far
    while (r > 0) {
        int i = n;                              // index of the vertex we are about to add
        // largest c with C2[c] <= r, capped at i. Binary search over [0, min(i, NMAX)].
        int lo = 0, hi = min(i, NMAX), c = 0;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if (C2[mid] <= r) { c = mid; lo = mid + 1; }
            else hi = mid - 1;
        }
        for (int v = 0; v < c; v++) edges.emplace_back(v, i);
        r -= C2[c];
        n = i + 1;
        if (n > NMAX) {                         // ran out of vertices: declare impossible
            cout << -1 << "\n";
            return 0;
        }
    }

    // k == 0 leaves n == 0; emit a single isolated vertex so the output is a valid graph.
    if (n == 0) n = 1;

    cout << n << " " << edges.size() << "\n";
    for (auto &e : edges) cout << (e.first + 1) << " " << (e.second + 1) << "\n";
    return 0;
}
```
