**Problem.** A rooted tree of `n` relay stations (root = gateway). Station `u` transmits *downward only*: it covers a strict descendant `v` iff `depth(v) - depth(u) <= p[u]`, where `p[u]` is `u`'s power and `depth` counts edges from the root. A station is **covered** if some proper ancestor reaches it; the root is never covered. Read the tree (each station's parent — `-1` for the root — and power) from stdin; print how many stations are covered.

**Key idea — one DFS carrying a running reach.** Ancestor `u` reaches down to absolute depth `depth(u) + p[u]`. For a node `v` define `reach(v) = max over PROPER ancestors u of ( depth(u) + p[u] )`. Then

```
v is covered  <=>  depth(v) <= reach(v).
```

Descending from a node to its child only *adds* an ancestor to that max, so `reach` is monotone nondecreasing down every root-to-node path. Hence a single DFS that threads a running maximum down each path computes `reach(v)` for every node in `O(n)`: at node `u`, test `depth(u) <= reach` (the value inherited from `u`'s proper ancestors), then hand each child `max(reach, depth(u) + p[u])`.

**Correctness.** By induction on depth, the `reach` value carried into node `u` equals the maximum of `depth+p` over `u`'s entire proper-ancestor chain (each edge folds in exactly one more ancestor — the node just left). The coverage test `depth(u) <= reach` is the rearrangement of "some proper ancestor `a` has `depth(u) - depth(a) <= p[a]`". Monotonicity guarantees a running max never needs to drop an ancestor, so one pass suffices.

**Pitfalls (both are boundaries).**
1. *Self-coverage / strict descendant.* Test `u` against the `reach` inherited from its **proper** ancestors, then fold in `u`'s own `depth(u)+p[u]` only for the children. Folding it in before the test lets a node cover itself — the root would get counted, which is wrong (a trace of the sample shows the root being counted if you do this).
2. *Inclusive vs exclusive.* "At most `p[u]` edges" means `depth(v) <= reach`, an inclusive `<=`. Writing `<` silently drops every node sitting exactly on an ancestor's range boundary (in the sample, nodes `4` and `3` are exactly such boundary nodes and would be missed).

**Edge cases.** `n = 0` and `n = 1` -> `0` (no ancestors to cover anything). All powers `0` -> `0` (a power-`0` station has no strict descendant within distance `0`). Maximum power `10^9` -> reaches the whole subtree. Deep chain of length `2*10^5`: use an *explicit-stack* DFS, not recursion, or the stack overflows.

**Complexity.** `O(n)` time, `O(n)` space (children lists + the explicit stack).

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (scanf("%d", &n) != 1) return 0;
    if (n <= 0) { printf("0\n"); return 0; }

    vector<int> par(n), p(n);
    vector<vector<int>> ch(n);
    int root = -1;
    for (int i = 0; i < n; i++) {
        int pa;
        if (scanf("%d %d", &pa, &p[i]) != 2) return 0; // parent (-1 for root) and power
        par[i] = pa;
        if (pa == -1) root = i;
        else ch[pa].push_back(i);
    }

    // A station u covers a STRICT descendant v iff depth(v) - depth(u) <= p[u].
    // Carry along the DFS, for each node v, the value
    //   reach = max over PROPER ancestors u of ( depth(u) + p[u] ).
    // Then v is covered  <=>  depth(v) <= reach   (inclusive boundary, excludes self).
    long long covered = 0;
    const long long NEG = LLONG_MIN / 4; // "no ancestor reaches here yet"

    struct Frame { int node; int depth; long long reach; };
    vector<Frame> st;
    st.reserve(n);
    st.push_back({root, 0, NEG}); // root has no proper ancestor

    while (!st.empty()) {
        Frame f = st.back();
        st.pop_back();
        int u = f.node;
        int d = f.depth;
        long long reach = f.reach;

        if ((long long)d <= reach) covered++; // some proper ancestor reaches depth d

        long long reachIncludingU = max(reach, (long long)d + p[u]);
        for (int w : ch[u]) {
            st.push_back({w, d + 1, reachIncludingU});
        }
    }

    printf("%lld\n", covered);
    return 0;
}
```
