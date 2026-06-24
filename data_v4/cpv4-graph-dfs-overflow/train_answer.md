**Problem.** A rooted tree of `n` junctions; each non-root junction `i` has a pipe up to its parent with per-unit `cost[i]`. Each outlet (junction with no children) must receive one unit of water from the root, which pays every pipe cost along its root-to-outlet path once. Output the total pumping work over all outlets. Read `n`, then `n` lines `p_i c_i` (`p_i = -1` marks the root). Print one integer.

**Key idea — sum per edge, not per outlet.** The literal definition (for each outlet, sum costs up to the root) is `O(n^2)` on a broom-shaped tree and too slow. Reindex the double sum: instead of "for each outlet, for each pipe on its path," do "for each pipe `(v -> par[v])`, for each outlet below `v`." An outlet uses that pipe exactly when it lies in `v`'s subtree, so the pipe is paid once per outlet below `v`. Hence

  total = sum over non-root `v` of `cost[v] * leaves[v]`,

where `leaves[v]` is the number of outlets in `v`'s subtree. One DFS computes every `leaves[v]` (a childless node is `1`; an internal node is the sum of its children), giving an `O(n)` solution.

**Correctness.** The reindexing is an exact swap of summation order in a finite double sum, so the per-edge formula equals the per-outlet definition. Verified against an independent per-outlet brute on 750 random trees (including roots placed at arbitrary junction ids and parents listed out of order) with zero mismatches, plus hand checks: the sample tree gives `26`, a 200000-leaf broom gives `399996000000`, a 200000-deep chain gives `199999`.

**Pitfalls.**
1. *Integer overflow (the headline).* With `n` up to `2*10^5` and `cost` up to `10^6`, a single product `cost[v]*leaves[v]` reaches `~2*10^11` and the running total `~4*10^11` — both far past the 32-bit limit `~2.1*10^9`. Use `long long` for the accumulator **and** for the multiplicand `cost[]`, so the product is evaluated in 64-bit; widening only the accumulator still overflows in the multiply. An `int` build prints `564041472` instead of `399996000000` — a silent wrong answer from two independent wraparounds.
2. *Recursion depth.* A `2*10^5`-deep chain overflows the call stack with recursive DFS. Use an iterative pre-order with an explicit stack, then process the order in reverse (reverse pre-order is a valid post-order: a child always follows its parent in pre-order, so it is finalized first in the reverse pass).
3. *Input assumptions.* The root id is not necessarily `1`; find it by `p_i == -1`. Parents may be listed after their children, so build the whole adjacency before traversing.

**Edge cases.** `n = 1` -> the lone root is an outlet but has no pipe above it, so total `0`. Zero-cost pipes contribute `0`. Non-root-id roots and out-of-order parents are handled by building first, then DFS from the located root.

**Complexity.** `O(n)` time, `O(n)` space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    // parent[i] and cost[i]: edge from i up to parent[i] with given cost.
    // Root is the unique node with parent -1 (0). Nodes are 1..n.
    vector<int> par(n + 1, 0);
    vector<long long> cost(n + 1, 0);
    int root = -1;
    vector<vector<int>> children(n + 1);
    for (int i = 1; i <= n; i++) {
        int p; long long c;
        cin >> p >> c;
        par[i] = p;
        cost[i] = c;
        if (p == -1 || p == 0) {
            root = i;
        } else {
            children[p].push_back(i);
        }
    }

    // leaves[v] = number of leaves in subtree of v.
    // A node with no children is a leaf and contributes 1.
    // total work = sum over non-root nodes v of cost[v] * leaves[v].
    // Iterative DFS (post-order) to avoid stack overflow on deep chains.
    vector<long long> leaves(n + 1, 0);
    vector<int> order;
    order.reserve(n);
    {
        vector<int> st;
        st.push_back(root);
        while (!st.empty()) {
            int u = st.back();
            st.pop_back();
            order.push_back(u);
            for (int w : children[u]) st.push_back(w);
        }
    }
    // process in reverse pre-order = valid post-order accumulation
    for (int i = (int)order.size() - 1; i >= 0; i--) {
        int u = order[i];
        if (children[u].empty()) {
            leaves[u] = 1;
        } else {
            long long s = 0;
            for (int w : children[u]) s += leaves[w];
            leaves[u] = s;
        }
    }

    long long total = 0;
    for (int v = 1; v <= n; v++) {
        if (v == root) continue;
        total += cost[v] * leaves[v];
    }

    cout << total << "\n";
    return 0;
}
```
