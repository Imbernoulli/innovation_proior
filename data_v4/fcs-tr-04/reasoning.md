I have an undirected tree on `n` nodes and need `S(v)`, the sum of distances from `v` to every other node, for *all* `n` roots at once. Two numbers frame everything before I pick an algorithm. On a path of `n` nodes the endpoint sees `1 + 2 + ... + (n-1) = n(n-1)/2`, which at `n = 2*10^5` is about `2*10^{10}` — past the ~`2.1*10^9` ceiling of a 32-bit `int`, so every distance accumulator has to be 64-bit `long long` or the path tests silently return garbage. And `2*10^5` nodes against a 1-second limit leaves room for roughly linear work, nothing quadratic. Those two constraints — 64-bit sums and near-linear time — decide the rest.

The single-source version is trivial: root anywhere, one DFS, and `S(root)` is the sum of all node depths. The naive all-nodes version just runs that from each root — `O(n)` each, `O(n^2)` total. Concretely `n^2 = 4*10^{10}` at the top end; even at an optimistic `10^9` operations per second that is forty seconds, and with queue overhead and cache misses on a `2*10^5`-node graph it is minutes. That is four orders of magnitude over budget, not a constant I can micro-optimize away — I need `O(1)` amortized per node, which forces *reusing* a neighbour's answer instead of recomputing from scratch.

So the productive question: if I know `S(par)` and `child` is adjacent to `par`, can I get `S(child)` without re-traversing? The root moves exactly one edge, so every node's distance to it changes by exactly `+1` or `-1`. Root at `par`; the edge `par–child` splits the tree into `child`'s subtree (`size[child]` nodes, counting `child` itself) and everything else (`n - size[child]` nodes, including `par`). Sliding the root from `par` to `child`, every node inside `child`'s subtree gets one step closer — its path to the root no longer climbs through the `par–child` edge — contributing `-size[child]`; every node outside gets one step farther, contributing `+(n - size[child])`. Net:

```
S(child) = S(par) + (n - 2 * size[child]).
```

One arithmetic step per node, given the parent's answer and `child`'s subtree size — this is the two-pass rerooting the whole solution hinges on. The sign doubles as a check: if `child`'s subtree is more than half the tree the increment is negative, so moving the root into the bulk of the nodes lowers the sum, exactly as it should.

The formula needs two ingredients against one fixed root (I take node `1`): `S(1)`, the sum of depths, and `size[v]` for every `v`. Both fall out of a single DFS. A second pass then walks outward applying the formula, and it must reach each parent before its child so `S(par)` is ready when I compute `S(child)` — discovery order (root first, every node after its parent) gives exactly that.

Two implementation traps are real here. First, recursion: the worst case is a path of `2*10^5` nodes, i.e. recursion depth `2*10^5`, which overflows the default ~1–8 MB stack long before it finishes — so both passes run iteratively over an explicit stack and a discovery-order array. Second, the DFS re-visit guard. I could skip the edge back to the parent with `w != parent[u]`, but I would rather set an explicit `visited[w]` at push time and never push a visited node: that enters each node into `order` exactly once no matter how the stack interleaves siblings, without my having to case-split on tree shapes, and I need `parent` recorded anyway.

```
st.push_back(1); visited[1] = 1; parent[1] = 0;
while (!st.empty()) {
    int u = st.back(); st.pop_back();
    order.push_back(u);
    for (int w : adj[u])
        if (!visited[w]) { visited[w] = 1; parent[w] = u; st.push_back(w); }
}
```

Each node is pushed exactly once, so `order` holds `n` entries with parents strictly before children. With `order` in hand the rest is bookkeeping: a forward scan sets `depth[u] = depth[parent[u]] + 1` and accumulates `rootSum` into `S(1)`; a backward scan over `order` does `size[parent[u]] += size[u]` (every `size` initialized to `1`, since in reverse order children precede parents); then `ans[1] = rootSum` and a final forward scan applies `ans[u] = ans[parent[u]] + n - 2*size[u]`, with `n` cast to `long long` so the increment never falls back to 32-bit arithmetic.

Running the plan on the given path `1-2-3-4` (expected `[6,4,4,6]`): rooted at `1` the depths `0,1,2,3` give `S(1) = 6`; subtree sizes are `size[2,3,4] = 3,2,1`; then `S(2) = 6 + (4-6) = 4`, `S(3) = 4 + (4-4) = 4`, `S(4) = 4 + (4-2) = 6` — all four match, so the transfer formula and the discovery ordering are right.

Edge cases: `n = 1` has no edges and `S(1) = 0` (distance to itself), which the general path would handle awkwardly on an empty `order`, so I special-case it up front and print `0`. `n = 2`: `S(1) = 1`, `S(2) = 1 + (2 - 2) = 1`. The long path exercises the iterative traversal and the 64-bit sums together. A star gives center `n-1` and each leaf `1 + 2*(n-2)`, both of which the formula reproduces. The output is up to `2*10^5` lines of ~11-digit numbers, so I build one `string` and write it once rather than making `n` separate stream writes.

For confidence I diff the rerooting output against an independent `O(n^2)` BFS-from-every-node oracle over a few hundred random small trees and the adversarial shapes — paths, stars, balanced binary, brooms, double-stars — with nodes relabelled and edges shuffled and randomly oriented, so nothing can lean on input order or on node `1` being special. On the `n = 2*10^5` path the endpoint must land on `n(n-1)/2 = 19999900000`, which only holds if every accumulator stayed 64-bit.
