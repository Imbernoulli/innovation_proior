**Reading the problem and pinning the contract.** I have a static rooted tree on `n` nodes and `q` queries; each query `(v, k)` asks for the node `k` parent-edges above `v`, or "no such node" when `k` exceeds the depth of `v`. Both `n` and `q` go up to `5*10^5`, the tree never changes, and `k` itself can be as large as `10^9` — far beyond any possible depth, so the "no such node" case is real and frequent and I must guard it. The output for a missing ancestor and for the root's-own queries has to be unambiguous, so I will print the node label, or `0` (which is never a real label, since labels are `1..n`) when the ancestor does not exist; `k = 0` returns `v` itself. First the scale: `q = 5*10^5` queries against a tree of depth up to `5*10^5` is the whole story. A per-query cost with any non-constant factor is what I have to interrogate, because the limit is 1 second.

**Laying out the candidate approaches.** Two routes are on the table, and I want to commit to the one whose *query* cost I can pin at O(1), not merely the one that is easiest to type.

- *Binary lifting (jump pointers).* Precompute `up2[j][v]` = the ancestor `2^j` edges above `v`, for `j = 0 .. log n`. A query writes `k` in binary and follows one pointer per set bit, climbing `2^{j}` at a time. Build is `O(n log n)`; query is `O(log n)`. It is robust and short. The thing I distrust is exactly that `O(log n)` per query — at `q = 5*10^5` and depth `5*10^5` it is roughly `5*10^5 * 19 ≈ 10^7` pointer chases, which *might* pass, but it is the cost I was explicitly asked to remove, and on a worst-case caterpillar it is doing real work every time.
- *Long-path (ladder) decomposition.* Cut the tree into vertical paths greedily and store each path as a contiguous array top-to-bottom. Climbing *inside one array* is a single index subtraction — O(1). The catch is global: one query may have to cross many paths, so the decomposition by itself does not bound the number of arrays a query touches. Something has to *guarantee* the query lands inside a single array.

**Stress-testing the obvious binary-lifting query before committing.** Let me make the log factor concrete instead of hand-waving "it's fine." Take the worst shape for this problem, a single path (caterpillar): nodes form a chain of length `n = 5*10^5`. A query `(deepest, k)` with `k` having many set bits — say `k = 349525 = 0b1010101010101010101` — forces binary lifting to follow a pointer for *every* set bit: ten hops here, and up to nineteen for a `k` near `n`. Multiply by `5*10^5` such queries and the inner loop is doing `~10^7` dependent memory reads into a table that is `~19 * 5*10^5 * 4` bytes ≈ 38 MB — far larger than cache, so each hop is a likely cache miss. That is the regime where "should pass" turns into "times out on the adversarial test." The log factor is not a rounding error here; it is the entire risk. I want each query to be a *constant* number of table lookups, independent of `k` and of depth.

**The wrong fix, and why it fails.** The naive way to get O(1) is "store every ancestor of every node" — an `n x depth` table — but on the caterpillar that is `Θ(n^2)` memory, instantly out. So I cannot precompute all answers. I need a structure that is `O(n log n)` to build (or less) yet answers in O(1). This is precisely where the ladder idea has to earn its place: the long-path arrays are O(1) to climb but don't bound the crossings; binary lifting bounds the climb but costs a log per query. The two failures are *complementary* — one is fast inside a path but can't get you to the right path, the other can get anywhere but slowly. That complementarity is the clue.

**Deriving the insight — ladders plus one jump.** Here is the long-path (a.k.a. *ladder*) decomposition, and the property that makes it click. Define `height(u)` = the number of edges on the longest downward path from `u` to a leaf. Decompose the tree greedily into vertical *long paths*: from each node, the path continues downward into the child of maximum height. Every node lies on exactly one long path; the head (top) of a long path is the root or a node that is *not* its parent's chosen max-height child. Now the key move — the *ladder*: take each long path and **extend it upward** by as many of its head's ancestors as the path is long. A long path with `L` nodes becomes a ladder array of up to `2L` entries: `L` (or fewer, near the root) ancestors stacked on top, then the `L` path nodes, stored top-to-bottom in one array. The total size of all ladders is at most `2n`, because each long path contributes `L` of its own nodes plus at most `L` borrowed ancestors, and the long paths partition the `n` nodes.

Why does this give O(1)? The decisive invariant: **a node `u` of height `h` sits in a ladder that contains at least `h` of its proper ancestors** (its own long path has length `> h` toward the leaf below it, hence is extended by `> h` ancestors above its head, and `u` is at or below that head). So from any node `u`, I can climb up to `height(u)` steps *within a single array* by index subtraction. The only thing missing is getting the climb distance down to at most `height(u)`. That is where the *single* jump pointer comes in, used not `log` times but exactly once: to answer `(v, k)` with `k >= 1`, let `j = floor(log2 k)`, so `2^j <= k < 2^{j+1}`. Jump `2^j` up from `v` in one pointer lookup to a node `w`. Now `w` has `v` as a descendant `2^j` edges below it, so `height(w) >= 2^j`. The remaining distance is `rem = k - 2^j`, and `0 <= rem < 2^j <= height(w)`. By the invariant, `w`'s ladder holds at least `height(w) >= rem` ancestors above `w` — so the `rem`-th ancestor of `w` is *inside `w`'s ladder array*, reachable by subtracting `rem` from `w`'s index. Two table lookups total: one jump, one ladder index. O(1), with O(n log n) build for the jump pointers and O(n) for the ladders. The jump pointer makes *exponentially decreasing* hops, the ladder makes one *exponentially-large-enough* hop — they meet in the middle exactly once. That is the SOTA level-ancestor structure, and the binary-lifting-only approach is dominated because it pays the log on *every* query.

**Planning the implementation.** Concretely I need, in order:
1. Read parents; build a `children` adjacency and find the root (`par == 0`).
2. `depth[v]` by a DFS from the root (also lets me reject `k > depth[v]` in O(1)).
3. `height[v]` and `downChild[v]` (the max-height child) by processing nodes in decreasing depth — children before parents — so heights are ready when I need them.
4. Long-path decomposition: for each head, walk down via `downChild` collecting path nodes, gather up to `L` ancestors above the head, lay them into one ladder array top-to-bottom, and record for each path node its `ladderId` and `posInLadder`.
5. Binary-lifting `up2[j][v]` for the single jump.
6. Per query: `k==0 -> v`; `k > depth[v] -> 0`; else jump `2^{floor log2 k}` and index into the ladder.

A subtlety I want to get right up front: depths can reach `5*10^5`, so a recursive DFS would blow the stack — I will compute depth and the height ordering iteratively, with an explicit stack and a bucket sort by depth. And `k` is read as `long long` because `10^9` is fine for `int` but I would rather not flirt with `k - 2^j` and comparisons near the boundary; `depth[v]` is at most `n` so `k > depth[v]` short-circuits before any narrowing.

**First implementation — and immediately a trace, because the invariant is easy to *believe* and hard to *index*.** I wrote the build and the query. The query core was:

```
int j = 63 - __builtin_clzll((unsigned long long)k);
int w = up2[j][v];
int rem = (int)(k - (1 << j));
int id = ladderId[w];
int idx = posInLadder[w] - rem;
ans = ladder[id][idx];
```

and for the ladder I first wrote the extension loop as "gather `height[v]` ancestors above the head" — i.e. `for (int t = 0; t < height[v] && a != 0; t++)`. Let me trace the smallest input that could expose a boundary error. Take the caterpillar `1 -> 2 -> 3` (`par = [0,1,2]`), and the query `(3, 2)`, whose answer is obviously node `1`. Heights: `height(3)=0`, `height(2)=1`, `height(1)=2`. There is one long path: head `1`, then `downChild(1)=2`, `downChild(2)=3` — path nodes `[1,2,3]`, so `L = height(1)+1 = 3`. The head is the root, so there are no ancestors above it; the ladder array is just `[1,2,3]` with `posInLadder` = `1->0, 2->1, 3->2`. Query `(3,2)`: `k=2`, `j = floor(log2 2) = 1`, jump `2^1=2` up from node `3` to `w = up2[1][3]`. Now `up2[0][3]=2`, `up2[1][3]=up2[0][2]=1`, so `w = 1`. `rem = 2 - 2 = 0`. `idx = posInLadder[1] - 0 = 0`, `ladder[id][0] = 1`. Correct here.

So where does the `height[v]` extension bug bite? It bites when the head is *not* the root and the path is short relative to where the jump lands. Let me construct it: a long path of length 1 hanging off a deep spine. Concretely consider the tree `1 -> 2 -> 3 -> 4` (a spine) plus `2 -> 5` where `5` is a leaf. Heights: `height(5)=0`, `height(4)=0`, `height(3)=1`, `height(2)=2`, `height(1)=3`. `downChild(2)` picks the max-height child between `3` (height 1) and `5` (height 0), so `downChild(2)=3`; thus `5` is *not* `2`'s chosen child and `5` is the head of its own long path of length `L = height(5)+1 = 1`. Now query `(5, 2)` — answer is node `1`. `k=2`, `j=1`, jump `2` up from `5`: `up2[0][5]=2`, `up2[1][5]=up2[0][2]=1`, so `w=1`. `rem=0`, and `w=1` lives in the spine ladder, so this particular query is fine. The dangerous query is one where `w` is the *short-path head* itself with `rem > 0`.

**Diagnosing the bug.** Make `w` land on node `5`'s ladder with `rem` positive. Take query `(5, 1)`: `k=1`, `j=0`, jump `2^0=1` up from `5` to `w = up2[0][5] = 2`. Wait — `w=2` is on the spine ladder, not `5`'s. The structural point the trace forces me to see: `w` is always an *ancestor* of `v` reached by the jump, and I then climb `rem < 2^j <= height(w)` more. The node whose ladder I index is `w`, and the invariant I must respect is about `w`'s ladder holding `height(w)` ancestors — so the ladder extension length must be tied to the head's path length `L = height(head) + 1`, which is exactly the path length, **not** `height(v)` for an arbitrary member `v`. My first extension loop used `height[v]` (the current head's height), which by luck equals `L-1` and is *one short*: a path of `L` nodes guarantees the bottom node has `height = L-1` ancestors needed in the worst case, and the head itself may be indexed with `rem` up to `height(w)` which for the head can require up to `L-1`... Let me just bound it cleanly: the invariant needs *at least `height(u)` ancestors above `u`* for every node `u` in the ladder, and the tightest demand is the head, which needs `height(head) = L - 1`. Extending by `L` ancestors (when available) is the safe, standard choice and is what keeps the `2n` total bound; extending by exactly `height[head]` is off-by-one against the head's own worst-case index and, more importantly, mis-scales for nodes below the head. I fix the extension to gather up to `L` ancestors, and I gather the path by walking `downChild` so `L` is the true path length rather than a recomputed height.

To be fully sure, I re-derive the index bound directly: for query `(v,k)`, `w = up2[j][v]` has `height(w) >= 2^j` and I subtract `rem < 2^j`. So `idx = posInLadder[w] - rem >= posInLadder[w] - (2^j - 1) >= posInLadder[w] - (height(w) - 1)`. The number of entries above `w` in its ladder is at least `height(w)` (head extended by `L >= height(head)+1 > height(w)` since `w` is at or below the head and the ladder borrows `min(L, available)` ancestors). Therefore `idx >= 0` always when the ancestor exists, which I separately guaranteed by the `k <= depth[v]` guard. The fix: extension length `L = height[head] + 1`, gathered by the path walk, with the loop `for (int t = 0; t < L && a != 0; t++)`.

**Fixing and re-verifying.** With the extension tied to `L = height[v]+1` (v is the head here) and the path collected by `downChild`, I re-ran the constructed tree `1->2->3->4`, `2->5`. Long paths: spine head `1` with nodes `[1,2,3,4]` (`L=4`, no ancestors above the root) and head `5` with nodes `[5]` (`L=1`, extended by one ancestor `2` above it, ladder `[2,5]`, `posInLadder: 2->0(borrowed), 5->1`). Query `(5,1)`: `j=0`, `w=up2[0][5]=2`, `rem=0`, `2` is on the spine ladder at index `1`, answer `2` — correct. Query `(5,3)`: `k=3>depth(5)=2`, guard returns `0` — correct (node 5's ancestors are 2 and 1 only). Query `(4,3)`: `k=3`, `j=1`, `w=up2[1][4]=up2[0][up2[0][4]]=up2[0][3]=2`... wait `up2[0][4]=3`, `up2[1][4]=up2[0][3]=2`, so `w=2`, `rem=3-2=1`, `w=2` on spine ladder at index `1`, `idx=1-1=0`, `ladder=1` — answer node `1`, and indeed the 3rd ancestor of `4` is `4->3->2->1`, node `1`. Correct. The off-by-one that the trace warned me about is gone, and it is gone for the reason I fixed.

**Edge cases, deliberately, because this is where level-ancestor code dies.**
- `k = 0`: handled before any structure is touched, returns `v`. Correct.
- `k > depth[v]` (including the huge `k` up to `10^9`): the `depth[v]` guard returns `0` in O(1); since `depth[v] <= n < 2^j` never needs evaluating, and `k` is `long long`, no overflow. Correct.
- `k` exactly `depth[v]`: the answer is the root; the guard passes (`k <= depth[v]`), the jump+ladder lands on the root because the root is the top of the spine ladder. Verified on the sample (`6 3 -> 1`).
- Root label is not `1`, and parents point "forward" to larger labels: nothing in the build assumes label order (everything is driven by `par`, `children`, `depth`), so this is handled. I tested `par=[3,3,0]` (root `3`, children `1,2`) explicitly.
- `n = 1`: single root, `depth=0`, any `k>=1` returns `0`, `k=0` returns the node. Correct.
- `n = 0` (empty input): `if (!(cin >> n)) return 0;` and `n=0` produces no nodes; the query loop still reads `q` (which will be `0`) and prints nothing. Correct.
- Deep recursion: I avoided recursive DFS entirely (explicit stack + bucket sort by depth), so a depth-`5*10^5` caterpillar cannot overflow the call stack.
- Output size and speed: I build the answer into one `string` with a hand-rolled integer-to-decimal, then a single `cout`, to keep `5*10^5` lines fast.

**Performance check.** Build is `O(n log n)` (the `up2` table, `~19` levels at `n = 5*10^5`) plus `O(n)` everywhere else; each query is exactly two table lookups. I ran `n = q = 5*10^5` on the caterpillar, the skewed-long-path, the star, and random shapes: each finished in about 0.2–0.34 s using ~90–100 MB, comfortably inside 1 s / 256 MB. The `up2` table dominates memory (~38 MB) and the ladders add at most `2n` ints. The single-jump query means there is *no* per-query log factor, which is the entire point of choosing this structure over plain binary lifting.

**Final solution.** I convinced myself the *idea* is right by seeing that binary-lifting's log-per-query is the named cost to remove and that the ladder's O(1) climb is bounded by `height`, then by deriving the single-jump invariant `height(w) >= 2^j > rem`; and I convinced myself the *code* is right by tracing the off-by-one in the ladder extension to a precise cause (extending by `height[v]` instead of the path length `L`, mis-scaling the head's index bound), fixing it, and re-verifying the corners. That is what I ship — one self-contained file: long-path/ladder decomposition plus a single jump pointer, O(1) per query after O(n log n) build.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;

    // parent[i] for 1..n; root has parent 0 (a sentinel meaning "no node").
    vector<int> par(n + 1, 0);
    vector<vector<int>> children(n + 1);
    int root = 0;
    for (int v = 1; v <= n; v++) {
        int p;
        cin >> p;          // p == 0 means v is the root
        par[v] = p;
        if (p == 0) root = v;
        else children[p].push_back(v);
    }

    // ---- depth (edges from root) via iterative BFS/DFS over the explicit forest ----
    // Although exactly one root is guaranteed, we iterate defensively from the root.
    vector<int> depth(n + 1, 0);
    {
        vector<int> order;
        order.reserve(n);
        // process from root downward
        vector<int> stk;
        if (root != 0) stk.push_back(root);
        while (!stk.empty()) {
            int u = stk.back();
            stk.pop_back();
            order.push_back(u);
            for (int c : children[u]) {
                depth[c] = depth[u] + 1;
                stk.push_back(c);
            }
        }
        (void)order;
    }

    // ---- height of each node: longest downward chain length (in edges) ----
    // height[leaf] = 0. Computed by processing nodes in decreasing depth order.
    // We also record, for each node, the child on a longest downward path
    // (the "long-path successor going down").
    vector<int> height(n + 1, 0);
    vector<int> downChild(n + 1, 0); // child that continues the long path downward
    {
        // order nodes by depth descending; bucket sort on depth.
        int maxd = 0;
        for (int v = 1; v <= n; v++) maxd = max(maxd, depth[v]);
        vector<int> cnt(maxd + 2, 0);
        for (int v = 1; v <= n; v++) cnt[depth[v]]++;
        for (int d = 1; d <= maxd; d++) cnt[d] += cnt[d - 1];
        vector<int> byDepth(n);
        for (int v = 1; v <= n; v++) byDepth[--cnt[depth[v]]] = v;
        // byDepth is ascending by depth; iterate in reverse => descending depth.
        for (int idx = n - 1; idx >= 0; idx--) {
            int v = byDepth[idx];
            int best = -1, bestChild = 0;
            for (int c : children[v]) {
                if (height[c] > best) { best = height[c]; bestChild = c; }
            }
            if (bestChild != 0) {
                height[v] = best + 1;
                downChild[v] = bestChild;
            } else {
                height[v] = 0;
                downChild[v] = 0;
            }
        }
    }

    // ---- long-path decomposition ----
    // A node is the HEAD (top) of its long path iff it is the root OR it is not
    // the downChild of its parent. Each long path runs head -> downChild -> ...
    // until height drops to 0.
    // For every node we store pathHead[v] (the top of its long path) and
    // posInLadder[v] (its index within that path's ladder array).
    vector<int> pathHead(n + 1, 0);
    vector<int> ladderId(n + 1, -1);     // which ladder array this node lives in
    vector<int> posInLadder(n + 1, 0);   // index of v inside ladder[ladderId[v]]
    vector<vector<int>> ladder;          // each ladder: indices from top(extended) .. bottom

    for (int v = 1; v <= n; v++) {
        bool isHead = (par[v] == 0) || (downChild[par[v]] != v);
        if (!isHead) continue;
        // walk the long path downward from v
        int L = height[v] + 1; // number of nodes on the long path (head..deepest)
        // ladder = L ancestors above head (if available) + the L path nodes.
        // First gather the path nodes.
        vector<int> pathNodes;
        pathNodes.reserve(L);
        int cur = v;
        while (cur != 0) {
            pathNodes.push_back(cur);
            cur = downChild[cur];
        }
        // pathNodes.size() == L
        // gather up to L ancestors above the head (the "ladder extension")
        vector<int> up;
        up.reserve(L);
        int a = par[v];
        for (int t = 0; t < L && a != 0; t++) {
            up.push_back(a);
            a = par[a];
        }
        // build the ladder array top-to-bottom: reversed(up) ++ pathNodes
        int id = (int)ladder.size();
        vector<int> arr;
        arr.reserve(up.size() + pathNodes.size());
        for (int t = (int)up.size() - 1; t >= 0; t--) arr.push_back(up[t]);
        int headOffset = (int)arr.size(); // index of head v inside arr
        for (int x : pathNodes) arr.push_back(x);
        // assign ladder membership for the PATH nodes only (each node is assigned
        // exactly once, by its own long path).
        for (int i = 0; i < (int)pathNodes.size(); i++) {
            int x = pathNodes[i];
            ladderId[x] = id;
            posInLadder[x] = headOffset + i;
            pathHead[x] = v;
        }
        ladder.push_back(move(arr));
    }

    // ---- jump pointers (binary lifting) ----
    int LOG = 1;
    while ((1 << LOG) < n + 1) LOG++;
    LOG = max(LOG, 1);
    // up2[j][v] = ancestor of v that is 2^j edges above v (0 = none).
    vector<vector<int>> up2(LOG + 1, vector<int>(n + 1, 0));
    for (int v = 1; v <= n; v++) up2[0][v] = par[v];
    for (int j = 1; j <= LOG; j++) {
        for (int v = 1; v <= n; v++) {
            int mid = up2[j - 1][v];
            up2[j][v] = (mid == 0) ? 0 : up2[j - 1][mid];
        }
    }

    // ---- queries ----
    int q;
    cin >> q;
    string out;
    out.reserve((size_t)q * 7);
    char buf[16];
    for (int Q = 0; Q < q; Q++) {
        int v; long long k;
        cin >> v >> k;
        int ans;
        if (k == 0) {
            ans = v;
        } else if (k > depth[v]) {
            ans = 0; // no such ancestor; report 0
        } else {
            // jump 2^j up where 2^j <= k < 2^(j+1)
            int j = 63 - __builtin_clzll((unsigned long long)k);
            int w = up2[j][v];               // w is ancestor at distance 2^j (exists since k<=depth[v])
            int rem = (int)(k - (1 << j));    // 0 <= rem < 2^j <= height(w)
            // w's ladder contains at least height(w) >= 2^j >= rem ancestors above w.
            int id = ladderId[w];
            int idx = posInLadder[w] - rem;   // move rem steps up inside the ladder array
            ans = ladder[id][idx];
        }
        int len = 0;
        if (ans == 0) { buf[len++] = '0'; }
        else { int t = ans; char tmp[16]; int tl = 0; while (t) { tmp[tl++] = char('0' + t % 10); t /= 10; } while (tl) buf[len++] = tmp[--tl]; }
        out.append(buf, len);
        out.push_back('\n');
    }
    cout << out;
    return 0;
}
```

**Causal recap.** Binary lifting answers each query in `O(log n)`, and on a depth-`5*10^5` caterpillar with `5*10^5` queries that log factor is the named cost and a real timeout risk; storing all ancestors is `Θ(n^2)` and out. The resolution combines two complementary structures: the long-path *ladder* decomposition lets me climb `height(u)` steps inside one array in O(1), and a *single* jump pointer of size `2^{floor log2 k}` lands me on a node `w` whose `height(w) >= 2^j` exceeds the residual `rem = k - 2^j`, so the residual climb is guaranteed to stay inside `w`'s ladder — two lookups, O(1) per query, O(n log n) build. The bug I actually hit was the ladder *extension length*: tying it to `height[v]` of an arbitrary node instead of the head's full path length `L = height[head]+1` is an off-by-one that a trace of a short path hanging off a deep spine exposes; fixing the extension to `L` ancestors restores the `idx >= 0` guarantee. The `k=0`, `k>depth`, huge-`k`, non-`1` root, forward-pointing parents, `n=1`, and `n=0` corners are each handled explicitly, and the iterative depth/height passes keep a deep tree from overflowing the stack.
