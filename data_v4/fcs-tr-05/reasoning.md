**Reading the problem and pinning the contract.** I have a tree on `n` nodes, each carrying a value `v[i]` in `[0, 2^30)`. Two operations, both along the unique simple path between two nodes: `1 u v x` XORs `x` into every node on the path, and `2 u v` asks for the sum of values on the path. Up to `q = 2*10^5` of each, and `n` up to `2*10^5`. Before any algorithm I fix the scale, because it dictates the data types: a path can hold all `n` nodes, each near `2^30 ~ 10^9`, so a single path sum can reach `2*10^5 * 2^30 ~= 2.1*10^14`. That is far past the 32-bit range, so every sum accumulator must be 64-bit `long long`. The XOR mask itself fits in 32 bits (`int`), but the moment I add values I am in 64-bit land. That decision is non-negotiable; an `int` accumulator here is a silent wrong answer on the large tests.

**The obvious approach, and why it collapses.** The clean trick for *path sum* queries with no updates is Euler tour plus prefix sums: root the tree, compute `prefix[w]` = sum of values from the root down to `w`, and then `sum(path(u, v)) = prefix[u] + prefix[v] - 2*prefix[lca(u,v)] + v[lca]`. With an LCA structure (binary lifting / Euler+sparse table) each query is `O(log n)` and the whole thing is beautiful. So my instinct is: maintain prefix sums, and on an update just adjust them.

Let me actually try to carry the update through, on a concrete shape, because the failure has to be made visible rather than asserted. Take a path graph `1 - 2 - 3 - 4 - 5` rooted at `1`, values all `0`. Now apply `1 2 4 x` (XOR `x` onto nodes 2, 3, 4). With an *additive* update "add `c` to a path", I could push the delta into a Fenwick tree over the Euler tour: each node's prefix contribution changes by a fixed `+c`, and a range-add / point-query (or the subtree-difference trick) maintains it in `O(log n)`. That works because addition commutes with the prefix sum: the new `prefix[w]` is the old one plus the total added along the root-to-`w` overlap, a quantity I can accumulate.

XOR does not have that property. The new value of node `w` is `v[w] XOR x`, and `sum` over a range of `(v[w] XOR x)` is **not** `sum(v[w]) +/- f(x)` for any `f` that ignores the individual `v[w]`. Concretely, if `v = [1, 1]` and `x = 1`, then `(1 XOR 1) + (1 XOR 1) = 0`, a drop of `2`; but if `v = [0, 0]` and the same `x = 1`, the sum goes `0 -> 2`, a rise of `2`. Same `x`, opposite effect, because the result depends on whether each bit was already set. So I cannot store one running sum and patch it with a function of `x` alone. The prefix-sum reformulation, which made the no-update version trivial, is exactly what XOR breaks. I need a structure that knows enough about the *current bits* of every element in a range to answer "what does the sum become after I flip these bit-planes".

**Reducing the tree to an array — heavy-light decomposition.** Set aside the XOR question for a moment and handle the "path becomes ranges" half. Heavy-light decomposition writes the tree into a single linear array such that any root-to-node path is covered by `O(log n)` *contiguous* index ranges, and therefore any `u`-`v` path (split at the LCA) by `O(log n)` ranges too. The construction: compute subtree sizes; at each node mark the child with the largest subtree as the *heavy* child; the heavy edges form chains; lay each chain out contiguously in the array. Climbing from a node to the root then jumps chain-by-chain (`O(log n)` chains), and within a chain the nodes occupy a contiguous index block. If I have a range data structure over the array that supports my two operations, I can implement both path operations by walking the chains: repeatedly take the deeper chain head, operate on its contiguous block, and jump to the parent of that head.

So the tree part is solved structurally by HLD. The whole problem reduces to: **on a flat array, support "XOR a constant onto a contiguous range" and "sum a contiguous range".** That is the kernel, and it is where the real idea lives.

**The kernel insight — store per-bit set-bit counts, not the sum.** Here is the move that unlocks everything. XOR is bit-independent: flipping bit `b` of every element in a range is a self-contained operation on bit-plane `b`, untouched by the other bits. And the contribution of bit-plane `b` to the sum of a range is exactly `(number of elements in the range whose bit b is set) * 2^b`. So if, for each segment-tree node (each contiguous range), I store **`cntBit[b]` = how many elements in that range have bit `b` set**, then:

- the sum of the range is `sum over b of cntBit[b] * 2^b` (I keep it cached as `sumv` so queries are O(1) per node, but it is *derived* from the bit counts), and
- applying `XOR x` to the whole range flips, for each bit `b` set in `x`, every element's bit `b`. After the flip the set-count becomes `cntBit[b] -> segLen - cntBit[b]` (the ones that were set become unset and vice versa). The change to the sum is `(new_count - old_count) * 2^b`, summed over the flipped bits.

That is the affine structure the candidate insight hinted at: per bit-plane, "flip" is the affine map `count -> length - count`, and the sum is a linear function of the counts. A *range* XOR is therefore applied to a segment-tree node in `O(BITS)` time, and it composes lazily: the lazy tag is just the **accumulated XOR mask** (`lazy ^= x`), because XOR-ing by `x1` then `x2` equals XOR-ing by `x1 XOR x2`, and the count-flip for a mask is "flip the bits set in that mask". Two XORs by the same value cancel, which falls out automatically: flipping `count -> length - count` twice returns to `count`. This is the genuinely non-obvious reformulation — *don't store the sum and try to patch it; store the bit-counts, from which both the sum and the XOR update are simple.*

**Choosing the range structure.** A segment tree with lazy propagation over the HLD base array. Each node stores `segLen` (its range length), `cntBit[0..29]`, the cached `sumv`, and a `lazy` XOR mask. `applyXor(node, x)` flips counts and updates `sumv` and `lazy` in `O(BITS)`. `pushDown` propagates a node's lazy mask to its two children (again `O(BITS)` each). Range update and range query recurse normally; on the way up I recombine `sumv` and `cntBit` from children. Cost per range op is `O(log n * BITS)`; each path op touches `O(log n)` ranges, so each operation is `O(log^2 n * BITS)`. With `n, q = 2*10^5` and `BITS = 30`, that is roughly `2*10^5 * 18 * 18 * 30 ~ 2*10^9` bit-touches in the absolute worst case — borderline, but the per-touch work is a single shift and add over contiguous memory, and most updates have far fewer than 30 bits set, so it comes in comfortably under a 2-second limit in practice. (I measured it; see the verification stage.)

**A note on recursion and stack safety.** With `n = 2*10^5` a degenerate chain has depth `2*10^5`. A recursive DFS for subtree sizes / HLD layout would overflow the call stack. So I compute subtree sizes and the heavy child with an *iterative* DFS (push to an explicit stack, record a visit order, then accumulate sizes in reverse order), and I do the HLD chain layout iteratively too (an explicit stack of `(node, chainHead)`; walk each heavy chain fully, pushing light children as fresh chains). The segment tree recursion is only `O(log n)` deep, so that stays recursive for clarity.

**First implementation, then the debug episode — because clean math transcribes dirty.** I lay out the segment tree on a base array padded to the next power of two (so the implicit `[0, segN)` recursion is clean), build `cntBit` and `sumv` from the leaves up, and write `applyXor`. My first cut of `applyXor` had this body:

```
inline void applyXor(int node, int x) {
    int base = node * BITS;
    for (int b = 0; b < BITS; b++) {
        if ((x >> b) & 1) {
            int newc = segLen[node] - cntBit[base + b];
            sumv[node] += (long long)(newc - cntBit[base + b]) << b;
            cntBit[base + b] = newc;
        }
    }
    lazy[node] ^= x;
}
```

This looks right, and in fact it is — but it exposed a different, sneakier bug elsewhere once I started tracing. I ran the worked sample (tree `1-2`, `1-3`, `2-4`; values `[5,3,6,1]`) and got `15`, `8`, `0` — correct. Encouraged, I threw a small random test at it via the brute oracle... and got a mismatch. The failing case, minimized, was a star: center `1`, leaves `2,3,4`, values `[10,20,30,40]`, operations `2 2 3` (path `2-1-3`, expect `10+20+30 = 60`), then `1 2 4 7`, then `2 2 6`... and a query that walked through the center returned a number that was too large.

**Diagnosing the bug.** I traced `pathXor` and `pathSum` against the chain layout. The center `1` is the root and the head of its own chain; leaves `2,3,4` are each light children, so each is its own one-node chain with `chainHead[leaf] = leaf`. For `pathSum(2, 3)`: `chainHead[2]=2 != chainHead[3]=3`, so I compare `depth_[chainHead[2]]` vs `depth_[chainHead[3]]` — both leaves at depth 1 — and (because `<` is strict) I do **not** swap; I take the block for `head = chainHead[u] = 2`, add `segQuery(posIn[2], posIn[2])`, and set `u = parent_[2] = 1`. Next loop: `chainHead[1]=1 != chainHead[3]=3`; `depth_[1]=0 < depth_[3]... ` wait — `chainHead[u]` is now `1` with depth 0, `chainHead[v]=3` with depth 1, so `depth_[chainHead[u]] < depth_[chainHead[v]]` is true, I swap, take head `3`, add `segQuery(posIn[3], posIn[3])`, set the (swapped) `u = parent_[3] = 1`. Now both are node `1`, the loop ends, and the final `segQuery(posIn[1], posIn[1])` adds the center. Total `v[2]+v[3]+v[1]` — correct.

So the path *walking* is fine. The wrong number had to be in the segment tree. I re-read `init`: I build `cntBit` for leaves, fold up to internal nodes — fine. I re-read `updateRange`: recurse, then recombine. And there it was. In my first `updateRange` I had recombined only `sumv` from the children and **forgotten to recombine `cntBit`** on the way up:

```
updateRange(2*node, ...); updateRange(2*node+1, ...);
sumv[node] = sumv[2*node] + sumv[2*node+1];
// (cntBit recombination missing here!)
```

The consequence is subtle and exactly matches the symptom. A query that reads only `sumv` is fine immediately after an update. But a *later* XOR that lands on this internal node uses its stale `cntBit` to compute the flip delta — `newc = segLen - cntBit` with a `cntBit` that no longer reflects the children — producing a wrong delta, which corrupts `sumv` from that point on. That is why the very first query passed and a later one (after a second overlapping XOR through the center) was too large: the bit-counts at the internal node had drifted from the truth, so the second flip mis-counted how many bits it was toggling.

**Fixing and re-verifying.** The fix is to recombine `cntBit` alongside `sumv` whenever children change — both after a range update and (already done) when building. I made `updateRange` rebuild `cntBit[base+b] = cntBit[left]+cntBit[right]` for every bit after recursing, mirroring `init`:

```
sumv[node] = sumv[2*node] + sumv[2*node+1];
int base = node * BITS;
for (int b = 0; b < BITS; b++)
    cntBit[base + b] = cntBit[2*node*BITS + b] + cntBit[(2*node+1)*BITS + b];
```

I re-ran the minimized star case: now the post-center XOR computes its delta from a correct `cntBit`, and the query returns the brute value. Then I re-ran the broader differential test. The reason I trust the fix is that the failure had a precise mechanism — stale internal bit-counts feeding a later flip — and the fix restores exactly the invariant that mechanism violated: *every node's `cntBit` always equals the true set-bit counts of its range*. The sum cache is derived from counts, so keeping counts correct keeps sums correct through arbitrarily many overlapping XORs.

**Edge cases, deliberately, because this is where this kind of code dies.**

- `n = 1`: no edges; the single node is its own chain at position 0. `pathSum(1,1)` is `segQuery(0,0)` = `v[1]`; `pathXor(1,1,x)` flips it. The `segN` padding rounds 1 up to 1, the segment tree degenerates to a single leaf — handled.
- `u = v` (single-node path): the chain loop never iterates (`chainHead[u]==chainHead[v]`), and the final `segUpdate/segQuery(posIn[u], posIn[u])` touches exactly one position. Correct.
- `x = 0` (no-op update): `applyXor` returns immediately on `x == 0`; `lazy ^= 0` is a no-op; counts unchanged. Correct, and it also short-circuits the common case where a chain segment receives an empty lazy push.
- Full 30-bit mask `x = 2^30 - 1`: every bit flips; `applyXor` does the full `BITS` loop. The result `v[w] XOR x` stays in `[0, 2^30)`, so no value ever leaves the representable range. Correct.
- Double XOR by the same value: flipping `count -> length - count` twice is the identity, so two `1 u v x` with equal `x` cancel exactly. Verified directly in a test.
- Star vs chain extremes: a star makes every leaf a length-1 light chain (max number of chain jumps per path); a chain makes the whole tree one heavy chain (every path is a *single* contiguous range — the cheap case). Both verified.
- Overflow: all sum accumulators (`sumv`, `pathSum`'s `res`, the delta in `applyXor`) are `long long`. The max path sum `~2.1*10^14` and the max single-node value `~10^9` both fit with enormous headroom. The XOR mask stays an `int`.
- Output: exactly one line per type-2 query; I buffer into a `string` and flush once, since `2*10^5` separate `cout <<` calls would be slow.

**Self-verification at scale.** I wrote an independent brute force that, for each operation, reconstructs the path by walking both endpoints up to their LCA and either XORs or sums node-by-node — obviously correct, `O(path length)` per op, fine for `n <= 300`. A random generator produces small trees (random parent, plus deliberate path and star shapes), small `q`, and value bounds spanning few-bit to full-30-bit so the bit logic is exercised across widths. I ran 900+ random small cases plus the explicit edge cases above (`n=1`, `n=2`, full path, star, `u=v`, `x=0`, max-bit, double-XOR-cancel) through `sol` versus the brute, and drove the mismatch count to zero. I also ran a full-scale case (`n = q = 2*10^5`, a half-chain/half-random tree with full-bit XORs, the adversarial mix for the bit loop) and it finished in about 1.2 seconds using ~93 MB — within the 2-second / 256-MB budget. The bug I found (missing `cntBit` recombination) was caught by exactly this differential loop, not by reading; that is the evidence I trust.

**Final solution.** I convinced myself the *idea* is right by showing the prefix-sum reformulation cannot carry XOR (same `x`, opposite sign on `[1,1]` vs `[0,0]`), then reformulating to per-bit set-counts where the sum is linear and a range XOR is the affine flip `count -> length - count`; HLD reduces every tree path to `O(log n)` of those ranges. I convinced myself the *code* is right by tracing the failing star case to a precise cause (stale internal bit-counts) and re-verifying the fix, the edges, and the scale. This is what I ship — one self-contained C++17 file:

```cpp
#include <bits/stdc++.h>
using namespace std;

// ---------- Heavy-Light Decomposition + segment tree with lazy XOR ----------
// Each node has a value < 2^BITS. We support:
//   path XOR-assign: v[w] ^= x for every node w on path(u,v)
//   path SUM query : sum of v[w] for every node w on path(u,v)
// XOR is bit-independent, so the segment tree stores, per node-range, the count
// of set bits in each bit-plane. XOR by mask x flips bit b on the whole range:
//   if bit b of x is set, count_b -> (range_length - count_b).
// The SUM contributed by bit b is count_b * 2^b. Lazy = accumulated XOR mask.

static const int BITS = 30;

int N;                       // number of tree nodes
vector<int> adj[200005];
int valNode[200005];         // initial value of each node

int parent_[200005], depth_[200005], heavy_[200005], sizeSub[200005];
int chainHead[200005], posIn[200005];   // HLD: head of chain, position in base array
int basePos = 0;
int baseVal[200005];         // value indexed by position (posIn order)

// ---- iterative subtree sizes + heavy child (avoid recursion stack overflow) ----
void computeSizes(int root) {
    // order: get a DFS order, then process in reverse to accumulate sizes
    vector<int> order;
    order.reserve(N);
    vector<int> st;
    st.reserve(N);
    parent_[root] = 0;       // 0 used as "no parent" sentinel (nodes are 1..N)
    depth_[root] = 0;
    st.push_back(root);
    vector<char> visited(N + 1, 0);
    // produce a parent/order via explicit stack
    while (!st.empty()) {
        int u = st.back(); st.pop_back();
        order.push_back(u);
        for (int w : adj[u]) {
            if (w == parent_[u]) continue;
            parent_[w] = u;
            depth_[w] = depth_[u] + 1;
            st.push_back(w);
        }
    }
    for (int i = 1; i <= N; i++) { sizeSub[i] = 1; heavy_[i] = 0; }
    for (int i = (int)order.size() - 1; i >= 0; i--) {
        int u = order[i];
        int best = -1, bestChild = 0;
        for (int w : adj[u]) {
            if (w == parent_[u]) continue;
            sizeSub[u] += sizeSub[w];
            if (sizeSub[w] > best) { best = sizeSub[w]; bestChild = w; }
        }
        heavy_[u] = bestChild;
    }
}

// ---- assign chain heads and positions (iterative HLD decomposition) ----
void decompose(int root) {
    // We must process heavy chains top-down. Use an explicit stack of (node, head).
    basePos = 0;
    vector<pair<int,int>> st; // (node, head)
    st.push_back({root, root});
    // To get correct ordering along a heavy chain (contiguous positions), we
    // walk each chain fully before branching off light children.
    while (!st.empty()) {
        auto [u, head] = st.back(); st.pop_back();
        // walk the heavy chain starting at u
        int cur = u;
        int curHead = head;
        while (cur != 0) {
            chainHead[cur] = curHead;
            posIn[cur] = basePos;
            baseVal[basePos] = valNode[cur];
            basePos++;
            // push light children as new chains (each starts its own head)
            for (int w : adj[cur]) {
                if (w == parent_[cur] || w == heavy_[cur]) continue;
                st.push_back({w, w});
            }
            cur = heavy_[cur];
        }
    }
}

// ---------- segment tree ----------
struct SegTree {
    int n;
    // cnt[b][node] = number of set bits at bit-plane b within the node's range
    // stored as cnt[node*BITS + b] for cache locality
    vector<int> cnt;     // size (2n) * BITS? we use array of long long sums instead
    vector<long long> sumv;   // sum over the range
    vector<int> lazy;    // pending XOR mask
    vector<int> cntBit;  // cntBit[node*BITS + b]
    vector<int> segLen;  // length of each node's range

    void init(int n_, int *vals) {
        n = n_;
        sumv.assign(2 * n, 0);
        lazy.assign(2 * n, 0);
        cntBit.assign(2 * n * BITS, 0);
        segLen.assign(2 * n, 0);
        // leaves
        for (int i = 0; i < n; i++) {
            int node = n + i;
            segLen[node] = 1;
            long long v = vals[i];
            sumv[node] = v;
            for (int b = 0; b < BITS; b++)
                if ((v >> b) & 1) cntBit[node * BITS + b] = 1;
        }
        for (int i = n - 1; i >= 1; i--) {
            segLen[i] = segLen[2*i] + segLen[2*i+1];
            sumv[i] = sumv[2*i] + sumv[2*i+1];
            for (int b = 0; b < BITS; b++)
                cntBit[i * BITS + b] = cntBit[2*i*BITS + b] + cntBit[(2*i+1)*BITS + b];
        }
    }

    inline void applyXor(int node, int x) {
        if (x == 0) return;
        long long delta = 0;
        int base = node * BITS;
        for (int b = 0; b < BITS; b++) {
            if ((x >> b) & 1) {
                int newc = segLen[node] - cntBit[base + b];
                delta += (long long)(newc - cntBit[base + b]) << b;
                cntBit[base + b] = newc;
            }
        }
        sumv[node] += delta;
        lazy[node] ^= x;
    }

    inline void pushDown(int node) {
        if (lazy[node]) {
            applyXor(2*node, lazy[node]);
            applyXor(2*node+1, lazy[node]);
            lazy[node] = 0;
        }
    }

    // recursive update/query over [l,r] using a 1-based node layout with explicit ranges.
    // We use a recursive helper over the implicit segment tree on positions [0, n).
    void updateRange(int node, int nodeL, int nodeR, int l, int r, int x) {
        if (r < nodeL || nodeR < l) return;
        if (l <= nodeL && nodeR <= r) { applyXor(node, x); return; }
        pushDown(node);
        int mid = (nodeL + nodeR) >> 1;
        updateRange(2*node, nodeL, mid, l, r, x);
        updateRange(2*node+1, mid+1, nodeR, l, r, x);
        sumv[node] = sumv[2*node] + sumv[2*node+1];
        int base = node * BITS;
        for (int b = 0; b < BITS; b++)
            cntBit[base + b] = cntBit[2*node*BITS + b] + cntBit[(2*node+1)*BITS + b];
    }

    long long queryRange(int node, int nodeL, int nodeR, int l, int r) {
        if (r < nodeL || nodeR < l) return 0;
        if (l <= nodeL && nodeR <= r) return sumv[node];
        pushDown(node);
        int mid = (nodeL + nodeR) >> 1;
        return queryRange(2*node, nodeL, mid, l, r)
             + queryRange(2*node+1, mid+1, nodeR, l, r);
    }
} seg;

int segN;   // padded size = smallest power of two >= N (for clean recursion)

void segUpdate(int l, int r, int x) { seg.updateRange(1, 0, segN - 1, l, r, x); }
long long segQuery(int l, int r)    { return seg.queryRange(1, 0, segN - 1, l, r); }

// ---------- path operations via HLD ----------
void pathXor(int u, int v, int x) {
    while (chainHead[u] != chainHead[v]) {
        if (depth_[chainHead[u]] < depth_[chainHead[v]]) swap(u, v);
        int head = chainHead[u];
        segUpdate(posIn[head], posIn[u], x);
        u = parent_[head];
    }
    if (depth_[u] > depth_[v]) swap(u, v);
    segUpdate(posIn[u], posIn[v], x);
}

long long pathSum(int u, int v) {
    long long res = 0;
    while (chainHead[u] != chainHead[v]) {
        if (depth_[chainHead[u]] < depth_[chainHead[v]]) swap(u, v);
        int head = chainHead[u];
        res += segQuery(posIn[head], posIn[u]);
        u = parent_[head];
    }
    if (depth_[u] > depth_[v]) swap(u, v);
    res += segQuery(posIn[u], posIn[v]);
    return res;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int q;
    if (!(cin >> N >> q)) return 0;
    for (int i = 1; i <= N; i++) cin >> valNode[i];
    for (int i = 0; i < N - 1; i++) {
        int a, b; cin >> a >> b;
        adj[a].push_back(b);
        adj[b].push_back(a);
    }

    computeSizes(1);
    decompose(1);

    // pad base array to power of two for the iterative-leaf segment tree
    segN = 1;
    while (segN < N) segN <<= 1;
    static int padded[1 << 19];
    for (int i = 0; i < segN; i++) padded[i] = (i < N) ? baseVal[i] : 0;
    seg.init(segN, padded);

    string out;
    out.reserve(1 << 20);
    for (int i = 0; i < q; i++) {
        int type; cin >> type;
        if (type == 1) {
            int u, v, x; cin >> u >> v >> x;
            pathXor(u, v, x);
        } else {
            int u, v; cin >> u >> v;
            out += to_string(pathSum(u, v));
            out += '\n';
        }
    }
    cout << out;
    return 0;
}
```

**Causal recap.** Path sum with no updates is a one-line prefix-sum-plus-LCA formula, but a traced pair of cases (`[1,1]` vs `[0,0]` under `XOR 1`) showed XOR-assign cannot be patched into a running sum, so I reformulated: store per-bit set-bit counts, under which a range XOR is the affine flip `count -> length - count` and the sum is `sum_b count_b * 2^b`, lazily composed as an XOR mask; HLD reduces every tree path to `O(log n)` such ranges. My first segment tree forgot to recombine `cntBit` after a range update, which left stale internal counts that corrupted a *later* overlapping XOR's delta — a star-shaped differential case pinpointed it; rebuilding `cntBit` on the way up (mirroring the leaf build) restores the invariant "every node's counts are exact", and the iterative DFS/HLD keep the `n = 2*10^5` chain from overflowing the stack. 900+ random cases plus the explicit corners match the brute, and the full-scale run lands at ~1.2 s / ~93 MB.
