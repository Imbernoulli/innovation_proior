I start with the fixed array and the query I have to answer: in `a[l..r]`, return the `k`-th smallest value. If I copy that slice, sort it, and index into it, the answer is obviously right, including duplicates, but one wide query costs `O(n log n)`. With many wide queries, I am paying for the same overlapping elements again and again. I need the query to reuse preprocessing from the fixed array.

The order-statistic part feels more tractable if I stop thinking about positions and think about values. Suppose I compress all distinct array values into indices `1..self.len` and keep a segment tree over that value domain, where a node stores how many inserted elements fall into its value interval. If the tree already held the exact multiset of some range, could I find the `k`-th smallest cheaply? At the root, count the items in the left value half. If `k` is no larger than that count, the answer is among those left-half items, so I keep looking left with the same `k`; otherwise the answer is to the right, and I have stepped over exactly that many smaller items, so I look right with `k` reduced by that count. The leaf I reach is a compressed value index, and `self.vals[index - 1]` recovers the real value. So a single tree holding `a[l..r]` would let me answer in `O(log self.len)`. The trouble is producing that tree per query without re-scanning the range.

For a static array, a range is the difference of two prefixes, and that might carry over to the counts. Let `T_i` be a value-count tree holding the multiset `a[1..i]`. The count of `a[l..r]` elements in a value interval `[lo, hi]` should be `count_{T_r}[lo,hi] - count_{T_{l-1}}[lo,hi]`, because `a[1..r]` minus `a[1..l-1]` is exactly `a[l..r]` as a multiset and counting is additive over disjoint multisets. I do not fully trust that until I see it on numbers, so I will hold it as the working hypothesis and check it on the concrete example below before committing. If it holds, the order-statistic descent never needs the difference tree materialized: walking `T_r` and `T_{l-1}` in lockstep, the number of range elements in the left value half is `sum[left child of T_r] - sum[left child of T_{l-1}]`. If `k` fits in that, descend into both left children; otherwise descend into both right children with `k` reduced by that left count. The sign must be later prefix minus earlier prefix, so the paired query receives `u = root[l - 1]` and `v = root[r]` and computes `self.sum[self.ls[v]] - self.sum[self.ls[u]]`.

So the cost now rests entirely on having all the prefix trees `T_0..T_n`. The naive route is to store them independently, but that is `O(n)` trees times `O(self.len)` nodes each, and since `self.len` can be as large as `n`, that is `O(n^2)` memory — unusable at competitive sizes. What rescues it is that consecutive prefixes barely differ: moving from `T_{i-1}` to `T_i` inserts only `a[i]`. In a value-count segment tree one leaf goes up by one, and that bump propagates only up the ancestors of that single leaf. Every subtree hanging off that one root-to-leaf path is bit-for-bit identical before and after the insert.

That identity is the lever: the new tree only needs to own the changed path and can point at the old tree for everything else. The usual implicit heap layout where children are `2 * node` and `2 * node + 1` cannot express sharing, because a node's identity is its array position, so two trees cannot reuse the same physical subtree. I need explicit node ids and child links instead — which is exactly what `self.sum`, `self.ls`, `self.rs` give as a node pool: a node id maps to its stored count and its two child ids. To insert into an old node `prev`, allocate a fresh node `cur`, set its count to `self.sum[prev] + 1`, and copy both child links from `prev`. Then recurse only into the half holding the inserted compressed value, overwriting that one child link with the freshly returned child; the other child link still points into the old subtree. Because every write lands in a newly allocated node, every older root stays valid and unmodified.

That yields one root per prefix: `root[0]` is the empty tree, and `root[i]` is the root returned after inserting `a[i-1]` into `root[i-1]`. I never build an explicit all-zero tree. Node `0` doubles as every absent zero-count subtree, since `self.sum[0]`, `self.ls[0]`, `self.rs[0]` are all zero, so following a missing link reads consistent zeros. Each insert creates one node per level plus the leaf, i.e. `O(log self.len)` new nodes per element and `O(n log n)` total.

Before I trust any of this I want to walk it on real numbers, since three separate claims are riding on the same machinery: the prefix-difference identity, the lockstep descent, and the subtree sharing. Take `a = [5, 1, 3]`. The distinct values sorted are `[1, 3, 5]`, so the compressed map is `1 -> 1`, `3 -> 2`, `5 -> 3`, and the value domain is `[1, 3]` with `mid = 2`. Building the prefix roots and dumping the node pool, I get `root = [0, 1, 3, 6]` and these nodes (id: sum, ls, rs):

```text
node 0: 0, 0, 0        (the shared empty node)
node 1: 1, 0, 2        root[1], holds {5}
node 2: 1, 0, 0        leaf for value index 3
node 3: 2, 4, 2        root[2], holds {5,1}
node 4: 1, 5, 0        interval [1,2] of root[2]
node 5: 1, 0, 0        leaf for value index 1
node 6: 3, 7, 2        root[3], holds {5,1,3}
node 7: 2, 5, 8        interval [1,2] of root[3]
node 8: 1, 0, 0        leaf for value index 2
```

First the sharing claim. `root[1] = 1` has `rs = 2`, and `root[2] = 3` also has `rs = 2` — the same physical node, reused untouched, because inserting `1` (index 1, the left half) never reaches the right subtree. Meanwhile the left child changed from `0` to `4`. So one insert really did copy only the left path and share the right subtree, as the count argument promised.

Now the identity, on `a[2..3] = [1, 3]` over the lower value half `[1, 2]`. From `root[3] = 6`: its left child is node `7` with sum `2`. From `root[1] = 1`: its left child is node `0` with sum `0`. The difference is `2 - 0 = 2`, and indeed `a[2..3]` has two elements (`1` and `3`) with compressed index in `[1, 2]`. The prefix subtraction reproduces the true range count, so the hypothesis survives the check.

Finally the full descent, for `kth(2, 3, 1)` — the smallest of `[1, 3]`, which should be `1`. I take `u = root[1] = 1`, `v = root[3] = 6`, interval `[1, 3]`. Left children are `ls[u] = 0` (sum 0) and `ls[v] = 7` (sum 2), so `x = 2`. Since `k = 1 <= 2`, descend left into `[1, 2]` with `u = 0`, `v = 7`. There the left children are `0` (sum 0) and `5` (sum 1), so `x = 1`; `k = 1 <= 1`, descend left into the leaf `[1, 1]`, returning compressed index `1`, value `self.vals[0] = 1`. Correct.

And `kth(1, 3, 2)` — the median of `[5, 1, 3]`, which is `3`. Here `u = root[0] = 0`, `v = root[3] = 6`. Left children `0` (sum 0) and `7` (sum 2), `x = 2`; `k = 2 <= 2`, descend left into `[1, 2]` with `v = 7`. Now left children `0` (sum 0) and `5` (sum 1), `x = 1`; this time `k = 2 > 1`, so descend right into `[2, 2]` with `k - x = 1`, reaching leaf index `2`, value `self.vals[1] = 3`. Correct, and worth noting the path was left-then-right rather than straight: compressed index `2` lives in the left subtree of the `[1, 3]` split, not the right, so the lockstep `x` subtraction at the `[1, 2]` node is doing real work in skipping the one smaller element. To be confident this is not a coincidence of one tiny array, I also ran the build-and-query against a brute-force "sort the slice and index" over a couple thousand random small arrays and saw no mismatch.

With the identity and the descent both checked, the implementation follows the invariants directly: `_insert` returns a new root after copying one path; `build` stores one root per prefix; `_query` walks two roots in lockstep on the difference of their left counts; `kth` maps the reached compressed index back to the original value.

```cpp
// Reads: n q, then n array values, then q queries (l, r, k);
// prints, one per line, the k-th smallest value in a[l..r] (1-based l, r, k).
#include <bits/stdc++.h>
using namespace std;

// Persistent (value-indexed count) segment tree stored in parallel arrays.
// One persisted version per array prefix: root[i] holds the multiset a[1..i].
// A query window a[l..r] is the node-wise difference of root[r] and root[l-1].
static vector<long long> tsum;   // count stored at a node
static vector<int> ls, rs;       // child node ids (node 0 is the shared empty node)

static int newNode(long long s, int l, int r) {
    tsum.push_back(s);
    ls.push_back(l);
    rs.push_back(r);
    return (int)tsum.size() - 1;
}

// Insert value-index k into the version rooted at prev; return the new root.
// Copies only the root-to-leaf path; every untouched child stays shared.
static int insertNode(int k, int l, int r, int prev) {
    int cur = newNode(tsum[prev] + 1, ls[prev], rs[prev]);
    if (l == r) return cur;
    int mid = (l + r) >> 1;
    if (k <= mid) ls[cur] = insertNode(k, l, mid, ls[cur]);
    else          rs[cur] = insertNode(k, mid + 1, r, rs[cur]);
    return cur;
}

// u = root[l-1], v = root[r]; window counts are v minus u, node by node.
static int queryNode(int u, int v, int l, int r, long long k) {
    if (l == r) return l;
    int mid = (l + r) >> 1;
    long long x = tsum[ls[v]] - tsum[ls[u]];   // window items in the lower value half
    if (k <= x) return queryNode(ls[u], ls[v], l, mid, k);
    return queryNode(rs[u], rs[v], mid + 1, r, k - x);
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    if (!(cin >> n >> q)) return 0;

    vector<long long> a(n);
    for (int i = 0; i < n; ++i) cin >> a[i];

    // Coordinate compression: dense value domain [1, len].
    vector<long long> vals(a.begin(), a.end());
    sort(vals.begin(), vals.end());
    vals.erase(unique(vals.begin(), vals.end()), vals.end());
    int len = (int)vals.size();
    auto getid = [&](long long x) {
        return (int)(lower_bound(vals.begin(), vals.end(), x) - vals.begin()) + 1;
    };

    // Node 0 is the shared empty node; root[0] = empty version.
    tsum.assign(1, 0);
    ls.assign(1, 0);
    rs.assign(1, 0);
    tsum.reserve((size_t)1 + (size_t)n * (33 - __builtin_clz(len > 1 ? len : 1)));
    ls.reserve(tsum.capacity());
    rs.reserve(tsum.capacity());

    vector<int> root(n + 1, 0);
    if (len > 0) {
        for (int i = 1; i <= n; ++i)
            root[i] = insertNode(getid(a[i - 1]), 1, len, root[i - 1]);
    }

    string out;
    for (int i = 0; i < q; ++i) {
        int l, r;
        long long k;
        cin >> l >> r >> k;
        int idx = queryNode(root[l - 1], root[r], 1, len, k);
        out += to_string(vals[idx - 1]);
        out += '\n';
    }
    cout << out;
    return 0;
}
```
