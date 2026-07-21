What makes this problem specific is the pair of "no updates" and the twin ceiling `n, q <= 2*10^5`: the array is read once and only ever queried, so I can afford heavy preprocessing, but a query `(l, r, k)` — the `k`-th smallest of the block `a[l-1 .. r-1]`, positions and `k` both 1-based with `1 <= k <= r-l+1` — must not cost anything that scales with the window length `r-l`, because a query can span the whole array and there can be `2*10^5` of them. Values are `±10^9` and can repeat, so they need coordinate compression, but the value *range* is not the pressure; the window length times the query count is. I keep the values in `long long` to avoid any narrowing on the `±10^9` magnitudes, and I read `n q`, the `n` values, then `q` triples `l r k`.

Two routes are on the table. The brute one — copy `a[l-1 .. r-1]`, sort, return element `k-1` — is the definition of the problem turned into code, obviously correct, and exactly what I keep as an oracle. Its cost is the disqualifier. Put numbers on the worst case: `n = q = 2*10^5`, every query the full range. Each sort is about `2*10^5 * log2(2*10^5) ~ 2*10^5 * 18 = 3.6*10^6` comparisons; times `q = 2*10^5` queries that is `7.2*10^11`, roughly 700 seconds against a 2-second limit — three orders of magnitude over. A single full-range query is fine in isolation; the adversary just repeats it. The structural waste is that re-sorting re-derives a window's order from scratch, discarding the enormous overlap between adjacent windows. I need `O(log n)` per query after near-linear preprocessing, so `q log n ~ 3.6*10^6` total, comfortably in budget.

The standard move for an order statistic is to turn it into counting on the *value* axis. For a window `[l, r]`, let `cnt(x)` be the number of positions in `[l, r]` whose value is `<= x`. It is monotone non-decreasing in `x`, and the `k`-th smallest is the smallest `x` with `cnt(x) >= k`. So the query becomes a search on `x` against a window count. Two pieces remain: compress the values so `x` ranges over a small integer index set, and make the window count fast. Compression first — sort the distinct values into `sorted[0..m-1]`, map each `a[i]` to its rank via `lower_bound`, so every value becomes an integer in `[0, m-1]` with `m <= n`. Duplicates collapse to the same rank, which is exactly right: a window with three copies of value 7 should count three at 7's rank.

Now the counting. Build, conceptually, a value-frequency segment tree over the rank axis `[0, m-1]` for each prefix length `t = 0..n`: leaf `x` in version `t` holds how many of the first `t` elements have rank `x`, internal nodes hold subtree sums. The frequency of any rank-subtree `S` *inside* the window `[l, r]` is then `version_r(S) - version_{l-1}(S)`: every element at index `<= l-1` cancels, leaving exactly positions `l..r`. With those per-node differences I can descend the value tree in one pass — at a node whose left child spans the smaller ranks, the window count on the left is `version_r(left) - version_{l-1}(left)`; if that is `>= k` the answer lives left, else it lives right and I subtract the left count from `k`. One root-to-leaf descent, `O(log m) = O(log n)`. The reduction is right; the naive storage is what kills it — `n+1` full trees is `O(n*m) = O(n^2)`, about `4*10^10` cells.

The rescue is that consecutive prefixes barely differ. From version `t-1` to version `t` exactly one element is inserted, and inserting a single value into a segment tree touches only the nodes on the one root-to-leaf path to its leaf — `O(log m)` nodes increment, every other node is byte-for-byte identical to version `t-1`. So instead of copying the whole tree I create new nodes only along that one path and let them point at the *old, unchanged* children for every direction I do not descend. Version `t` is a fresh root plus `O(log m)` new nodes sharing all `O(m)` untouched nodes with version `t-1` — a persistent (functional) segment tree, one retained root per prefix. Total memory is `n` insertions times `O(log n)` new nodes, `O(n log n) ~ 3.6*10^6` nodes, feasible. And I never build a tree *for* a window: the difference `version_r - version_{l-1}` is taken on the fly during the descent.

For speed and cache behaviour I use flat arrays rather than pointer nodes: `lc[]`, `rc[]`, `cnt[]`, a global `nodeCount`, and `root[t]` per prefix. Node 0 is the canonical empty node with zero count and children that resolve back to 0, so every absent subtree is just node 0 and I never special-case a missing child. Budget: `n` insertions, each at most `18` new nodes for `m <= 2*10^5`, plus slack, so `MAXNODES = 200005 * 20` is a safe ceiling.

The update routine returns a new root equal to `prev` with one extra unit at rank `pos`:

```
int update(int prev, int lo, int hi, int pos) {
    int cur = ++nodeCount;
    lc[cur] = lc[prev]; rc[cur] = rc[prev]; cnt[cur] = cnt[prev] + 1;
    if (lo == hi) return cur;
    int mid = (lo + hi) >> 1;
    if (pos <= mid) lc[cur] = update(lc[prev], lo, mid, pos);
    else            rc[cur] = update(rc[prev], mid + 1, hi, pos);
    return cur;
}
```

The load-bearing line is copying *both* children from `prev` first, then overwriting only the child on the descent direction. The other child keeps pointing into the old version — that is the structure sharing; building both children fresh would defeat persistence and blow the budget. `cnt[cur] = cnt[prev] + 1` because exactly one element was added under this node.

The query walks the two versions in lockstep, left-child window count = `cnt[lc[uRoot]] - cnt[lc[vRoot]]`:

```
int kth(int vRoot, int uRoot, int lo, int hi, int k) {
    if (lo == hi) return lo;
    int mid = (lo + hi) >> 1;
    int leftCnt = cnt[lc[uRoot]] - cnt[lc[vRoot]];
    if (k <= leftCnt) return kth(lc[vRoot], lc[uRoot], lo, mid, k);
    else              return kth(rc[vRoot], rc[uRoot], mid + 1, hi, k - leftCnt);
}
```

The one genuine trap here is the *order* of the two versions at the call site. The descent computes `cnt[...uRoot] - cnt[...vRoot]` and recurses keeping that same role, so `uRoot` must be the larger prefix `root[r]` (the minuend) and `vRoot` the smaller prefix `root[l-1]` (the subtrahend). Writing the call "like the formula `r - (l-1)`" gets it backwards, because the parameter order here is `(subtrahend, minuend)`. A swap negates every `leftCnt`, and a negative count is impossible: on `a = [2, 1]`, query `1 2 1` (answer `1`), swapped versions give `leftCnt = cnt[lc[root[0]]] - cnt[lc[root[2]]] = 0 - 1 = -1`, so `k=1 <= -1` fails, the descent goes right, and it returns `sorted[1] = 2`. The correct call is `kth(root[l-1], root[r], 0, m-1, k)`: `leftCnt = 1 - 0 = 1`, `k=1 <= 1` descends left to leaf rank 0, `sorted[0] = 1`.

The advertised corners fall out of the same structure with no special-casing. An all-equal array (and `n = 1`) collapses to `m = 1`: the axis is the single leaf `[0,0]`, so every `(l, r, k)` returns that one value — right, since every order statistic of a constant window is equal. A single-position window `l = r` takes `root[r] − root[r-1]`, isolating one element's rank on the descent. Negatives and `±10^9` never reach the tree, which sees only ranks in `[0, m-1]`; I decompress through `sorted[]` so the printed value is the true element. Counts are `int`, capped at `n = 2*10^5`, well inside 32 bits.

Differential-testing against the sorted-slice brute — hundreds of randomized small cases across value regimes (heavy duplicates, tiny ranges, full `±10^9`, all-equal), medium cases up to `n, q ~ 300`, and the corners above — turns up zero mismatches. A worst-case `n = q = 2*10^5` instance with random full-spread values and random `(l, r, k)` runs in roughly half a second and about 50 MB, inside the 2 s / 256 MB budget. The full self-contained module is in the answer.
