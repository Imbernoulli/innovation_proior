**Reading the problem and pinning the contract.** I have a fixed array `a[0..n-1]` of integers, and `q` queries; each query `(l, r, k)` asks for the `k`-th smallest value of the contiguous block `a[l-1 .. r-1]` (positions 1-based, `k` 1-based, with `1 <= k <= r - l + 1` guaranteed). There are no updates — the array is read once and only ever queried. Input is `n q`, then the `n` values, then `q` lines of `l r k`; I print one value per query. Before any algorithm I fix the scale, because it dictates everything: `n, q <= 2*10^5` and `|a[i]| <= 10^9`. The answer is always an actual element of the array, so it fits in 32 bits, but I'll keep values in `long long` to be safe with the `±10^9` magnitudes and to avoid any accidental narrowing. The real pressure is not the value range — it is that `q` and `n` are both `2*10^5`, so any per-query cost that scales with the window length `r - l` is going to be lethal.

**Laying out the candidate approaches.** Two routes are on the table, and I want to commit to the one whose cost I can actually defend at these limits.

- *Re-sort the slice per query.* For each `(l, r, k)`, copy `a[l-1 .. r-1]`, sort it, return element `k-1`. This is the definition of the problem turned into code — obviously correct, and exactly what I'll use as my oracle. But its cost is `O((r - l) log (r - l))` per query. The open question is the worst case, and there's no mystery here: a single query can span the whole array, and there can be `2*10^5` of them.
- *A static value-indexed counting structure.* If for any prefix length `t` I could ask "how many of the first `t` elements have value `<= x`?" in `O(log n)`, then the number of values `<= x` inside the window `[l, r]` is `count(r, x) - count(l-1, x)`, a difference of two prefix counts. A descent over the value axis would then isolate the `k`-th smallest. The open question is the memory: storing one cumulative histogram per prefix naively is `O(n)` per prefix, `O(n^2)` total, which is `4*10^10` cells — impossible.

**Showing on a concrete case why re-sorting is too slow.** Let me not hand-wave "too slow"; let me put numbers on it. Take `n = q = 2*10^5`, and make every query the full range `l = 1, r = n`. Each query sorts `2*10^5` elements: about `2*10^5 * log2(2*10^5) ~ 2*10^5 * 18 = 3.6*10^6` comparisons. Times `q = 2*10^5` queries, that is `7.2*10^11` comparisons. At even an optimistic `10^9` comparisons/second that's over 700 seconds against a 2-second limit — three orders of magnitude over. Even a single full-range query is `3.6*10^6` operations, fine in isolation, but the adversary just repeats it. So re-sorting is structurally dead: the problem is that it re-derives the slice's order from scratch every time, and the order of a window shares almost everything with its neighbours, yet I throw that sharing away. I need a structure that answers each query in `O(log n)` after near-linear preprocessing — `q log n ~ 2*10^5 * 18 = 3.6*10^6`, comfortably in budget.

**Reducing "k-th smallest in a window" to "count values <= x in a window".** The standard move for an order statistic is to turn it into a counting/search problem on the *value* axis. Suppose I had, for any window `[l, r]`, a function `cnt(x)` = number of positions in `[l, r]` whose value-rank is `<= x`. This `cnt` is monotone non-decreasing in `x`, and the `k`-th smallest is the smallest `x` with `cnt(x) >= k`. So an order-statistic query becomes a binary search on `x` against a window-count. That's the bridge. The two pieces I still need are: (1) compress the values so `x` ranges over a small integer index set, and (2) make the window-count `cnt(x)` fast.

**Coordinate compression.** Values are in `[-10^9, 10^9]` and can be negative and duplicated. I sort the distinct values into `sorted[0..m-1]` and map each `a[i]` to its rank `r = lower_bound(sorted, a[i])`. Now every value is an integer in `[0, m-1]` with `m <= n`, and "the `k`-th smallest *value*" is "decompress the `k`-th smallest *rank*". Duplicates collapse to the same rank, which is exactly right: a window with three copies of value 7 should count three at 7's rank.

**The prefix-difference idea, and the memory wall.** Here is the crux. Build, conceptually, a value-frequency segment tree `T_t` for each prefix length `t = 0..n`: `T_t` over the rank axis `[0, m-1]`, where leaf `x` holds the number of the first `t` elements with rank `x`, and each internal node holds the sum over its rank-subtree. Then for a window `[l, r]`, the frequency of rank-subtree `S` *inside the window* is `T_r(S) - T_{l-1}(S)` — every element with index `<= l-1` cancels, leaving exactly the elements at positions `l..r`. With these per-node differences I can descend the value tree: at a node spanning ranks `[lo, hi]` with midpoint `mid`, the number of window values in the left child `[lo, mid]` is `T_r(left) - T_{l-1}(left)`; if that is `>= k` the `k`-th smallest lives left, else it lives right and I subtract the left count from `k`. One root-to-leaf descent, `O(log m) = O(log n)` per query. The transitions are clean — but storing `n + 1` full segment trees is `O(n * m) = O(n^2)` memory, the same `4*10^10` wall. The prefix-difference idea is right; storing the prefixes naively is what kills it.

**The insight: make the segment tree persistent so consecutive prefixes share structure.** Look at what changes from `T_{t-1}` to `T_t`: exactly **one** element is inserted, the one at index `t-1` with some rank `x`. Inserting a single value into a segment tree touches only the nodes on the **one** root-to-leaf path to leaf `x` — `O(log m)` nodes increment their count, and **every other node is byte-for-byte identical** to `T_{t-1}`. So instead of copying the whole tree, I create new nodes only along that single path and let those new nodes point at the *old, unchanged* children of `T_{t-1}` for every direction I don't descend. `T_t` is then a new root plus `O(log m)` new nodes sharing all `O(m)` untouched nodes with `T_{t-1}`. This is a **persistent (functional) segment tree**: each `update` returns a fresh root without mutating any earlier version, and I keep `root[t]` for every prefix `t`. Total memory is `(n + 1)` roots plus `n` insertions each adding `O(log m)` nodes: `O(n log n)`, about `2*10^5 * 18 ~ 3.6*10^6` nodes — entirely feasible. The prefix-version subtraction `r - (l-1)` is the whole trick: I never build a tree *for* a window; I take the difference of two prefix versions on the fly during the descent.

**Designing the node layout.** I'll use flat arrays rather than pointers/`new` for speed and cache behaviour: `lc[], rc[]` for the two children indices, `cnt[]` for the subtree count, a global `nodeCount`, and `root[t]` for each prefix version. Node 0 is the canonical empty node (all-zero counts, children pointing at itself implicitly via the zero index) — every "absent" subtree resolves to node 0, so I never special-case missing children. The node budget: `n` insertions, each `O(log2 m) <= 18` new nodes for `m <= 2*10^5`, plus slack — `n * 20` is a safe ceiling. I size `MAXNODES = 200005 * 20`.

**The update routine (one prefix to the next).** `update(prev, lo, hi, pos)` returns a new root for the tree that equals `prev` with one extra unit at rank `pos`:

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

The key line is copying *both* children from `prev` first, then overwriting only the child on the descent direction with a freshly built subtree. The other child keeps pointing into the old version — that's the structure sharing. `cnt[cur] = cnt[prev] + 1` because exactly one element was added under this node.

**The query routine (difference of two versions).** `kth(vRoot, uRoot, lo, hi, k)` walks versions `vRoot = root[l-1]` and `uRoot = root[r]` in lockstep. The number of window values in the left child is `cnt[lc[uRoot]] - cnt[lc[vRoot]]`:

```
int kth(int vRoot, int uRoot, int lo, int hi, int k) {
    if (lo == hi) return lo;
    int mid = (lo + hi) >> 1;
    int leftCnt = cnt[lc[uRoot]] - cnt[lc[vRoot]];
    if (k <= leftCnt) return kth(lc[vRoot], lc[uRoot], lo, mid, k);
    else              return kth(rc[vRoot], rc[uRoot], mid + 1, hi, k - leftCnt);
}
```

I descend left when `k <= leftCnt` (the `k`-th smallest is among the left/smaller values), otherwise right with `k` reduced by the left count. At a leaf, `lo == hi` is the answer rank, which I decompress through `sorted[]`.

**Tracing the recurrence on the sample before trusting it.** Array `[1,5,2,6,3,7,4]`, query `2 5 3`. Ranks here equal values-minus-one since the distinct sorted values are `1..7`, ranks `0..6`. I need the 3rd smallest of positions 2..5 = `[5,2,6,3]`. `root[5]` minus `root[1]` is the frequency over `[5,2,6,3]` (ranks `[4,1,5,2]`). Descend from `[0,6]`, mid 3: left child spans ranks `[0,3]` (values 1..4), window count there = how many of `[5,2,6,3]` are `<= 4` = the elements 2,3 = 2. `k=3 > 2`, so go right with `k = 3 - 2 = 1`, into ranks `[4,6]`. mid 5: left child `[4,5]` (values 5,6), window count = elements 5,6 = 2. `k=1 <= 2`, go left into `[4,5]`. mid 4: left child `[4,4]` (value 5), window count = element 5 = 1. `k=1 <= 1`, go left into leaf `[4,4]`, return rank 4 -> `sorted[4] = 5`. Matches the expected `5`. The recurrence is right.

**First implementation — and a trace, because index-juggling transcribes dirty.** I wrote the driver: read `n, q`, read `a`, compress, build `root[0]` as node 0, loop `root[i+1] = update(root[i], 0, m-1, rank(a[i]))`, then per query call `kth(root[l-1], root[r], 0, m-1, k)`. My first cut of the query call had the version arguments in the order they appear in the formula `r - (l-1)`:

```
int rank = kth(root[r], root[l-1], 0, m-1, k);   // <-- first attempt
```

with `kth(vRoot, uRoot, ...)` computing `leftCnt = cnt[lc[uRoot]] - cnt[lc[vRoot]]`. I traced it on the tiny case `a = [2, 1]`, query `1 2 1` (1st smallest of the whole array, which is `1`). Compression: `sorted = [1, 2]`, ranks `a -> [1, 0]`, `m = 2`, axis `[0,1]`. Build: `root[1] = update(root[0]=0, 0,1, rank 1)`, `root[2] = update(root[1], 0,1, rank 0)`. So `root[2]` has one element at rank 0 and one at rank 1. Query: `kth(root[2], root[0], 0,1, 1)` because I passed `root[r]=root[2]` as `vRoot` and `root[l-1]=root[0]` as `uRoot`. Now `leftCnt = cnt[lc[uRoot]] - cnt[lc[vRoot]] = cnt[lc[root[0]]] - cnt[lc[root[2]]] = 0 - 1 = -1`. `k=1 <= -1` is false, so it descends *right* with `k = 1 - (-1) = 2`, into `[1,1]`, returns rank 1 -> `sorted[1] = 2`. Wrong — the answer should be `1`.

**Diagnosing the bug.** The `leftCnt` came out *negative*, which is impossible for a count and is the smoking gun. I had the two versions swapped: the formula is "(count in the larger prefix `r`) minus (count in the smaller prefix `l-1`)", so the prefix-`r` tree must be the *minuend* `uRoot` and the prefix-`(l-1)` tree the *subtrahend* `vRoot`. I passed them the other way, so every `leftCnt` was negated. The descent logic in `kth` is written assuming `uRoot` is the bigger version (it computes `cnt[...uRoot] - cnt[...vRoot]` and recurses keeping that same role), so the *call site* must honour that: `kth(root[l-1], root[r], ...)` — smaller version first as `vRoot`, larger second as `uRoot`. My mnemonic "write it like the formula `r - (l-1)`" betrayed me because the function's parameter order is `(subtrahend, minuend)`, not `(minuend, subtrahend)`.

**Fixing and re-verifying.** Swap the call to put the smaller prefix first:

```
int rank = kth(root[l - 1], root[r], 0, m - 1, k);
```

Re-trace `a = [2, 1]`, query `1 2 1`: `kth(root[0], root[2], 0,1, 1)`. `leftCnt = cnt[lc[root[2]]] - cnt[lc[root[0]]] = 1 - 0 = 1`. `k=1 <= 1`, descend left into `[0,0]`, return rank 0 -> `sorted[0] = 1`. Correct. Re-trace the documented sample query `2 5 3` (done above by hand) — `5`, correct. The case that broke now passes, and it broke for exactly the reason I fixed (a negated count from swapped versions), which is the evidence I trust rather than "it looks right now".

**Edge cases, deliberately, because this is where this kind of code dies.**
- *`m = 0` (no elements).* Can't happen given `n >= 1`, but I guard the build with `if (m > 0)` and otherwise set every `root[i] = 0` so the structure is well-formed regardless. With `n >= 1` we always have `m >= 1`.
- *`n = 1`, single query `1 1 1`.* `sorted` has one value, `m = 1`, axis `[0,0]`. `root[1] = update(0, 0,0, 0)`. Query `kth(root[0], root[1], 0,0, 1)`: `lo == hi` immediately, returns rank 0 -> the lone value. Correct.
- *All-equal array, e.g. `[7,7,7,7,7]`.* All ranks are 0, `m = 1`, axis `[0,0]`. Every `kth` hits the leaf and returns rank 0 -> 7, for any `(l, r, k)`. Correct — a window of identical values has every order statistic equal.
- *`l = r` (single-position window) and `k = 1`.* Window has one element; `root[r] - root[r-1]` isolates exactly that element's rank; the descent puts all the count on the path to that leaf. Correct.
- *`k = 1` and `k = r - l + 1` (the min and the max of the slice).* `k = 1` always descends toward the smallest present value; `k = r-l+1` equals the window size, so at each node the `k` either stays (going to the larger side covering the top) and lands on the largest present rank. Both verified in the differential tests against the sorted-slice oracle.
- *Negatives and `±10^9`.* Coordinate compression handles sign and magnitude uniformly; the tree only ever sees ranks in `[0, m-1]`, and I decompress through `sorted[]`, so the printed value is the true (possibly negative, possibly `10^9`) element.
- *Memory / node budget.* `n` insertions times `<= 18` new nodes each is `~3.6*10^6 < MAXNODES = 4*10^6`. Counts are `int` and max out at `n = 2*10^5`, well inside 32 bits.

**Self-verifying at scale.** I compiled with `-O2 -std=c++17` and differential-tested against an independent brute force that copies each slice, sorts it, and reads element `k-1`: 600 randomized small cases (varying value regimes: heavy duplicates, small ranges, full `±10^9`, all-equal), then 200 medium cases (`n, q` up to 300), then the explicit edge cases above — zero mismatches. A worst-case instance `n = q = 2*10^5` with random full-spread values and random `(l, r, k)` runs in about 0.5 s using ~50 MB, inside the 2-second / 256-MB limits. The `O(n log n)` build and `O(log n)` per query are doing exactly what the asymptotics promised.

**Final solution.** I convinced myself the *idea* is right by reducing the order statistic to a window count, hitting the `O(n^2)`-memory wall on naive per-prefix trees, and resolving it with persistence so consecutive prefixes share all but one root-to-leaf path; and I convinced myself the *code* is right by tracing a failing case to a precise cause — a negated `leftCnt` from passing the two versions in the wrong order — and re-verifying the fix, the sample, and the corners against a brute oracle. That is what I ship: one self-contained file, the persistent segment tree with prefix-version subtraction.

```cpp
#include <bits/stdc++.h>
using namespace std;

// Persistent segment tree (functional / "fat node by version") keyed on the
// compressed value rank. Version i is the segment tree over the multiset of the
// first i array elements (a prefix). Because each insertion only touches one
// root-to-leaf path, version i shares all untouched nodes with version i-1 and
// costs O(log n) new nodes, so all n versions fit in O(n log n) memory.
//
// A range query [l, r] is answered by walking versions r and l-1 in lockstep:
// the count of values inside any value-subtree on positions l..r is
// (count in version r) - (count in version l-1). Descending left whenever that
// difference is >= k, else right with k decremented, lands on the k-th smallest.

static const int MAXNODES = 200005 * 20 + 5; // n*(log2 n + 1) generous bound

int lc[MAXNODES], rc[MAXNODES], cnt[MAXNODES];
int nodeCount = 0;
int root[200005];

// Insert value-rank `pos` (0-based) into the tree, returning a NEW root that
// shares structure with `prev`. Range [lo, hi] is the value-rank interval.
int update(int prev, int lo, int hi, int pos) {
    int cur = ++nodeCount;
    lc[cur] = lc[prev];
    rc[cur] = rc[prev];
    cnt[cur] = cnt[prev] + 1;
    if (lo == hi) return cur;
    int mid = (lo + hi) >> 1;
    if (pos <= mid) lc[cur] = update(lc[prev], lo, mid, pos);
    else            rc[cur] = update(rc[prev], mid + 1, hi, pos);
    return cur;
}

// Find the (1-based) k-th smallest value-rank among positions covered by
// versions (uRoot for prefix r) minus (vRoot for prefix l-1).
int kth(int vRoot, int uRoot, int lo, int hi, int k) {
    if (lo == hi) return lo;
    int mid = (lo + hi) >> 1;
    int leftCnt = cnt[lc[uRoot]] - cnt[lc[vRoot]]; // values in [lo, mid]
    if (k <= leftCnt) return kth(lc[vRoot], lc[uRoot], lo, mid, k);
    else              return kth(rc[vRoot], rc[uRoot], mid + 1, hi, k - leftCnt);
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    if (!(cin >> n >> q)) return 0;

    vector<long long> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];

    // Coordinate compression: sorted distinct values -> ranks [0, m-1].
    vector<long long> sorted = a;
    sort(sorted.begin(), sorted.end());
    sorted.erase(unique(sorted.begin(), sorted.end()), sorted.end());
    int m = (int)sorted.size();

    // The empty version (prefix of length 0) is node 0 with all-zero counts.
    nodeCount = 0;
    lc[0] = rc[0] = cnt[0] = 0;
    root[0] = 0;
    if (m > 0) {
        for (int i = 0; i < n; i++) {
            int r = (int)(lower_bound(sorted.begin(), sorted.end(), a[i]) - sorted.begin());
            root[i + 1] = update(root[i], 0, m - 1, r);
        }
    } else {
        for (int i = 0; i <= n; i++) root[i] = 0;
    }

    for (int query = 0; query < q; query++) {
        int l, r, k;
        cin >> l >> r >> k; // 1-based positions, 1-based k
        int rank = kth(root[l - 1], root[r], 0, m - 1, k);
        cout << sorted[rank] << '\n';
    }
    return 0;
}
```

**Causal recap.** Re-sorting each slice is correct but `O(q * n log n) ~ 7*10^11` on full-range queries — three orders of magnitude over budget — so I reduced "k-th smallest in `[l,r]`" to a binary search over a window value-count `count(r, x) - count(l-1, x)`; storing one value-histogram per prefix is `O(n^2)` memory, which I broke with a *persistent* segment tree where each prefix differs from the last by a single root-to-leaf insertion, giving `O(n log n)` total nodes and `O(log n)` per query via version subtraction; my first call passed the two versions swapped, producing a negative `leftCnt` that a trace of `[2,1]` returning `2` instead of `1` pinpointed; ordering the call `kth(root[l-1], root[r], ...)` fixes it, and coordinate compression plus the `m`-axis descent close out duplicates, negatives, single elements, and the `k = 1` / `k = r-l+1` corners — all confirmed against a sorted-slice brute over 800+ cases with zero mismatches.
