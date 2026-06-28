**Problem.** Given a fixed array `a[0..n-1]` of integers (`|a[i]| <= 10^9`) and `q` queries `(l, r, k)`, answer for each: the `k`-th smallest value of the contiguous block `a[l-1 .. r-1]` (positions 1-based, `k` 1-based, `1 <= k <= r - l + 1`). There are no updates. `n, q <= 2*10^5`. Read `n q`, the values, then the `q` triples; print one answer per query.

**Why the obvious approach is too slow.** Re-sorting the slice per query — copy `a[l-1 .. r-1]`, sort, return element `k-1` — is correct (it is the oracle) but `O((r-l) log(r-l))` each. With full-range queries, `q * n log n ~ 2*10^5 * 2*10^5 * 18 ~ 7*10^11` operations, roughly 700 s against a 2 s limit. It re-derives a window's order from scratch every time, discarding the massive overlap between adjacent windows.

**Key idea — persistent segment tree with prefix-version subtraction.** Reduce "k-th smallest in `[l,r]`" to a binary search over a window value-count: the `k`-th smallest is the smallest value `x` with `count(values <= x in [l,r]) >= k`. Build a value-frequency segment tree (over compressed value ranks `[0, m-1]`) for **every prefix**: version `t` counts the first `t` elements. The window-count is then a difference of two prefix versions:

> frequency of any value-subtree `S` inside `[l, r]` = `version_r(S) - version_{l-1}(S)`

because every position `<= l-1` cancels. Descending the value tree with this difference — go left when the left-child window-count is `>= k`, else go right with `k` decremented — lands on the `k`-th smallest in `O(log n)`.

The catch is memory: storing `n+1` full trees is `O(n^2)`. The insight that saves it is **persistence**. Going from prefix `t-1` to `t` inserts exactly one element, which touches only one root-to-leaf path. So version `t` is a new root plus `O(log n)` new nodes that *share* every untouched subtree with version `t-1`. Total memory drops to `O(n log n)` (~`3.6*10^6` nodes), and a query never builds a window tree — it walks versions `r` and `l-1` in lockstep, subtracting counts on the fly. The prefix-version subtraction `r - (l-1)` is the whole trick.

**Pitfalls to get right.**
1. *Version order at the call site.* The descent computes `cnt[left of bigger version] - cnt[left of smaller version]`, so you must pass the smaller prefix (`root[l-1]`) first and the larger (`root[r]`) second. Swapping them negates every count — a `leftCnt` of `-1` is the tell — and the query walks the wrong way. (A trace of `[2,1]` query `1 2 1` returning `2` instead of `1` exposes exactly this.)
2. *Structure sharing in `update`.* Copy *both* children from the previous version first, then overwrite only the child on the descent direction with a freshly built subtree; the other child must keep pointing into the old version. Building both children fresh would defeat persistence and blow the node budget.
3. *Coordinate compression.* Values are signed and up to `±10^9`; map them to dense ranks `[0, m-1]` and decompress the answer rank through `sorted[]`. Duplicates collapse to one rank, which correctly counts multiplicities.
4. *Node budget / types.* `n` insertions times `<= 18` new nodes each needs `MAXNODES ~ n*20`. Counts are `int` (max `n`).

**Edge cases (all verified against a sorted-slice brute over 800+ cases):** `n = 1` and single-position windows `l = r`; all-equal arrays (every order statistic equal); `k = 1` (window minimum) and `k = r-l+1` (window maximum); negatives and `±10^9` magnitudes via compression; the empty-prefix version is node 0 with zero counts so missing children never need special-casing.

**Complexity.** `O((n + q) log n)` time, `O(n log n)` memory. The max-size case (`n = q = 2*10^5`) runs in ~0.5 s and ~50 MB.

**Code.**

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
