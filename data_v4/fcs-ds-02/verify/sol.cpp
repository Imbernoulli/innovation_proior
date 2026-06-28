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
