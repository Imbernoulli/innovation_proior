# Persistent Segment Tree for Static Range K-th Smallest

## Method

Compress the array values to a dense value domain `1..len`, then store a
value-indexed count segment tree for each array prefix. The tree for prefix `i`
contains the multiset `a[1..i]`. A query range `a[l..r]` is represented by the
difference between the prefix tree at `r` and the prefix tree at `l - 1`.

The query descends through those two roots at the same time. At each internal
value interval, compute

```text
x = sum[left child of root[r]] - sum[left child of root[l - 1]]
```

This is the number of range elements in the lower value half. If `k <= x`, the
answer is in that lower half; otherwise the answer is in the upper half with
rank `k - x`. The reached leaf is the compressed value index.

The prefix trees are stored persistently by path copying. Inserting one value
changes only one root-to-leaf path, so each new prefix root allocates
`O(log n)` nodes and shares every untouched subtree with the previous prefix.
Thus preprocessing uses `O(n log n)` time and memory, and each query costs
`O(log n)`.

## Code

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

## Complexity

Preprocessing performs one path-copying insert per array element, so it takes
`O(n log n)` time and stores `O(n log n)` nodes. Each query performs one paired
descent through the value domain, so it takes `O(log n)` time.
