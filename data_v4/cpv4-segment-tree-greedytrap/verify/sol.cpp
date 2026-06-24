#include <bits/stdc++.h>
using namespace std;

// Segment tree node for maximum-subarray-sum queries with point updates.
// For a segment we keep:
//   total = sum of all elements,
//   pre   = best sum of a prefix that may be empty (>= 0),
//   suf   = best sum of a suffix that may be empty (>= 0),
//   best  = best sum of a possibly-empty contiguous block (>= 0).
// "may be empty" => every field is >= 0, which encodes the empty selection
// that the problem explicitly allows.
struct Node {
    long long total, pre, suf, best;
};

static int N;
static vector<Node> tree;

Node makeLeaf(long long v) {
    Node nd;
    nd.total = v;
    nd.pre = max(0LL, v);
    nd.suf = max(0LL, v);
    nd.best = max(0LL, v);
    return nd;
}

Node combine(const Node &L, const Node &R) {
    Node nd;
    nd.total = L.total + R.total;
    nd.pre   = max(L.pre, L.total + R.pre);
    nd.suf   = max(R.suf, R.total + L.suf);
    nd.best  = max({L.best, R.best, L.suf + R.pre});
    return nd;
}

void build(int node, int lo, int hi, const vector<long long> &a) {
    if (lo == hi) { tree[node] = makeLeaf(a[lo]); return; }
    int mid = (lo + hi) / 2;
    build(2 * node, lo, mid, a);
    build(2 * node + 1, mid + 1, hi, a);
    tree[node] = combine(tree[2 * node], tree[2 * node + 1]);
}

void update(int node, int lo, int hi, int pos, long long val) {
    if (lo == hi) { tree[node] = makeLeaf(val); return; }
    int mid = (lo + hi) / 2;
    if (pos <= mid) update(2 * node, lo, mid, pos, val);
    else            update(2 * node + 1, mid + 1, hi, pos, val);
    tree[node] = combine(tree[2 * node], tree[2 * node + 1]);
}

// Identity element for an empty range: all zeros (the empty selection).
Node query(int node, int lo, int hi, int l, int r) {
    if (r < lo || hi < l) return Node{0, 0, 0, 0};
    if (l <= lo && hi <= r) return tree[node];
    int mid = (lo + hi) / 2;
    if (r <= mid) return query(2 * node, lo, mid, l, r);
    if (l > mid)  return query(2 * node + 1, mid + 1, hi, l, r);
    return combine(query(2 * node, lo, mid, l, r),
                   query(2 * node + 1, mid + 1, hi, l, r));
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    if (!(cin >> n >> q)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    N = n;
    tree.assign(4 * max(1, n), Node{0, 0, 0, 0});
    if (n > 0) build(1, 0, n - 1, a);

    string out;
    for (int i = 0; i < q; i++) {
        int type;
        cin >> type;
        if (type == 1) {
            // point update: position p (1-indexed) becomes value v
            int p; long long v;
            cin >> p >> v;
            update(1, 0, n - 1, p - 1, v);
        } else {
            // query: maximum-sum contiguous block within [l, r] (1-indexed), empty allowed
            int l, r;
            cin >> l >> r;
            Node res = query(1, 0, n - 1, l - 1, r - 1);
            out += to_string(res.best);
            out += '\n';
        }
    }
    cout << out;
    return 0;
}
