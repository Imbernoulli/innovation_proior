#include <bits/stdc++.h>
using namespace std;

struct Node {
    long long pre, suf, tot, best;
};

const long long NEG = LLONG_MIN / 4; // sentinel for "no element": best/pre/suf impossible

// Identity for merge: an empty segment. tot = 0 (sums nothing); pre/suf/best = NEG
// because a non-empty subarray cannot be formed from nothing, so it must never win a max.
Node identity() { return Node{NEG, NEG, 0, NEG}; }

Node leaf(long long v) { return Node{v, v, v, v}; }

Node merge(const Node &L, const Node &R) {
    Node res;
    res.tot  = L.tot + R.tot;
    res.pre  = max(L.pre, L.tot + R.pre);
    res.suf  = max(R.suf, R.tot + L.suf);
    res.best = max(max(L.best, R.best), L.suf + R.pre);
    return res;
}

int n, q;
vector<Node> tree;
vector<long long> a;

void build(int node, int lo, int hi) {
    if (lo == hi) { tree[node] = leaf(a[lo]); return; }
    int mid = (lo + hi) / 2;
    build(node * 2, lo, mid);
    build(node * 2 + 1, mid + 1, hi);
    tree[node] = merge(tree[node * 2], tree[node * 2 + 1]);
}

void update(int node, int lo, int hi, int pos, long long val) {
    if (lo == hi) { tree[node] = leaf(val); return; }
    int mid = (lo + hi) / 2;
    if (pos <= mid) update(node * 2, lo, mid, pos, val);
    else update(node * 2 + 1, mid + 1, hi, pos, val);
    tree[node] = merge(tree[node * 2], tree[node * 2 + 1]);
}

Node query(int node, int lo, int hi, int l, int r) {
    if (r < lo || hi < l) return identity();
    if (l <= lo && hi <= r) return tree[node];
    int mid = (lo + hi) / 2;
    if (r <= mid) return query(node * 2, lo, mid, l, r);
    if (l > mid)  return query(node * 2 + 1, mid + 1, hi, l, r);
    return merge(query(node * 2, lo, mid, l, r),
                 query(node * 2 + 1, mid + 1, hi, l, r));
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    if (!(cin >> n >> q)) return 0;
    a.resize(n);
    for (auto &x : a) cin >> x;

    tree.assign(4 * max(n, 1), identity());
    if (n > 0) build(1, 0, n - 1);

    string out;
    for (int k = 0; k < q; k++) {
        int type;
        cin >> type;
        if (type == 1) {
            int i; long long x;
            cin >> i >> x;
            update(1, 0, n - 1, i, x);
        } else {
            int l, r;
            cin >> l >> r;
            Node res = query(1, 0, n - 1, l, r);
            out += to_string(res.best);
            out += '\n';
        }
    }
    cout << out;
    return 0;
}
