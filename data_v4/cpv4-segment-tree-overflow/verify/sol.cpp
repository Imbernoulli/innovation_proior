#include <bits/stdc++.h>
using namespace std;

int n, q;
vector<long long> tree;   // subtree sums
vector<long long> lazy;   // pending per-element add to push down

void build(const vector<long long> &a, int node, int lo, int hi) {
    if (lo == hi) { tree[node] = a[lo]; return; }
    int mid = (lo + hi) / 2;
    build(a, 2 * node, lo, mid);
    build(a, 2 * node + 1, mid + 1, hi);
    tree[node] = tree[2 * node] + tree[2 * node + 1];
}

// apply an "add v to every element of this node's range" to the node aggregate + its lazy
void applyAdd(int node, int lo, int hi, long long v) {
    tree[node] += (long long)(hi - lo + 1) * v;   // count of elements times v
    lazy[node] += v;
}

void push(int node, int lo, int hi) {
    if (lazy[node] != 0) {
        int mid = (lo + hi) / 2;
        applyAdd(2 * node, lo, mid, lazy[node]);
        applyAdd(2 * node + 1, mid + 1, hi, lazy[node]);
        lazy[node] = 0;
    }
}

// add v to all elements in [ql, qr]
void update(int node, int lo, int hi, int ql, int qr, long long v) {
    if (qr < lo || hi < ql) return;
    if (ql <= lo && hi <= qr) { applyAdd(node, lo, hi, v); return; }
    push(node, lo, hi);
    int mid = (lo + hi) / 2;
    update(2 * node, lo, mid, ql, qr, v);
    update(2 * node + 1, mid + 1, hi, ql, qr, v);
    tree[node] = tree[2 * node] + tree[2 * node + 1];
}

// sum of elements in [ql, qr]
long long query(int node, int lo, int hi, int ql, int qr) {
    if (qr < lo || hi < ql) return 0;
    if (ql <= lo && hi <= qr) return tree[node];
    push(node, lo, hi);
    int mid = (lo + hi) / 2;
    return query(2 * node, lo, mid, ql, qr) + query(2 * node + 1, mid + 1, hi, ql, qr);
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    if (!(cin >> n >> q)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    tree.assign(4 * n, 0);
    lazy.assign(4 * n, 0);
    if (n > 0) build(a, 1, 0, n - 1);

    while (q--) {
        int type;
        cin >> type;
        if (type == 1) {
            int l, r;
            long long v;
            cin >> l >> r >> v;             // 1-indexed inclusive range, add v
            update(1, 0, n - 1, l - 1, r - 1, v);
        } else {
            int l, r;
            cin >> l >> r;                  // 1-indexed inclusive range, query sum
            cout << query(1, 0, n - 1, l - 1, r - 1) << "\n";
        }
    }
    return 0;
}
