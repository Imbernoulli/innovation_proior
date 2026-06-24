#include <bits/stdc++.h>
using namespace std;

/*
  Longest strictly-increasing contiguous run inside a query window [l, r],
  with point assignments. Segment tree of "runs": each node covering a block
  of positions stores
    pre  = length of the longest increasing run starting at the block's left end,
    suf  = length of the longest increasing run ending at the block's right end,
    best = length of the longest increasing run fully inside the block,
    lval = value at the block's left end,
    rval = value at the block's right end,
    len  = number of positions the block covers.
  Merging adjacent blocks L (left) then R (right) glues a run across the seam
  iff L.rval < R.lval, i.e. the seam is a STRICT ascent.
*/

struct Node {
    long long lval, rval;
    int pre, suf, best, len;
};

int n;
vector<long long> a;     // 1-indexed values, a[1..n]
vector<Node> tr;         // segment tree, size 4n

Node makeLeaf(long long v) {
    return Node{v, v, 1, 1, 1, 1};
}

Node merge(const Node &L, const Node &R) {
    Node res;
    res.len  = L.len + R.len;
    res.lval = L.lval;
    res.rval = R.rval;
    bool join = (L.rval < R.lval);          // seam joins only on a strict ascent
    res.pre = L.pre;
    if (L.pre == L.len && join) res.pre = L.len + R.pre;
    res.suf = R.suf;
    if (R.suf == R.len && join) res.suf = R.len + L.suf;
    res.best = max(L.best, R.best);
    if (join) res.best = max(res.best, L.suf + R.pre);
    return res;
}

void build(int node, int lo, int hi) {
    if (lo == hi) { tr[node] = makeLeaf(a[lo]); return; }
    int mid = (lo + hi) / 2;
    build(node*2, lo, mid);
    build(node*2+1, mid+1, hi);
    tr[node] = merge(tr[node*2], tr[node*2+1]);
}

void update(int node, int lo, int hi, int pos, long long val) {
    if (lo == hi) { tr[node] = makeLeaf(val); return; }
    int mid = (lo + hi) / 2;
    if (pos <= mid) update(node*2, lo, mid, pos, val);
    else            update(node*2+1, mid+1, hi, pos, val);
    tr[node] = merge(tr[node*2], tr[node*2+1]);
}

// Returns the Node summarizing the intersection of [lo,hi] with [ql,qr].
// Only descends into children whose range overlaps [ql,qr], so the returned
// Node represents EXACTLY the positions in [lo,hi] that lie inside [ql,qr].
Node query(int node, int lo, int hi, int ql, int qr) {
    if (ql <= lo && hi <= qr) return tr[node];
    int mid = (lo + hi) / 2;
    bool goL = (ql <= mid);
    bool goR = (qr > mid);
    if (goL && goR) {
        Node L = query(node*2, lo, mid, ql, qr);
        Node R = query(node*2+1, mid+1, hi, ql, qr);
        return merge(L, R);
    } else if (goL) {
        return query(node*2, lo, mid, ql, qr);
    } else {
        return query(node*2+1, mid+1, hi, ql, qr);
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int q;
    if (!(cin >> n >> q)) return 0;
    a.assign(n + 1, 0);
    for (int i = 1; i <= n; i++) cin >> a[i];
    tr.assign(4 * (n + 1), Node{});
    build(1, 1, n);

    string out;
    for (int t = 0; t < q; t++) {
        int type;
        cin >> type;
        if (type == 1) {            // set p x  : assign a[p] = x
            int p; long long x;
            cin >> p >> x;
            a[p] = x;
            update(1, 1, n, p, x);
        } else {                    // query l r : longest increasing run in [l,r]
            int l, r;
            cin >> l >> r;
            Node res = query(1, 1, n, l, r);
            out += to_string(res.best);
            out += '\n';
        }
    }
    cout << out;
    return 0;
}
