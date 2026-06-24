#include <bits/stdc++.h>
using namespace std;

int n, q;
vector<long long> h;       // current heights
vector<long long> mx;      // segment-tree node maxima

void build(int node, int lo, int hi) {
    if (lo == hi) { mx[node] = h[lo]; return; }
    int mid = (lo + hi) / 2;
    build(2 * node, lo, mid);
    build(2 * node + 1, mid + 1, hi);
    mx[node] = max(mx[2 * node], mx[2 * node + 1]);
}

void pointSet(int node, int lo, int hi, int pos, long long val) {
    if (lo == hi) { mx[node] = val; return; }
    int mid = (lo + hi) / 2;
    if (pos <= mid) pointSet(2 * node, lo, mid, pos, val);
    else            pointSet(2 * node + 1, mid + 1, hi, pos, val);
    mx[node] = max(mx[2 * node], mx[2 * node + 1]);
}

// Within this node's range, count positions that are STRICTLY greater than
// `bound` AND than everything to their left inside the node. Equivalent to:
// number of strict prefix maxima of the node's segment when the running max
// starts at `bound`. O(log n) because we prune any subtree whose max <= bound.
long long countVisible(int node, int lo, int hi, long long bound) {
    if (mx[node] <= bound) return 0;          // whole subtree is dominated
    if (lo == hi) return 1;                    // single element, mx > bound
    int mid = (lo + hi) / 2;
    long long leftCnt = countVisible(2 * node, lo, mid, bound);
    // For the right half the bound is raised by the left half's max: any new
    // prefix maximum in the right half must beat both the external bound and
    // everything in the left half.
    long long newBound = max(bound, mx[2 * node]);
    long long rightCnt = countVisible(2 * node + 1, mid + 1, hi, newBound);
    return leftCnt + rightCnt;
}

// Walk the query range [ql,qr] left to right over the O(log n) canonical
// segments. `bound` is the running prefix maximum accumulated from the query
// segments already consumed to the LEFT; after counting inside a fully covered
// node we raise `bound` by that node's max so the next segment is gated.
long long queryCount(int node, int lo, int hi, int ql, int qr, long long &bound) {
    if (ql <= lo && hi <= qr) {
        long long c = countVisible(node, lo, hi, bound);
        bound = max(bound, mx[node]);
        return c;
    }
    int mid = (lo + hi) / 2;
    long long res = 0;
    if (ql <= mid)  res += queryCount(2 * node, lo, mid, ql, qr, bound);
    if (qr > mid)   res += queryCount(2 * node + 1, mid + 1, hi, ql, qr, bound);
    return res;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    if (!(cin >> n >> q)) return 0;
    h.assign(max(1, n), 0);
    for (int i = 0; i < n; i++) cin >> h[i];
    mx.assign(4 * max(1, n), LLONG_MIN);
    if (n > 0) build(1, 0, n - 1);

    string out;
    for (int i = 0; i < q; i++) {
        int type;
        cin >> type;
        if (type == 1) {
            int p; long long x;
            cin >> p >> x;
            h[p] = x;
            pointSet(1, 0, n - 1, p, x);
        } else {
            int l, r;
            cin >> l >> r;
            long long bound = LLONG_MIN;           // nothing to the left yet
            long long c = queryCount(1, 0, n - 1, l, r, bound);
            out += to_string(c);
            out += '\n';
        }
    }
    cout << out;
    return 0;
}
