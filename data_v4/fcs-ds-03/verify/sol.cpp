#include <bits/stdc++.h>
using namespace std;

/*
  dp[0] = 0
  dp[i] = c[i] + min_{0 <= j < i} ( dp[j] + b[j] * a[i] )   for i = 1..n
  answer = dp[n]

  Each j contributes the LINE  y = b[j] * X + dp[j]  evaluated at X = a[i].
  We need the minimum over a set of lines at a query point => lower envelope.
  Because lines are inserted (one per j) interleaved with queries (one per i),
  and the slopes b[j] and query points a[i] are NOT sorted/monotone in general
  (values may be negative), the robust SOTA structure is the Li Chao tree built
  over the fixed set of query coordinates = the distinct values of a[1..n].
*/

const long long INF = (long long)4e18;

struct Line {
    long long m, c;            // y = m*X + c
    bool valid;
    long long eval(long long x) const { return m * x + c; }
};

int LN;                        // number of distinct query coordinates (leaves)
vector<long long> xs;          // sorted distinct query coordinates
vector<Line> tree;             // Li Chao segment tree over [0, LN-1] index space

// Insert line `nw` into the node covering [l, r].
void insert(int node, int l, int r, Line nw) {
    if (!nw.valid) return;
    int mid = (l + r) >> 1;
    Line &cur = tree[node];
    if (!cur.valid) { cur = nw; return; }
    // Compare at the midpoint coordinate xs[mid].
    bool leftBetter  = nw.eval(xs[l])   < cur.eval(xs[l]);
    bool midBetter   = nw.eval(xs[mid]) < cur.eval(xs[mid]);
    if (midBetter) swap(cur, nw);      // keep the better line at the midpoint
    if (l == r) return;
    // Push the loser down to the side where it can still win.
    if (leftBetter != midBetter) insert(node << 1,     l,       mid, nw);
    else                         insert(node << 1 | 1, mid + 1, r,   nw);
}

// Minimum y over all inserted lines at coordinate index p (query point xs[p]).
long long query(int node, int l, int r, int p) {
    Line &cur = tree[node];
    long long res = cur.valid ? cur.eval(xs[p]) : INF;
    if (l == r) return res;
    int mid = (l + r) >> 1;
    if (p <= mid) res = min(res, query(node << 1,     l,       mid, p));
    else          res = min(res, query(node << 1 | 1, mid + 1, r,   p));
    return res;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    if (n == 0) { cout << 0 << "\n"; return 0; }   // no stations: dp[0]=0 is the answer

    vector<long long> a(n + 1), b(n), c(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];      // a[1..n]
    for (int j = 0; j < n; j++)  cin >> b[j];      // b[0..n-1]
    for (int i = 1; i <= n; i++) cin >> c[i];      // c[1..n]

    // Coordinate-compress the query points (the distinct values of a[1..n]).
    xs.assign(a.begin() + 1, a.end());
    sort(xs.begin(), xs.end());
    xs.erase(unique(xs.begin(), xs.end()), xs.end());
    LN = (int)xs.size();

    // index of coordinate v among xs (v is guaranteed present)
    auto idx = [&](long long v) -> int {
        return int(lower_bound(xs.begin(), xs.end(), v) - xs.begin());
    };

    tree.assign(4 * LN, Line{0, 0, false});

    vector<long long> dp(n + 1, INF);
    dp[0] = 0;
    // Insert the line for j = 0:  y = b[0]*X + dp[0].
    insert(1, 0, LN - 1, Line{b[0], dp[0], true});

    for (int i = 1; i <= n; i++) {
        long long best = query(1, 0, LN - 1, idx(a[i]));   // min_j (dp[j] + b[j]*a[i])
        dp[i] = c[i] + best;
        // Now dp[i] is final; if i can serve as a future start (j = i < n), insert its line.
        if (i < n) insert(1, 0, LN - 1, Line{b[i], dp[i], true});
    }

    cout << dp[n] << "\n";
    return 0;
}
