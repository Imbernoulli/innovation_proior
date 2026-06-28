# Aliens Trick

## Method

For the reduced interval sequence, the exact-count recurrence is

$$
f_{i,j}=\min_{0\le t<i}\left(
f_{t,j-1}
+(r_{i-1}-l_t+1)^2
-\max(0,r_{t-1}-l_t+1)^2
\right).
$$

The exact-count bases are `f_{0,0}=0`, `f_{i,0}=+\infty` for `i>0`, and `f_{0,j}=+\infty` for `j>0`; clamp `K` to the reduced interval count before solving.

The values `F(k)=f_{n,k}` are discretely convex:

$$
F(k-1)-F(k) \ge F(k)-F(k+1).
$$

Equivalently, the marginal savings from one more interval do not increase. Add a price `C` to every chosen interval and solve the unconstrained problem

$$
G(C)=\min_k(F(k)+Ck).
$$

The prefix recurrence becomes one-dimensional:

$$
g_i=\min_{0\le t<i}\left(g_t+\operatorname{cost}(t,i)+C\right).
$$

After expanding the square, every split `t` contributes a line queried at `x=r[i-1]`:

$$
-2(l_t-1)x + g_t + (l_t-1)^2
-\max(0,r_{t-1}-l_t+1)^2 + C.
$$

Track the interval count with each optimum and break equal penalized values toward the larger count. As `C` grows, the chosen count cannot increase. Binary-search the largest integer `C` whose optimum still uses at least `K` intervals; then the next integer price is already below `K` except in the `K=1` endpoint case. For the adjacent savings `S_x=F(x-1)-F(x)`, this gives `S_K=C`, so recover

$$
F(K)=G(C)-C K.
$$

The recovery subtracts `C*K`, not `C` times the returned count. If the optimum count jumps across `K`, the skipped counts lie on the same linear segment of the convex chain, so evaluating that tangent at `K` is exact.

## Code

```cpp
// IOI 2016 "Aliens" trick: cover n segments with exactly K larger segments,
// minimizing the sum of squared spans minus already-paid overlaps.
// Reads:  first line "n K"; then n lines "r_i c_i" (each point gives segment
//         [min(r_i,c_i), max(r_i,c_i)]). Writes: the minimum cost f(n,K). Uses long long.
#include <bits/stdc++.h>
using namespace std;

using ll = long long;
using i128 = __int128_t;
const i128 INF = (i128(1) << 120);

struct Line {
    ll m = 0;
    i128 b = INF;
    int cnt = 0;
    pair<i128,int> eval(ll x) const {
        if (b >= INF / 2) return {INF, cnt};
        return {(i128)m * x + b, cnt};
    }
};

struct LiChao {
    vector<int> xs;
    vector<Line> st;
    LiChao(vector<int> xs_) : xs(move(xs_)), st(4 * max<size_t>(1, xs.size())) {}

    static bool better(pair<i128,int> a, pair<i128,int> b) {
        if (a.first != b.first) return a.first < b.first;
        return a.second > b.second;
    }

    void add(Line nw, int p, int l, int r) {
        int m = (l + r) >> 1;
        bool lef = better(nw.eval(xs[l]), st[p].eval(xs[l]));
        bool mid = better(nw.eval(xs[m]), st[p].eval(xs[m]));
        if (mid) swap(nw, st[p]);
        if (l == r) return;
        if (lef != mid) add(nw, p << 1, l, m);
        else add(nw, p << 1 | 1, m + 1, r);
    }

    void add(Line nw) {
        if (xs.empty()) return;
        add(nw, 1, 0, (int)xs.size() - 1);
    }

    pair<i128,int> query_idx(int idx, int p, int l, int r) const {
        int x = xs[idx];
        auto res = st[p].eval(x);
        if (l == r) return res;
        int m = (l + r) >> 1;
        auto child = (idx <= m) ? query_idx(idx, p << 1, l, m)
                                : query_idx(idx, p << 1 | 1, m + 1, r);
        return better(child, res) ? child : res;
    }

    pair<i128,int> query(int x) const {
        int idx = lower_bound(xs.begin(), xs.end(), x) - xs.begin();
        return query_idx(idx, 1, 0, (int)xs.size() - 1);
    }
};

i128 sq(ll x) { return (i128)x * x; }

// Sort, split, and drop intervals contained in an earlier kept interval,
// leaving a strictly-increasing endpoint sequence.
vector<pair<int,int>> reduce_intervals(vector<int> row, vector<int> col) {
    vector<pair<int,int>> seg;
    for (int i = 0; i < (int)row.size(); ++i) {
        int l = min(row[i], col[i]);
        int r = max(row[i], col[i]);
        seg.push_back({l, r});
    }
    sort(seg.begin(), seg.end(), [](auto a, auto b) {
        if (a.first != b.first) return a.first < b.first;
        return a.second > b.second;
    });
    vector<pair<int,int>> keep;
    int far = numeric_limits<int>::min();
    for (auto [l, r] : seg) {
        if (r > far) {
            keep.push_back({l, r});
            far = r;
        }
    }
    return keep;
}

ll minimum_cost_exact_k(vector<int> row, vector<int> col, int K) {
    auto seg = reduce_intervals(row, col);
    int n = seg.size();
    if (n == 0 || K <= 0) return 0;
    K = min(K, n);
    vector<int> l(n), r(n);
    for (int i = 0; i < n; ++i) {
        l[i] = seg[i].first;
        r[i] = seg[i].second;
    }

    // overlap(t): cells of split t already paid for by the previous block.
    auto overlap = [&](int t) -> i128 {
        if (t == 0) return 0;
        return sq(max(0LL, (ll)r[t - 1] - l[t] + 1));
    };

    auto sweep = [&](i128 C) -> pair<i128,int> {
        vector<i128> dp(n + 1, INF);
        vector<int> cnt(n + 1, 0);
        LiChao hull(r);

        auto add_split = [&](int t) {
            if (dp[t] >= INF / 2) return;
            Line ln;
            ln.m = -2LL * ((ll)l[t] - 1);
            ln.b = dp[t] + sq((ll)l[t] - 1) - overlap(t) + (i128)C;
            ln.cnt = cnt[t] + 1;
            hull.add(ln);
        };

        dp[0] = 0;
        add_split(0);
        for (int i = 1; i <= n; ++i) {
            auto q = hull.query(r[i - 1]);
            dp[i] = sq(r[i - 1]) + q.first;
            cnt[i] = q.second;
            if (i < n) add_split(i);
        }
        return {dp[n], cnt[n]};
    };

    ll maxSpan = (ll)r.back() - l.front() + 1;
    i128 lo = 0, hi = sq(maxSpan);
    while (lo < hi) {
        i128 mid = (lo + hi + 1) >> 1;
        if (sweep(mid).second >= K) lo = mid;
        else hi = mid - 1;
    }
    auto best = sweep(lo);
    return (ll)(best.first - lo * K);
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, K;
    if (!(cin >> n >> K)) return 0;
    vector<int> row(n), col(n);
    for (int i = 0; i < n; ++i) cin >> row[i] >> col[i];

    cout << minimum_cost_exact_k(row, col, K) << "\n";
    return 0;
}
```

Each price probe is one lower-envelope DP. The compressed Li Chao version above costs `O(n log n)` per probe after the initial sort and containment reduction, and the binary search uses `O(log M)` probes where `M` bounds the largest span length.
