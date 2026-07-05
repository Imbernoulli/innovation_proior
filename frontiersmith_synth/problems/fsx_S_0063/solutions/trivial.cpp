// TIER: trivial
// Single-type grid tiling: exactly the checker's baseline construction -> ratio ~0.1.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int W, H, n;
vector<vector<pair<int,int>>> shape;
vector<ll> stock;

static inline pair<int,int> tf(int a, int b, int k) {
    int f = k / 4, r = k % 4;
    int x = a, y = b;
    if (f) x = -x;
    for (int i = 0; i < r; i++) { int nx = y, ny = -x; x = nx; y = ny; }
    return {x, y};
}
static vector<pair<int,int>> normCells(int type, int k) {
    vector<pair<int,int>> v;
    int mnx = INT_MAX, mny = INT_MAX;
    for (auto& c : shape[type]) {
        auto p = tf(c.first, c.second, k);
        v.push_back(p); mnx = min(mnx, p.first); mny = min(mny, p.second);
    }
    for (auto& p : v) { p.first -= mnx; p.second -= mny; }
    return v;
}

int main() {
    scanf("%d %d %d", &W, &H, &n);
    shape.assign(n, {}); stock.assign(n, 0);
    for (int i = 0; i < n; i++) {
        int s; ll c; scanf("%d %lld", &s, &c); stock[i] = c;
        for (int j = 0; j < s; j++) { int a, b; scanf("%d %d", &a, &b); shape[i].push_back({a, b}); }
    }
    ll best = 0; int bt = -1, bk = 0, bbw = 1, bbh = 1; ll bcopies = 0, bcols = 1;
    for (int i = 0; i < n; i++) {
        int area = (int)shape[i].size();
        for (int k = 0; k < 8; k++) {
            auto v = normCells(i, k);
            int bw = 0, bh = 0;
            for (auto& p : v) { bw = max(bw, p.first + 1); bh = max(bh, p.second + 1); }
            if (bw > W || bh > H) continue;
            ll cols = W / bw, rows = H / bh;
            ll copies = min(cols * rows, stock[i]);
            ll cov = copies * (ll)area;
            if (cov > best) { best = cov; bt = i; bk = k; bbw = bw; bbh = bh; bcopies = copies; bcols = cols; }
        }
    }
    vector<array<int,4>> out;
    if (bt >= 0) {
        for (ll idx = 0; idx < bcopies; idx++) {
            ll col = idx % bcols, row = idx / bcols;
            out.push_back({bt + 1, bk, (int)(col * bbw), (int)(row * bbh)});
        }
    }
    printf("%d\n", (int)out.size());
    for (auto& o : out) printf("%d %d %d %d\n", o[0], o[1], o[2], o[3]);
    return 0;
}
