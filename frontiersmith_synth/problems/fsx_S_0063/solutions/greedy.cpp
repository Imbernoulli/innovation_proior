// TIER: greedy
// Area-descending first-fit: scan cells row-major; at each empty cell try the
// largest available piece (4 rotations only), anchoring its bounding-box corner.
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
    vector<char> occ((size_t)W * H, 0);
    vector<ll> rem = stock;
    // order types by area descending
    vector<int> ord(n); iota(ord.begin(), ord.end(), 0);
    sort(ord.begin(), ord.end(), [&](int p, int q){ return shape[p].size() > shape[q].size(); });
    // precompute normalized cells for 4 rotations
    vector<array<vector<pair<int,int>>,4>> pc(n);
    for (int i = 0; i < n; i++) for (int k = 0; k < 4; k++) pc[i][k] = normCells(i, k);

    vector<array<int,4>> out;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            if (occ[(size_t)y * W + x]) continue;
            bool placed = false;
            for (int ti = 0; ti < n && !placed; ti++) {
                int t = ord[ti];
                if (rem[t] <= 0) continue;
                for (int k = 0; k < 4 && !placed; k++) {
                    auto& v = pc[t][k];
                    bool ok = true;
                    for (auto& p : v) {
                        int cx = p.first + x, cy = p.second + y;
                        if (cx >= W || cy >= H || occ[(size_t)cy * W + cx]) { ok = false; break; }
                    }
                    if (!ok) continue;
                    for (auto& p : v) occ[(size_t)(p.second + y) * W + (p.first + x)] = 1;
                    rem[t]--; out.push_back({t + 1, k, x, y}); placed = true;
                }
            }
        }
    }
    printf("%d\n", (int)out.size());
    for (auto& o : out) printf("%d %d %d %d\n", o[0], o[1], o[2], o[3]);
    return 0;
}
