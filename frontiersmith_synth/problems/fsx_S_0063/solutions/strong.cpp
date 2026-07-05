// TIER: strong
// Cover-this-cell greedy: scan cells row-major; for each empty cell try every type
// (area-descending) and all 8 orientations, shifting the piece so one of ITS cells
// lands on the scanned cell. This always fills the scanned cell when any piece can,
// then repeatedly sweeps to squeeze leftover gaps -> denser than plain first-fit.
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
    vector<int> ord(n); iota(ord.begin(), ord.end(), 0);
    sort(ord.begin(), ord.end(), [&](int p, int q){ return shape[p].size() > shape[q].size(); });

    // distinct orientations per type, carrying the orientation index k
    vector<vector<pair<int,vector<pair<int,int>>>>> pk(n); // list of (k, cells)
    for (int i = 0; i < n; i++) {
        set<vector<pair<int,int>>> seen;
        for (int k = 0; k < 8; k++) {
            auto v = normCells(i, k);
            auto vs = v; sort(vs.begin(), vs.end());
            if (seen.insert(vs).second) pk[i].push_back({k, v});
        }
    }

    vector<array<int,4>> out;
    bool progress = true;
    int sweeps = 0;
    while (progress && sweeps < 3) {
        progress = false; sweeps++;
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                if (occ[(size_t)y * W + x]) continue;
                bool done = false;
                for (int ti = 0; ti < n && !done; ti++) {
                    int t = ord[ti];
                    if (rem[t] <= 0) continue;
                    for (auto& kc : pk[t]) {
                        int k = kc.first; auto& v = kc.second;
                        for (auto& anchor : v) {
                            int ox = x - anchor.first, oy = y - anchor.second;
                            if (ox < 0 || oy < 0) continue;
                            bool ok = true;
                            for (auto& p : v) {
                                int cx = p.first + ox, cy = p.second + oy;
                                if (cx >= W || cy >= H || occ[(size_t)cy * W + cx]) { ok = false; break; }
                            }
                            if (!ok) continue;
                            for (auto& p : v) occ[(size_t)(p.second + oy) * W + (p.first + ox)] = 1;
                            rem[t]--; out.push_back({t + 1, k, ox, oy});
                            done = true; progress = true; break;
                        }
                        if (done) break;
                    }
                }
            }
        }
    }
    printf("%d\n", (int)out.size());
    for (auto& o : out) printf("%d %d %d %d\n", o[0], o[1], o[2], o[3]);
    return 0;
}
