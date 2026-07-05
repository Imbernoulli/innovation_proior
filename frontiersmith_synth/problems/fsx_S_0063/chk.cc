#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int W, H, n;
vector<vector<pair<int,int>>> shape; // shape[type] = cells
vector<ll> stock;

// apply orientation k (0..7) to cell (a,b)
static inline pair<int,int> tf(int a, int b, int k) {
    int f = k / 4, r = k % 4;
    int x = a, y = b;
    if (f) x = -x;
    for (int i = 0; i < r; i++) { int nx = y, ny = -x; x = nx; y = ny; }
    return {x, y};
}

// normalized transformed cells (min col/row shifted to 0)
static vector<pair<int,int>> normCells(int type, int k) {
    vector<pair<int,int>> v;
    int mnx = INT_MAX, mny = INT_MAX;
    for (auto& c : shape[type]) {
        auto p = tf(c.first, c.second, k);
        v.push_back(p);
        mnx = min(mnx, p.first);
        mny = min(mny, p.second);
    }
    for (auto& p : v) { p.first -= mnx; p.second -= mny; }
    return v;
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    W = inf.readInt();
    H = inf.readInt();
    n = inf.readInt();
    shape.assign(n, {});
    stock.assign(n, 0);
    for (int i = 0; i < n; i++) {
        int s = inf.readInt();
        ll c = inf.readInt();
        stock[i] = c;
        for (int j = 0; j < s; j++) {
            int a = inf.readInt(), b = inf.readInt();
            shape[i].push_back({a, b});
        }
    }

    // ---- internal single-type baseline B ----
    // best coverage tiling copies of one type in a bounding-box grid, over all
    // 8 orientations, capped by that type's stock. B must be positive.
    ll B = 0;
    for (int i = 0; i < n; i++) {
        int area = (int)shape[i].size();
        for (int k = 0; k < 8; k++) {
            auto v = normCells(i, k);
            int bw = 0, bh = 0;
            for (auto& p : v) { bw = max(bw, p.first + 1); bh = max(bh, p.second + 1); }
            if (bw > W || bh > H) continue;
            ll copies = (ll)(W / bw) * (ll)(H / bh);
            copies = min(copies, stock[i]);
            B = max(B, copies * (ll)area);
        }
    }
    if (B <= 0) quitf(_fail, "bad instance: B=%lld", B);

    // ---- read & validate participant layout ----
    vector<char> occ((size_t)W * H, 0);
    vector<ll> used(n, 0);
    ll F = 0;
    ll P = ouf.readLong(0, (ll)W * H, "P");
    for (ll q = 0; q < P; q++) {
        int t = ouf.readInt(1, n, "type") - 1;
        int k = ouf.readInt(0, 7, "orient");
        int ox = ouf.readInt(0, W - 1, "x");
        int oy = ouf.readInt(0, H - 1, "y");
        auto v = normCells(t, k);
        for (auto& p : v) {
            int cx = p.first + ox, cy = p.second + oy;
            if (cx < 0 || cx >= W || cy < 0 || cy >= H)
                quitf(_wa, "placement %lld of type %d leaves the tray (cell %d,%d)", q + 1, t + 1, cx, cy);
            size_t id = (size_t)cy * W + cx;
            if (occ[id]) quitf(_wa, "placement %lld overlaps at cell (%d,%d)", q + 1, cx, cy);
            occ[id] = 1;
        }
        used[t]++;
        F += (ll)shape[t].size();
    }
    for (int i = 0; i < n; i++)
        if (used[i] > stock[i])
            quitf(_wa, "type %d used %lld > stock %lld", i + 1, used[i], stock[i]);
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
