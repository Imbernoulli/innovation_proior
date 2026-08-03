#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int H, W, T;
vector<vector<pair<int,int>>> shape;  // 1..T
vector<int> avail;
vector<int> area;

// Orientation semantics MUST match the statement exactly.
vector<pair<int,int>> orient(vector<pair<int,int>> cs, int o) {
    if (o >= 4) { for (auto& p : cs) p.second = -p.second; o -= 4; }
    for (int i = 0; i < o; i++)
        for (auto& p : cs) { int r = p.first, c = p.second; p.first = c; p.second = -r; }
    int mr = INT_MAX, mc = INT_MAX;
    for (auto& p : cs) { mr = min(mr, p.first); mc = min(mc, p.second); }
    for (auto& p : cs) { p.first -= mr; p.second -= mc; }
    return cs;
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    H = inf.readInt();
    W = inf.readInt();
    T = inf.readInt();
    shape.assign(T + 1, {});
    avail.assign(T + 1, 0);
    area.assign(T + 1, 0);
    for (int t = 1; t <= T; t++) {
        int k = inf.readInt();
        int c = inf.readInt();
        avail[t] = c;
        area[t] = k;
        for (int i = 0; i < k; i++) {
            int r = inf.readInt(), cc = inf.readInt();
            shape[t].push_back({r, cc});
        }
    }

    // ---- internal baseline B: regular bbox-grid tiling of type 1, capped by avail ----
    int bh = 0, bw = 0;
    for (auto& p : shape[1]) { bh = max(bh, p.first); bw = max(bw, p.second); }
    bh++; bw++;
    ll slots = (ll)(H / bh) * (ll)(W / bw);
    ll copies = min((ll)avail[1], slots);
    ll B = copies * (ll)area[1];
    if (B <= 0) quitf(_fail, "bad instance: B=%lld", B);

    // ---- read & validate participant's placements ----
    ll P = ouf.readInt(0, (ll)H * W, "P");
    vector<char> occ((ll)H * W, 0);
    vector<ll> used(T + 1, 0);
    ll F = 0;
    for (ll i = 0; i < P; i++) {
        int t = ouf.readInt(1, T, "type");
        int o = ouf.readInt(0, 7, "orient");
        int r0 = ouf.readInt(-1000000, 1000000, "r0");
        int c0 = ouf.readInt(-1000000, 1000000, "c0");
        auto cs = orient(shape[t], o);
        for (auto& p : cs) {
            int R = r0 + p.first, C = c0 + p.second;
            if (R < 0 || R >= H || C < 0 || C >= W)
                quitf(_wa, "module %lld leaves the parcel at tile (%d,%d)", i + 1, R, C);
            ll id = (ll)R * W + C;
            if (occ[id]) quitf(_wa, "module %lld overlaps another at tile (%d,%d)", i + 1, R, C);
            occ[id] = 1;
        }
        used[t]++;
        F += area[t];
    }
    for (int t = 1; t <= T; t++)
        if (used[t] > avail[t])
            quitf(_wa, "type %d placed %lld times but only %d available", t, used[t], avail[t]);
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
