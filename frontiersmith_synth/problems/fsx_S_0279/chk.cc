#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int H, W, T;
vector<vector<int>> wgt;                 // hazard weights
vector<vector<pair<int,int>>> shape;     // 1..T footprints (normalized)
vector<int> avail, area;

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
    wgt.assign(H, vector<int>(W, 0));
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++)
            wgt[r][c] = inf.readInt();
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

    // ---- internal baseline B: regular bbox-grid tiling of type 1 (orient 0), row-major,
    //      capped by avail[1]; B = total hazard weight it monitors. ----
    int bh = 0, bw = 0;
    for (auto& p : shape[1]) { bh = max(bh, p.first); bw = max(bw, p.second); }
    bh++; bw++;
    ll rows = H / bh, cols = W / bw;
    ll slots = rows * cols;
    ll copies = min((ll)avail[1], slots);
    ll B = 0, placed = 0;
    for (ll i = 0; i < rows && placed < copies; i++)
        for (ll j = 0; j < cols && placed < copies; j++) {
            for (auto& p : shape[1]) B += wgt[i * bh + p.first][j * bw + p.second];
            placed++;
        }
    if (B <= 0) quitf(_fail, "bad instance: B=%lld", B);

    // ---- read & validate participant's deployments ----
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
                quitf(_wa, "array %lld leaves the flank at cell (%d,%d)", i + 1, R, C);
            ll id = (ll)R * W + C;
            if (occ[id]) quitf(_wa, "array %lld overlaps another at cell (%d,%d)", i + 1, R, C);
            occ[id] = 1;
            F += wgt[R][C];
        }
        used[t]++;
    }
    for (int t = 1; t <= T; t++)
        if (used[t] > avail[t])
            quitf(_wa, "type %d deployed %lld times but only %d available", t, used[t], avail[t]);
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
