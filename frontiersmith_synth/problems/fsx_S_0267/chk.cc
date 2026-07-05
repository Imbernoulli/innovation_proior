#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int H, W, P;
vector<string> grid;                       // '.' free, '#' brood
vector<vector<pair<int,int>>> base;        // base offsets per type (0-indexed t-1)
vector<int> cnt;                            // supply per type

// dihedral transform of base offsets by orientation o, then normalize to non-negative
vector<pair<int,int>> orient(const vector<pair<int,int>>& b, int o) {
    vector<pair<int,int>> r;
    r.reserve(b.size());
    for (auto &p : b) {
        int a = p.first, c = p.second, na = 0, nc = 0;
        switch (o) {
            case 0: na =  a; nc =  c; break;
            case 1: na =  c; nc = -a; break;
            case 2: na = -a; nc = -c; break;
            case 3: na = -c; nc =  a; break;
            case 4: na =  a; nc = -c; break;
            case 5: na = -a; nc =  c; break;
            case 6: na =  c; nc =  a; break;
            case 7: na = -c; nc = -a; break;
        }
        r.push_back({na, nc});
    }
    int mr = INT_MAX, mc = INT_MAX;
    for (auto &p : r) { mr = min(mr, p.first); mc = min(mc, p.second); }
    for (auto &p : r) { p.first -= mr; p.second -= mc; }
    return r;
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    H = inf.readInt();
    W = inf.readInt();
    P = inf.readInt();
    grid.resize(H);
    for (int i = 0; i < H; i++) grid[i] = inf.readToken();
    base.resize(P); cnt.resize(P);
    for (int t = 0; t < P; t++) {
        int s = inf.readInt();
        cnt[t] = inf.readInt();
        for (int k = 0; k < s; k++) {
            int dr = inf.readInt();
            int dc = inf.readInt();
            base[t].push_back({dr, dc});
        }
    }

    // ---- internal baseline B: largest-area type with any feasible single placement ----
    vector<int> order(P);
    for (int t = 0; t < P; t++) order[t] = t;
    sort(order.begin(), order.end(), [&](int a, int b) {
        return base[a].size() > base[b].size();
    });
    ll B = 0;
    for (int t : order) {
        bool placeable = false;
        for (int o = 0; o < 8 && !placeable; o++) {
            auto sh = orient(base[t], o);
            int Mr = 0, Mc = 0;
            for (auto &p : sh) { Mr = max(Mr, p.first); Mc = max(Mc, p.second); }
            for (int r = 0; r + Mr < H && !placeable; r++)
                for (int c = 0; c + Mc < W && !placeable; c++) {
                    bool ok = true;
                    for (auto &p : sh) {
                        if (grid[r + p.first][c + p.second] == '#') { ok = false; break; }
                    }
                    if (ok) placeable = true;
                }
        }
        if (placeable) { B = (ll)base[t].size(); break; }
    }
    if (B <= 0) quitf(_fail, "bad instance: no piece can be placed, B=%lld", B);

    // ---- read & validate participant placements ----
    int N = ouf.readInt(0, H * W, "N");
    vector<int> used(P, 0);
    vector<vector<char>> covered(H, vector<char>(W, 0));
    ll F = 0;
    for (int i = 0; i < N; i++) {
        int t = ouf.readInt(1, P, "t") - 1;
        int o = ouf.readInt(0, 7, "o");
        int r = ouf.readInt(0, H - 1, "r");
        int c = ouf.readInt(0, W - 1, "c");
        used[t]++;
        if (used[t] > cnt[t]) quitf(_wa, "type %d used %d times > supply %d", t + 1, used[t], cnt[t]);
        auto sh = orient(base[t], o);
        for (auto &p : sh) {
            int rr = r + p.first, cc = c + p.second;
            if (rr < 0 || rr >= H || cc < 0 || cc >= W)
                quitf(_wa, "placement %d covers out-of-frame cell (%d,%d)", i, rr, cc);
            if (grid[rr][cc] == '#')
                quitf(_wa, "placement %d covers brood cell (%d,%d)", i, rr, cc);
            if (covered[rr][cc])
                quitf(_wa, "placement %d overlaps at cell (%d,%d)", i, rr, cc);
            covered[rr][cc] = 1;
            F++;
        }
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
