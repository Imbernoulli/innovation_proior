#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
using ll = long long;

struct Basin {
    int r, c, node, demand;
};

struct Pump {
    int r, c, node, cost, reach;
};

static long long edgeKey(int a, int b) {
    if (a > b) swap(a, b);
    return (static_cast<long long>(a) << 32) ^ static_cast<unsigned int>(b);
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int H = inf.readInt();
    int W = inf.readInt();
    int N = inf.readInt();
    int P = inf.readInt();
    int cells = H * W;
    auto node = [&](int r, int c) { return (r - 1) * W + c; };
    auto rowOf = [&](int v) { return (v - 1) / W + 1; };
    auto colOf = [&](int v) { return (v - 1) % W + 1; };

    vector<string> grid(H + 1);
    vector<int> width(cells + 1, 0);
    vector<char> canal(cells + 1, 0);
    for (int r = 1; r <= H; r++) {
        grid[r] = inf.readToken();
        if ((int)grid[r].size() != W) quitf(_fail, "bad grid row length");
        for (int c = 1; c <= W; c++) {
            char ch = grid[r][c - 1];
            int v = node(r, c);
            if (ch == '#') continue;
            if (ch < '1' || ch > '9') quitf(_fail, "bad grid character");
            canal[v] = 1;
            width[v] = ch - '0';
        }
    }

    vector<Basin> basins(N + 1);
    for (int i = 1; i <= N; i++) {
        int r = inf.readInt();
        int c = inf.readInt();
        int d = inf.readInt();
        int v = node(r, c);
        if (r < 1 || r > H || c < 1 || c > W || !canal[v]) quitf(_fail, "basin off canal");
        basins[i] = {r, c, v, d};
    }

    vector<Pump> pumps(P + 1);
    ll B = 0;
    for (int j = 1; j <= P; j++) {
        int r = inf.readInt();
        int c = inf.readInt();
        int f = inf.readInt();
        int R = inf.readInt();
        int v = node(r, c);
        if (r < 1 || r > H || c < 1 || c > W || !canal[v]) quitf(_fail, "pump off canal");
        if (f <= 0 || R < 0) quitf(_fail, "bad pump parameters");
        pumps[j] = {r, c, v, f, R};
        B += f;
    }
    if (B <= 0) quitf(_fail, "bad baseline");

    for (int i = 1; i <= N; i++) {
        bool ok = false;
        for (int j = 1; j <= P; j++) {
            if (pumps[j].node == basins[i].node && pumps[j].reach == 0) {
                ok = true;
                break;
            }
        }
        if (!ok) quitf(_fail, "basin %d has no co-located emergency pump", i);
    }

    unordered_map<long long, int> eid;
    eid.reserve((size_t)cells * 3);
    vector<int> cap;
    auto addEdge = [&](int a, int b) {
        if (!canal[a] || !canal[b]) return;
        long long k = edgeKey(a, b);
        if (eid.find(k) != eid.end()) return;
        int c = 4 * min(width[a], width[b]) + 2;
        eid[k] = (int)cap.size();
        cap.push_back(c);
    };
    for (int r = 1; r <= H; r++) {
        for (int c = 1; c <= W; c++) {
            int v = node(r, c);
            if (!canal[v]) continue;
            if (r < H) addEdge(v, node(r + 1, c));
            if (c < W) addEdge(v, node(r, c + 1));
        }
    }
    vector<ll> load(cap.size(), 0);

    int K = ouf.readInt(0, P, "K");
    vector<char> opened(P + 1, 0);
    ll F = 0;
    for (int t = 0; t < K; t++) {
        int p = ouf.readInt(1, P, "openedPump");
        if (opened[p]) quitf(_wa, "pump %d opened twice", p);
        opened[p] = 1;
        F += pumps[p].cost;
    }

    int maxLen = max(0, cells - 1);
    for (int i = 1; i <= N; i++) {
        int p = ouf.readInt(1, P, "assignedPump");
        if (!opened[p]) quitf(_wa, "basin %d assigned to unopened pump %d", i, p);
        int manh = abs(basins[i].r - pumps[p].r) + abs(basins[i].c - pumps[p].c);
        if (manh > pumps[p].reach) {
            quitf(_wa, "basin %d assigned outside pump reach: distance %d > %d",
                  i, manh, pumps[p].reach);
        }
        int L = ouf.readInt(0, maxLen, "pathLength");
        int prev = ouf.readInt(1, cells, "routeCell");
        if (!canal[prev]) quitf(_wa, "basin %d route starts on blocked cell", i);
        if (prev != basins[i].node) quitf(_wa, "basin %d route starts at %d, expected %d",
                                          i, prev, basins[i].node);
        for (int e = 0; e < L; e++) {
            int cur = ouf.readInt(1, cells, "routeCell");
            if (!canal[cur]) quitf(_wa, "basin %d route visits blocked cell %d", i, cur);
            int dr = abs(rowOf(prev) - rowOf(cur));
            int dc = abs(colOf(prev) - colOf(cur));
            if (dr + dc != 1) quitf(_wa, "basin %d route has non-adjacent step", i);
            auto it = eid.find(edgeKey(prev, cur));
            if (it == eid.end()) quitf(_fail, "missing canal edge");
            load[it->second] += basins[i].demand;
            prev = cur;
        }
        if (prev != pumps[p].node) {
            quitf(_wa, "basin %d route ends at %d, expected pump cell %d",
                  i, prev, pumps[p].node);
        }
        F += 8LL * basins[i].demand * L;
        F += 3LL * basins[i].demand * manh * manh;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output");

    for (size_t e = 0; e < load.size(); e++) {
        if (load[e] == 0) continue;
        F += (load[e] * load[e] + cap[e] - 1) / cap[e];
    }

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
