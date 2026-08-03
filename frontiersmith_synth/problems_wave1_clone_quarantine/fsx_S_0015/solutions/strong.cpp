// TIER: strong
// Randomized multi-restart local search. Each restart runs the FFD-style fill with a
// shuffled type-priority order and a rotated orientation-try order; keep the best
// coverage found. Restart 0 == the deterministic greedy, so strong >= greedy always,
// and random restarts usually improve on it -> per-test divergence. Deterministic seed.
#include <bits/stdc++.h>
using namespace std;

int H, W, T;
vector<vector<pair<int,int>>> sh;
vector<int> avail, area;
vector<vector<vector<pair<int,int>>>> orients;

vector<pair<int,int>> orient(vector<pair<int,int>> cs, int o) {
    if (o >= 4) { for (auto& p : cs) p.second = -p.second; o -= 4; }
    for (int i = 0; i < o; i++)
        for (auto& p : cs) { int r = p.first, c = p.second; p.first = c; p.second = -r; }
    int mr = INT_MAX, mc = INT_MAX;
    for (auto& p : cs) { mr = min(mr, p.first); mc = min(mc, p.second); }
    for (auto& p : cs) { p.first -= mr; p.second -= mc; }
    return cs;
}

long long fill(const vector<int>& order, int rot, vector<array<int,4>>& out) {
    vector<char> occ(H * W, 0);
    vector<int> used(T + 1, 0);
    out.clear();
    long long cov = 0;
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++) {
            if (occ[r * W + c]) continue;
            bool placed = false;
            for (int oi = 0; oi < T && !placed; oi++) {
                int t = order[oi];
                if (used[t] >= avail[t]) continue;
                for (int oo = 0; oo < 8 && !placed; oo++) {
                    int o = (oo + rot) % 8;
                    auto& cs = orients[t][o];
                    for (auto& anch : cs) {
                        int r0 = r - anch.first, c0 = c - anch.second;
                        bool ok = true;
                        for (auto& p : cs) {
                            int R = r0 + p.first, C = c0 + p.second;
                            if (R < 0 || R >= H || C < 0 || C >= W) { ok = false; break; }
                            if (occ[R * W + C]) { ok = false; break; }
                        }
                        if (ok) {
                            for (auto& p : cs) occ[(r0 + p.first) * W + (c0 + p.second)] = 1;
                            used[t]++; out.push_back({t, o, r0, c0}); cov += area[t]; placed = true; break;
                        }
                    }
                }
            }
        }
    return cov;
}

int main() {
    if (scanf("%d %d %d", &H, &W, &T) != 3) return 0;
    sh.assign(T + 1, {}); avail.assign(T + 1, 0); area.assign(T + 1, 0);
    for (int t = 1; t <= T; t++) {
        int k, c; scanf("%d %d", &k, &c); avail[t] = c; area[t] = k;
        for (int i = 0; i < k; i++) { int r, cc; scanf("%d %d", &r, &cc); sh[t].push_back({r, cc}); }
    }
    orients.assign(T + 1, {});
    for (int t = 1; t <= T; t++)
        for (int o = 0; o < 8; o++) orients[t].push_back(orient(sh[t], o));

    // deterministic greedy = restart 0
    vector<int> order(T);
    for (int i = 0; i < T; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int a, int b) { return area[a] > area[b]; });

    vector<array<int,4>> best, cur;
    long long bestCov = fill(order, 0, best);

    mt19937 rng(1234567u + (unsigned)(H * 7919 + W * 104729));
    int restarts = min(48, 6 + 60000 / max(1, H * W));
    for (int it = 0; it < restarts; it++) {
        vector<int> ord = order;
        shuffle(ord.begin(), ord.end(), rng);
        int rot = (int)(rng() % 8);
        long long cov = fill(ord, rot, cur);
        if (cov > bestCov) { bestCov = cov; best = cur; }
    }

    printf("%d\n", (int)best.size());
    for (auto& o : best) printf("%d %d %d %d\n", o[0], o[1], o[2], o[3]);
    return 0;
}
