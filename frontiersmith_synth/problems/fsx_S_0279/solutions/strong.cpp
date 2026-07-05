// TIER: strong
// Weight-density best-first greedy with randomized multi-restart local search.
// Each restart repeatedly deploys the array/orientation/anchor that adds the most hazard
// weight among still-available types; randomized restarts pick among near-best placements
// to escape the deterministic order. Keep the best total monitored weight found.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int H, W, T;
vector<vector<int>> wgt;
vector<vector<pair<int,int>>> shape;
vector<int> avail, area;

vector<pair<int,int>> orient(vector<pair<int,int>> cs, int o) {
    if (o >= 4) { for (auto& p : cs) p.second = -p.second; o -= 4; }
    for (int i = 0; i < o; i++)
        for (auto& p : cs) { int r = p.first, c = p.second; p.first = c; p.second = -r; }
    int mr = INT_MAX, mc = INT_MAX;
    for (auto& p : cs) { mr = min(mr, p.first); mc = min(mc, p.second); }
    for (auto& p : cs) { p.first -= mr; p.second -= mc; }
    return cs;
}

vector<vector<vector<pair<int,int>>>> ors;   // ors[t][o] = cells
vector<vector<int>> obh, obw;                 // bbox of oriented shape

int main() {
    if (scanf("%d %d %d", &H, &W, &T) != 3) return 0;
    wgt.assign(H, vector<int>(W, 0));
    for (int r = 0; r < H; r++) for (int c = 0; c < W; c++) scanf("%d", &wgt[r][c]);
    shape.assign(T + 1, {}); avail.assign(T + 1, 0); area.assign(T + 1, 0);
    for (int t = 1; t <= T; t++) {
        int k, c; scanf("%d %d", &k, &c);
        avail[t] = c; area[t] = k;
        for (int i = 0; i < k; i++) { int rr, cc; scanf("%d %d", &rr, &cc); shape[t].push_back({rr, cc}); }
    }
    ors.assign(T + 1, {}); obh.assign(T + 1, vector<int>(8, 0)); obw.assign(T + 1, vector<int>(8, 0));
    for (int t = 1; t <= T; t++)
        for (int o = 0; o < 8; o++) {
            auto cs = orient(shape[t], o);
            ors[t].push_back(cs);
            int bh = 0, bw = 0; for (auto& p : cs) { bh = max(bh, p.first); bw = max(bw, p.second); }
            obh[t][o] = bh + 1; obw[t][o] = bw + 1;
        }

    struct Place { int t, o, r0, c0; ll w; };

    // run one greedy pass; `rndPick` controls stochastic near-best selection.
    auto runPass = [&](mt19937& rng, bool rndPick, vector<array<int,4>>& outPlace) -> ll {
        vector<vector<char>> occ(H, vector<char>(W, 0));
        vector<int> used(T + 1, 0);
        ll total = 0;
        outPlace.clear();
        while (true) {
            ll bestW = 0;
            vector<Place> near;
            for (int t = 1; t <= T; t++) {
                if (used[t] >= avail[t]) continue;
                for (int o = 0; o < 8; o++) {
                    auto& cs = ors[t][o];
                    int mh = obh[t][o], mw = obw[t][o];
                    for (int r0 = 0; r0 + mh <= H; r0++)
                        for (int c0 = 0; c0 + mw <= W; c0++) {
                            ll add = 0; bool ok = true;
                            for (auto& p : cs) {
                                int R = r0 + p.first, C = c0 + p.second;
                                if (occ[R][C]) { ok = false; break; }
                                add += wgt[R][C];
                            }
                            if (!ok) continue;
                            if (add > bestW) {
                                bestW = add;
                                if (rndPick) { near.clear(); near.push_back({t, o, r0, c0, add}); }
                            } else if (rndPick && add * 100 >= bestW * 88) {
                                near.push_back({t, o, r0, c0, add});
                            }
                        }
                }
            }
            if (bestW <= 0) break;
            Place chosen;
            if (rndPick && !near.empty()) {
                // keep only entries still within the band of the final bestW
                vector<Place> band;
                for (auto& p : near) if (p.w * 100 >= bestW * 88) band.push_back(p);
                chosen = band[rng() % band.size()];
            } else {
                // deterministic best-first: rescan to grab the argmax (scan order stable)
                chosen = {0, 0, 0, 0, -1};
                for (int t = 1; t <= T && chosen.w < bestW; t++) {
                    if (used[t] >= avail[t]) continue;
                    for (int o = 0; o < 8 && chosen.w < bestW; o++) {
                        auto& cs = ors[t][o]; int mh = obh[t][o], mw = obw[t][o];
                        for (int r0 = 0; r0 + mh <= H && chosen.w < bestW; r0++)
                            for (int c0 = 0; c0 + mw <= W; c0++) {
                                ll add = 0; bool ok = true;
                                for (auto& p : cs) {
                                    int R = r0 + p.first, C = c0 + p.second;
                                    if (occ[R][C]) { ok = false; break; }
                                    add += wgt[R][C];
                                }
                                if (ok && add == bestW) { chosen = {t, o, r0, c0, add}; break; }
                            }
                    }
                }
            }
            auto& cs = ors[chosen.t][chosen.o];
            for (auto& p : cs) occ[chosen.r0 + p.first][chosen.c0 + p.second] = 1;
            used[chosen.t]++;
            total += chosen.w;
            outPlace.push_back({chosen.t, chosen.o, chosen.r0, chosen.c0});
        }
        return total;
    };

    vector<array<int,4>> best, cur;
    mt19937 rng(20260702u);
    ll bestF = runPass(rng, false, best);   // deterministic weight-best-first

    int restarts = (H <= 14) ? 24 : (H <= 20 ? 10 : 5);
    for (int it = 0; it < restarts; it++) {
        ll f = runPass(rng, true, cur);
        if (f > bestF) { bestF = f; best = cur; }
    }

    printf("%d\n", (int)best.size());
    for (auto& p : best) printf("%d %d %d %d\n", p[0], p[1], p[2], p[3]);
    return 0;
}
