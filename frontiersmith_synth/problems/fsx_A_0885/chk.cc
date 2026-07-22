#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Carving the Bakery Franchise Map". family: franchise-majority-carve.
// OBJECTIVE (MAX):
//   Each block (r,c) has favorite recipe f in 1..6 and spend s>0. Participant
//   assigns every block a territory id in 1..K. Every id actually used must
//   label a single edge-connected (4-dir) set of blocks -- else WA.
//   Each territory stocks the recipe with the largest household count inside
//   it (ties -> lowest recipe id). F = sum of spend of households whose own
//   favorite equals their own territory's stocked recipe.
// BASELINE B (checker-built, blind to any clever carve): the single whole-grid
//   territory (K=1 style). The `trivial` solution reproduces this exactly.
// Score: sc = min(1000, 100*F/max(1,B)); ratio = sc/1000 in [0,1] (cap = 10x B).
// -----------------------------------------------------------------------------

int R, C, K;
vector<vector<int>> f;
vector<vector<ll>> s;

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    R = inf.readInt();
    C = inf.readInt();
    K = inf.readInt();
    f.assign(R, vector<int>(C));
    s.assign(R, vector<ll>(C));
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++) f[r][c] = inf.readInt();
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++) s[r][c] = inf.readLong();

    // ---- read participant territory grid ----
    vector<vector<int>> id(R, vector<int>(C));
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++)
            id[r][c] = ouf.readInt(1, K, "territory id");
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after the territory grid");

    // ---- feasibility: every used id must be ONE connected component ----
    vector<vector<char>> visited(R, vector<char>(C, 0));
    vector<char> claimed(K + 1, 0);
    static const int dr[4] = {1, -1, 0, 0};
    static const int dc[4] = {0, 0, 1, -1};

    // per-territory aggregates, keyed by final territory id (1..K)
    vector<array<int,7>> cnt(K + 1, array<int,7>{0,0,0,0,0,0,0});
    vector<array<ll,7>>  spd(K + 1, array<ll,7>{0,0,0,0,0,0,0});

    for (int r0 = 0; r0 < R; r0++)
        for (int c0 = 0; c0 < C; c0++) {
            if (visited[r0][c0]) continue;
            int v = id[r0][c0];
            if (claimed[v]) quitf(_wa, "territory %d is not edge-connected (split into >1 piece)", v);
            // BFS this component
            vector<pair<int,int>> stack;
            stack.push_back({r0, c0});
            visited[r0][c0] = 1;
            while (!stack.empty()) {
                auto [r, c] = stack.back(); stack.pop_back();
                cnt[v][f[r][c]]++;
                spd[v][f[r][c]] += s[r][c];
                for (int d = 0; d < 4; d++) {
                    int nr = r + dr[d], nc = c + dc[d];
                    if (nr < 0 || nr >= R || nc < 0 || nc >= C) continue;
                    if (visited[nr][nc]) continue;
                    if (id[nr][nc] != v) continue;
                    visited[nr][nc] = 1;
                    stack.push_back({nr, nc});
                }
            }
            claimed[v] = 1;
        }

    ll F = 0;
    for (int v = 1; v <= K; v++) {
        if (!claimed[v]) continue;
        int best = 1;
        for (int rec = 2; rec <= 6; rec++) if (cnt[v][rec] > cnt[v][best]) best = rec;
        F += spd[v][best];
    }
    if (F <= 0) F = 1;

    // ---- baseline B: single whole-grid territory ----
    array<int,7> Bcnt{0,0,0,0,0,0,0};
    array<ll,7>  Bspd{0,0,0,0,0,0,0};
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++) { Bcnt[f[r][c]]++; Bspd[f[r][c]] += s[r][c]; }
    int Bbest = 1;
    for (int rec = 2; rec <= 6; rec++) if (Bcnt[rec] > Bcnt[Bbest]) Bbest = rec;
    ll B = Bspd[Bbest];
    if (B <= 0) B = 1;

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
