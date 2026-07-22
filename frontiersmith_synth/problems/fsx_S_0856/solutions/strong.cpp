// TIER: strong
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Insight: legal grids are exactly the sequences of rows (each an element of
// {0,1}^C) such that consecutive rows are "compatible" (no forbidden 2x2
// window between them). This is a subshift of finite type whose achievable
// per-row density ceiling is the maximum MEAN-WEIGHT CYCLE of the transfer
// graph on 2^C row-states (weight of arriving at a state = its popcount) --
// a classic transfer-matrix computation, found here via Karp's algorithm.
// The ceiling is realized by tiling that cycle periodically down the grid,
// NOT by repeating any single row and NOT by a myopic one-step-ahead choice.

bool forbidden[16];
static inline int patIdx(int w, int x, int y, int z) { return w * 8 + x * 4 + y * 2 + z; }

int main() {
    int R, C, K;
    scanf("%d %d %d", &R, &C, &K);
    for (int i = 0; i < K; i++) {
        int w, x, y, z;
        scanf("%d %d %d %d", &w, &x, &y, &z);
        forbidden[patIdx(w, x, y, z)] = true;
    }

    int N = 1 << C;
    vector<int> weight(N);
    for (int s = 0; s < N; s++) weight[s] = __builtin_popcount((unsigned)s);

    // compat[s][t] = true iff row-state s may be immediately followed by t.
    vector<vector<uint8_t>> compat(N, vector<uint8_t>(N, 1));
    for (int s = 0; s < N; s++) {
        for (int t = 0; t < N; t++) {
            bool ok = true;
            for (int c = 0; c + 1 < C && ok; c++) {
                int w = (s >> c) & 1, x = (s >> (c + 1)) & 1;
                int y = (t >> c) & 1, z = (t >> (c + 1)) & 1;
                if (forbidden[patIdx(w, x, y, z)]) ok = false;
            }
            compat[s][t] = ok ? 1 : 0;
        }
    }

    // Karp max-mean-cycle DP: d[k][v] = max total weight of a walk of exactly
    // k edges ending at v, starting from ANY vertex (d[0][v] = 0 for all v).
    const ll NEG = LLONG_MIN / 2;
    int n = N;
    vector<vector<ll>> d(n + 1, vector<ll>(N, NEG));
    vector<vector<int>> par(n + 1, vector<int>(N, -1));
    for (int v = 0; v < N; v++) d[0][v] = 0;
    for (int k = 1; k <= n; k++) {
        for (int u = 0; u < N; u++) {
            ll du = d[k - 1][u];
            if (du <= NEG / 2) continue;
            const uint8_t* row = compat[u].data();
            for (int v = 0; v < N; v++) {
                if (!row[v]) continue;
                ll cand = du + weight[v];
                if (cand > d[k][v]) { d[k][v] = cand; par[k][v] = u; }
            }
        }
    }

    // Find v* maximizing the total n-edge walk weight, then backtrack and
    // extract a cycle via pigeonhole (n+1 vertices visited among <=N states).
    int vBest = -1; ll best = NEG;
    for (int v = 0; v < N; v++) if (d[n][v] > best) { best = d[n][v]; vBest = v; }

    vector<int> walk(n + 1);
    walk[n] = vBest;
    for (int k = n; k >= 1; k--) walk[k - 1] = par[k][walk[k]];

    vector<int> firstPos(N, -1);
    vector<int> cyc;
    for (int i = 0; i <= n; i++) {
        int v = walk[i];
        if (firstPos[v] != -1) {
            for (int j = firstPos[v]; j < i; j++) cyc.push_back(walk[j]);
            break;
        }
        firstPos[v] = i;
    }
    if (cyc.empty()) cyc.push_back(vBest); // safety net (should not trigger)

    int L = (int)cyc.size();
    string row(C, '0');
    for (int r = 0; r < R; r++) {
        int s = cyc[r % L];
        for (int c = 0; c < C; c++) row[c] = ((s >> c) & 1) ? '1' : '0';
        printf("%s\n", row.c_str());
    }
    return 0;
}
