// TIER: greedy
// The obvious first approach: greedily color the BASE interference graph only,
// processing stations in corridor order and giving each station a channel not
// used by its already-colored base-neighbours (falling back to the least-
// conflicting channel when all C are taken locally). It nails local interference
// almost perfectly -- but it never looks at the 40 ducting scenarios, so its
// natural reuse period (about the local clique size) can land squarely inside
// the scenario sweep and cause a mass collision on the worst-case day.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int N, C, K;
    if (!(cin >> N >> C >> K)) return 0;
    for (int i = 1; i <= N; i++) { ll x; cin >> x; } // positions, unused

    int M0; cin >> M0;
    vector<vector<pair<int,ll>>> radj(N + 1); // radj[j] = {(i,w): base edge i<j}
    for (int e = 0; e < M0; e++) {
        int i, j; ll w; cin >> i >> j >> w;
        if (i > j) swap(i, j);
        radj[j].push_back({i, w});
    }
    // duct scenarios are intentionally NOT read/used by this tier.

    vector<int> color(N + 1, 1);
    for (int j = 1; j <= N; j++) {
        vector<char> usedCh(C + 1, 0);
        vector<ll> conf(C + 1, 0);
        for (auto &pr : radj[j]) {
            int nb = pr.first; ll w = pr.second;
            usedCh[color[nb]] = 1;
            conf[color[nb]] += w;
        }
        int chosen = -1;
        for (int c = 1; c <= C; c++) if (!usedCh[c]) { chosen = c; break; }
        if (chosen == -1) {
            chosen = 1; ll best = conf[1];
            for (int c = 2; c <= C; c++) if (conf[c] < best) { best = conf[c]; chosen = c; }
        }
        color[j] = chosen;
    }

    for (int i = 1; i <= N; i++) cout << color[i] << (i < N ? ' ' : '\n');
    return 0;
}
