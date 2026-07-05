// TIER: strong
// Multi-restart coordinate-ascent local search: run coordinate ascent to convergence from
// the best-uniform start and from several seeded random starts; keep the best coloring.
#include <bits/stdc++.h>
using namespace std;

int n, m, D;
vector<long long> w;
vector<vector<pair<int,int>>> cue;
vector<vector<pair<int,int>>> inc;
vector<int> matchCount;
vector<int> assign_;

long long curF() {
    long long F = 0;
    for (int j = 0; j < m; j++) if (matchCount[j] > 0) F += w[j];
    return F;
}

void recompute() {
    for (int j = 0; j < m; j++) {
        int mc = 0;
        for (auto& pr : cue[j]) if (assign_[pr.first] == pr.second) mc++;
        matchCount[j] = mc;
    }
}

// full coordinate ascent to convergence; returns final F
long long ascend(vector<long long>& extra) {
    bool improved = true;
    int sweeps = 0;
    while (improved && sweeps < 40) {
        improved = false;
        sweeps++;
        for (int v = 1; v <= n; v++) {
            if (inc[v].empty()) continue;
            for (int c = 0; c < D; c++) extra[c] = 0;
            for (auto& pr : inc[v]) {
                int j = pr.first, rc = pr.second;
                int base = matchCount[j] - (assign_[v] == rc ? 1 : 0);
                if (base == 0) extra[rc] += w[j];
            }
            int cur = assign_[v];
            int bc = cur;
            for (int c = 0; c < D; c++) if (extra[c] > extra[bc]) bc = c;
            if (bc != cur) {
                for (auto& pr : inc[v]) {
                    int j = pr.first, rc = pr.second;
                    if (cur == rc) matchCount[j]--;
                    if (bc == rc) matchCount[j]++;
                }
                assign_[v] = bc;
                improved = true;
            }
        }
    }
    return curF();
}

int main() {
    if (scanf("%d %d %d", &n, &m, &D) != 3) return 0;
    w.resize(m); cue.resize(m); inc.assign(n + 1, {}); matchCount.assign(m, 0);
    vector<long long> colorWeight(D, 0);
    for (int j = 0; j < m; j++) {
        int k; scanf("%d", &k);
        cue[j].resize(k);
        vector<int> cols;
        for (int e = 0; e < k; e++) {
            int v, c; scanf("%d %d", &v, &c);
            cue[j][e] = {v, c};
            inc[v].push_back({j, c});
            cols.push_back(c);
        }
        long long ww; scanf("%lld", &ww); w[j] = ww;
        sort(cols.begin(), cols.end());
        cols.erase(unique(cols.begin(), cols.end()), cols.end());
        for (int c : cols) colorWeight[c] += ww;
    }
    int bestUniform = 0;
    for (int c = 1; c < D; c++) if (colorWeight[c] > colorWeight[bestUniform]) bestUniform = c;

    vector<long long> extra(D, 0);
    assign_.assign(n + 1, bestUniform);
    vector<int> bestAssign(n + 1, bestUniform);
    long long bestF = -1;

    mt19937 rng(987654321u);

    int restarts = 6;
    for (int r = 0; r < restarts; r++) {
        if (r == 0) {
            for (int v = 1; v <= n; v++) assign_[v] = bestUniform;
        } else {
            assign_.assign(n + 1, 0);
            for (int v = 1; v <= n; v++) assign_[v] = (int)(rng() % D);
        }
        recompute();
        long long f = ascend(extra);
        if (f > bestF) { bestF = f; for (int v = 1; v <= n; v++) bestAssign[v] = assign_[v]; }
    }

    for (int i = 1; i <= n; i++) printf("%d%c", bestAssign[i], i == n ? '\n' : ' ');
    return 0;
}
