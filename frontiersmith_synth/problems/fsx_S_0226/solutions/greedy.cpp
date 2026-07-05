// TIER: greedy
// One coordinate-ascent sweep from the best-uniform start: for each marker in order,
// pick the color that maximizes the currently corroborated value (single pass, no restarts).
#include <bits/stdc++.h>
using namespace std;

int n, m, D;
vector<long long> w;
vector<vector<pair<int,int>>> cue;          // per clause: (marker, color)
vector<vector<pair<int,int>>> inc;          // per marker: (clause, required color)
vector<int> matchCount;                      // per clause
vector<int> assign_;                         // per marker (1-based)

void recompute() {
    for (int j = 0; j < m; j++) {
        int mc = 0;
        for (auto& pr : cue[j]) if (assign_[pr.first] == pr.second) mc++;
        matchCount[j] = mc;
    }
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
    int best = 0;
    for (int c = 1; c < D; c++) if (colorWeight[c] > colorWeight[best]) best = c;

    assign_.assign(n + 1, best);
    recompute();

    vector<long long> extra(D, 0);
    for (int v = 1; v <= n; v++) {
        for (int c = 0; c < D; c++) extra[c] = 0;
        for (auto& pr : inc[v]) {
            int j = pr.first, rc = pr.second;
            int base = matchCount[j] - (assign_[v] == rc ? 1 : 0); // matches from other markers
            if (base == 0) extra[rc] += w[j];  // gained only if we match this clause's color
        }
        int bc = assign_[v];
        for (int c = 0; c < D; c++) if (extra[c] > extra[bc]) bc = c;
        if (bc != assign_[v]) {
            int old = assign_[v];
            for (auto& pr : inc[v]) {
                int j = pr.first, rc = pr.second;
                if (old == rc) matchCount[j]--;
                if (bc == rc) matchCount[j]++;
            }
            assign_[v] = bc;
        }
    }

    for (int i = 1; i <= n; i++) printf("%d%c", assign_[i], i == n ? '\n' : ' ');
    return 0;
}
