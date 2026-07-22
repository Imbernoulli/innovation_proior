// TIER: greedy
// Textbook per-node marginal-gain forward selection. For each unselected
// candidate k it scores  v[k] + (synergy to the current team) - (k's OWN
// crowd penalty if k joins now)  and repeatedly adds the best positive
// scorer. This is the natural, efficient way to implement "greedy marginal
// gain" -- but it never re-checks how adding k changes the crowd penalty of
// candidates ALREADY on the team (k becoming a new selected neighbour for
// them). That externality is exactly what makes the true objective neither
// sub- nor super-modular, so this greedy keeps seeing "positive gain" long
// after the real total has turned negative and walks whole cliques past
// their true optimum.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N, M;
vector<ll> v, cap, pen;
vector<vector<pair<int,int>>> adj; // neighbor, s

int main() {
    scanf("%d %d", &N, &M);
    v.resize(N); cap.resize(N); pen.resize(N);
    for (int i = 0; i < N; i++) scanf("%lld", &v[i]);
    for (int i = 0; i < N; i++) scanf("%lld", &cap[i]);
    for (int i = 0; i < N; i++) scanf("%lld", &pen[i]);
    adj.assign(N, {});
    for (int e = 0; e < M; e++) {
        int a, b, s;
        scanf("%d %d %d", &a, &b, &s);
        adj[a].push_back({b, s});
        adj[b].push_back({a, s});
    }

    vector<char> selected(N, 0);
    vector<ll> degS(N, 0);          // # of already-selected neighbours
    vector<ll> runSyn(N, 0);        // sum of s over already-selected neighbours

    auto ownScore = [&](int k) -> ll {
        ll excess = degS[k] - cap[k];
        ll ownPen = excess > 0 ? pen[k] * excess : 0;
        return v[k] + runSyn[k] - ownPen;
    };

    for (int iter = 0; iter < N; iter++) {
        int best = -1; ll bestScore = 0;
        for (int k = 0; k < N; k++) {
            if (selected[k]) continue;
            ll sc = ownScore(k);
            if (sc > 0 && (best == -1 || sc > bestScore)) { best = k; bestScore = sc; }
        }
        if (best == -1) break;
        selected[best] = 1;
        // Update ONLY the local bookkeeping needed for future candidate scores
        // (neighbours' degS/runSyn) -- but never revisit the SCORE already
        // "locked in" for previously selected members. That omission is the bug.
        for (auto& pr : adj[best]) {
            int j = pr.first, s = pr.second;
            degS[j]++;
            runSyn[j] += s;
        }
    }

    vector<int> pick;
    for (int i = 0; i < N; i++) if (selected[i]) pick.push_back(i);
    printf("%d\n", (int)pick.size());
    for (size_t i = 0; i < pick.size(); i++) printf("%d%c", pick[i], (i + 1 == pick.size()) ? '\n' : ' ');
    if (pick.empty()) printf("\n");
    return 0;
}
