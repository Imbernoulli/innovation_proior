// TIER: greedy
// The obvious first attempt: pooled maximum-coverage set cover at a SINGLE fixed
// radius, ignoring which wing each demand belongs to entirely. Each round, sample a
// batch of currently-uncovered demands (from the whole pool, across every wing) and
// install a template at whichever sampled point covers the most raw, still-uncovered
// demands. This finds the biggest pooled clusters fast (a wing built from one huge
// dominant motif, or a motif repeated across many wings), but a FIXED radius is
// expensive, and once the big clusters are gone the budget frequently runs out
// before the many small, per-wing residual motifs are ever reached -- those wings
// stay stuck at a partial floor while others sit at 100%. It never asks "which wing
// is currently worst," only "what covers the most demands right now."
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static int L;

static inline int hammingDist(const string &a, const string &b) {
    int d = 0;
    for (int i = 0; i < L; i++) if (a[i] != b[i]) d++;
    return d;
}

int main() {
    int K, Mmax, Rmax;
    ll Budget;
    if (scanf("%d %d %d %d %lld", &K, &Mmax, &L, &Rmax, &Budget) != 5) return 0;

    vector<string> flat;
    vector<int> owner;
    vector<int> n(K);
    static char buf[64];
    for (int k = 0; k < K; k++) {
        scanf("%d", &n[k]);
        for (int i = 0; i < n[k]; i++) {
            scanf("%s", buf);
            flat.push_back(string(buf));
            owner.push_back(k);
        }
    }
    int total = (int)flat.size();

    int R_FIXED = max(1, Rmax / 2);
    ll cost = (ll)(R_FIXED + 1) * (R_FIXED + 1);

    vector<int> uncovered(total);
    for (int i = 0; i < total; i++) uncovered[i] = i;

    mt19937 rng(20260629u);
    vector<pair<string, int>> out;
    ll budgetLeft = Budget;
    const int CAND = 15;

    while ((int)out.size() < Mmax && budgetLeft >= cost && !uncovered.empty()) {
        int bestIdx = -1, bestGain = -1;
        int m = (int)uncovered.size();
        int trials = min(CAND, m);
        for (int t = 0; t < trials; t++) {
            int pick = uncovered[rng() % m];
            const string &center = flat[pick];
            int gain = 0;
            for (int idx : uncovered) if (hammingDist(flat[idx], center) <= R_FIXED) gain++;
            if (gain > bestGain) { bestGain = gain; bestIdx = pick; }
        }
        if (bestGain <= 0) break;
        out.push_back({flat[bestIdx], R_FIXED});
        budgetLeft -= cost;
        vector<int> remain;
        remain.reserve(uncovered.size());
        for (int idx : uncovered) if (hammingDist(flat[idx], flat[bestIdx]) > R_FIXED) remain.push_back(idx);
        uncovered.swap(remain);
    }

    printf("%d\n", (int)out.size());
    for (auto &p : out) printf("%s %d\n", p.first.c_str(), p.second);
    return 0;
}
