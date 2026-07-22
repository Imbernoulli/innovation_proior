// TIER: greedy
// The obvious first pass: build a run order by sorting ascending by
// concentration (kills carryover) and patching in stat deadlines with a
// standard "latest free slot <= deadline" construction (classic EDF-feasible
// scheduling), then handle the two reset budgets with two SEPARATE, INDEPENDENT
// textbook heuristics: spend the K1 tip changes on the K1 single largest raw
// gaps, and spend the K2 recalibrations at evenly-spaced positions. This never
// looks at the (1+BOOST) carryover/drift coupling, and it never checks whether
// a position it is about to recalibrate is one that a tip change already fixed
// (or vice versa) -- exactly the joint-placement blind spot the objective
// punishes.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int n, K1, K2; ll GAP_COEF, DRIFT_COEF, BOOST;
    scanf("%d %d %d %lld %lld %lld", &n, &K1, &K2, &GAP_COEF, &DRIFT_COEF, &BOOST);
    vector<ll> conc(n + 1);
    vector<int> isStat(n + 1, 0), deadline(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        scanf("%lld %d %d", &conc[i], &isStat[i], &deadline[i]);
    }

    // ---- build run order: EDF (latest-free-slot) for stat samples, then fill
    //      remaining positions ascending by concentration ---------------------
    vector<int> statIdx;
    for (int i = 1; i <= n; i++) if (isStat[i]) statIdx.push_back(i);
    sort(statIdx.begin(), statIdx.end(), [&](int a, int b) { return deadline[a] < deadline[b]; });

    vector<int> parent(n + 2);
    for (int x = 0; x <= n + 1; x++) parent[x] = x;
    function<int(int)> find = [&](int x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };

    vector<int> order(n + 1, 0); // order[j] = sample id at position j
    for (int s : statIdx) {
        int slot = find(min(deadline[s], n));
        order[slot] = s;
        parent[slot] = slot - 1;
    }
    vector<int> nonStat;
    for (int i = 1; i <= n; i++) if (!isStat[i]) nonStat.push_back(i);
    sort(nonStat.begin(), nonStat.end(), [&](int a, int b) { return conc[a] < conc[b]; });
    int ptr = 0;
    for (int j = 1; j <= n; j++) {
        if (order[j] == 0) order[j] = nonStat[ptr++];
    }

    // ---- gaps ---------------------------------------------------------------
    vector<ll> gap(n + 1, 0);
    for (int j = 2; j <= n; j++) gap[j] = max((ll)0, conc[order[j - 1]] - conc[order[j]]);

    // ---- tip: top-K1 positions by raw gap ------------------------------------
    vector<int> byGap(n + 1);
    for (int j = 1; j <= n; j++) byGap[j] = j;
    sort(byGap.begin() + 1, byGap.end(), [&](int a, int b) {
        if (gap[a] != gap[b]) return gap[a] > gap[b];
        return a < b;
    });
    vector<int> tip(n + 1, 0);
    for (int k = 0; k < K1 && k < n && gap[byGap[k + 1]] > 0; k++) tip[byGap[k + 1]] = 1;

    // ---- recal: evenly spaced, independent of gap structure ------------------
    vector<int> recal(n + 1, 0);
    if (K2 > 0) {
        int step = max(1, n / K2);
        int used = 0;
        set<int> placed;
        for (int k = 1; k <= K2; k++) {
            int posn = min(n, k * step);
            if (placed.count(posn)) continue;
            placed.insert(posn);
            recal[posn] = 1;
            used++;
            if (used >= K2) break;
        }
    }

    for (int j = 1; j <= n; j++) printf("%d%c", order[j], j == n ? '\n' : ' ');
    for (int j = 1; j <= n; j++) printf("%d%c", tip[j], j == n ? '\n' : ' ');
    for (int j = 1; j <= n; j++) printf("%d%c", recal[j], j == n ? '\n' : ' ');
    return 0;
}
