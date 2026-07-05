// TIER: strong
// Start from the value-greedy schedule, then hill-climb with two monotone local moves:
//   (1) relocate a target to a higher-value telescope that currently fits;
//   (2) swap in an unassigned/lower-value target by kicking out a lower-value occupant
//       of a telescope when that strictly increases total value.
// Every accepted move strictly increases F, so the result is >= value-greedy.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int N, T;
    scanf("%d %d", &N, &T);
    vector<ll> C(T);
    for (int j = 0; j < T; j++) scanf("%lld", &C[j]);
    vector<vector<ll>> cost(N, vector<ll>(T)), val(N, vector<ll>(T));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < T; j++) scanf("%lld %lld", &cost[i][j], &val[i][j]);

    // ---- initial: value-greedy (same construction as the greedy tier) ----
    vector<int> order(N);
    iota(order.begin(), order.end(), 0);
    vector<ll> bestv(N, 0);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < T; j++) bestv[i] = max(bestv[i], val[i][j]);
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (bestv[a] != bestv[b]) return bestv[a] > bestv[b];
        return a < b;
    });

    vector<ll> rem = C;
    vector<int> assign(N, 0); // 0 = unassigned, else telescope 1..T
    for (int i : order) {
        ll best = -1; int bj = 0;
        for (int j = 0; j < T; j++)
            if (cost[i][j] <= rem[j] && val[i][j] > best) { best = val[i][j]; bj = j + 1; }
        if (bj > 0) { assign[i] = bj; rem[bj - 1] -= cost[i][bj - 1]; }
    }

    // occupants[j] = list of target indices assigned to telescope j+1
    auto curVal = [&](int i) -> ll { return assign[i] ? val[i][assign[i] - 1] : 0; };

    // ---- local search ----
    int MAXPASS = 60;
    for (int pass = 0; pass < MAXPASS; pass++) {
        bool changed = false;

        // Move (1): relocation to a higher-value fitting telescope.
        for (int i = 0; i < N; i++) {
            int cj = assign[i];
            ll cv = curVal(i);
            // effective remaining if i were removed from its current telescope
            int bestJ = cj; ll bestV = cv;
            for (int j = 0; j < T; j++) {
                ll effRem = rem[j] + ((cj == j + 1) ? cost[i][j] : 0);
                if (cost[i][j] <= effRem && val[i][j] > bestV) {
                    bestV = val[i][j]; bestJ = j + 1;
                }
            }
            if (bestJ != cj) {
                if (cj) rem[cj - 1] += cost[i][cj - 1];
                rem[bestJ - 1] -= cost[i][bestJ - 1];
                assign[i] = bestJ;
                changed = true;
            }
        }

        // Move (2): swap an unassigned target in by evicting a lower-value occupant.
        for (int i = 0; i < N; i++) {
            if (assign[i] != 0) continue;
            ll bestGain = 0; int bestJ = 0, bestKick = -1;
            for (int j = 0; j < T; j++) {
                // if it already fits, relocation move (1) would have taken it next pass;
                // here handle the eviction case.
                if (cost[i][j] <= rem[j]) {
                    ll gain = val[i][j];
                    if (gain > bestGain) { bestGain = gain; bestJ = j + 1; bestKick = -1; }
                    continue;
                }
                // need to free room: find the single lowest-value occupant on j whose
                // removal makes i fit and yields positive net gain.
                for (int x = 0; x < N; x++) {
                    if (assign[x] != j + 1) continue;
                    if (rem[j] + cost[x][j] >= cost[i][j]) {
                        ll gain = val[i][j] - val[x][j];
                        if (gain > bestGain) { bestGain = gain; bestJ = j + 1; bestKick = x; }
                    }
                }
            }
            if (bestJ > 0) {
                if (bestKick >= 0) {
                    rem[bestJ - 1] += cost[bestKick][bestJ - 1];
                    assign[bestKick] = 0;
                }
                rem[bestJ - 1] -= cost[i][bestJ - 1];
                assign[i] = bestJ;
                changed = true;
            }
        }

        if (!changed) break;
    }

    for (int i = 0; i < N; i++) printf("%d\n", assign[i]);
    return 0;
}
