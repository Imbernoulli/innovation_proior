// TIER: strong
// Randomized multi-restart greedy + redundancy pruning.
//  * deterministic (fixed seed);
//  * each restart perturbs the greedy tie/near-tie choices, then prunes redundant
//    sensors (expensive-first), keeping the cheapest feasible net found.
// Beats plain greedy because pruning removes sensors made redundant by later picks,
// and restarts explore different cover structures on clustered vs uniform demand.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N, M;
vector<ll> dx, dy, sx, sy, sr, sc;
vector<vector<int>> cov;

// prune redundant sensors from a feasible set, most-expensive first.
ll pruneCost(vector<int>& sel) {
    // coverage count per demand
    vector<int> cnt(N, 0);
    for (int j : sel) for (int i : cov[j]) cnt[i]++;
    // sort candidates by cost descending
    sort(sel.begin(), sel.end(), [&](int a, int b){ return sc[a] > sc[b]; });
    vector<int> keep;
    vector<char> removed(M, 0);
    for (int j : sel) {
        bool redundant = true;
        for (int i : cov[j]) if (cnt[i] <= 1) { redundant = false; break; }
        if (redundant) { for (int i : cov[j]) cnt[i]--; removed[j] = 1; }
    }
    ll total = 0;
    for (int j : sel) if (!removed[j]) total += sc[j];
    // rebuild kept list in place
    vector<int> out;
    for (int j : sel) if (!removed[j]) out.push_back(j);
    sel.swap(out);
    return total;
}

int main() {
    if (scanf("%d %d", &N, &M) != 2) return 0;
    dx.resize(N); dy.resize(N);
    for (int i = 0; i < N; i++) scanf("%lld %lld", &dx[i], &dy[i]);
    sx.resize(M); sy.resize(M); sr.resize(M); sc.resize(M);
    for (int j = 0; j < M; j++) scanf("%lld %lld %lld %lld", &sx[j], &sy[j], &sr[j], &sc[j]);

    cov.assign(M, {});
    for (int j = 0; j < M; j++) {
        ll r2 = sr[j] * sr[j];
        for (int i = 0; i < N; i++) {
            ll ddx = sx[j] - dx[i], ddy = sy[j] - dy[i];
            if (ddx * ddx + ddy * ddy <= r2) cov[j].push_back(i);
        }
    }

    mt19937 rng(987654321u);

    auto oneRun = [&](double jitter) -> pair<ll, vector<int>> {
        vector<char> covered(N, 0);
        int remaining = N;
        vector<int> chosen;
        while (remaining > 0) {
            // collect candidates near the best efficiency, pick one at random
            double bestEff = -1;
            for (int j = 0; j < M; j++) {
                int gain = 0;
                for (int i : cov[j]) if (!covered[i]) gain++;
                if (gain == 0) continue;
                double eff = (double)gain / (double)sc[j];
                if (eff > bestEff) bestEff = eff;
            }
            if (bestEff < 0) break;
            double thr = bestEff * (1.0 - jitter);
            vector<int> pool;
            for (int j = 0; j < M; j++) {
                int gain = 0;
                for (int i : cov[j]) if (!covered[i]) gain++;
                if (gain == 0) continue;
                double eff = (double)gain / (double)sc[j];
                if (eff >= thr) pool.push_back(j);
            }
            int pick = pool[rng() % pool.size()];
            chosen.push_back(pick);
            for (int i : cov[pick]) if (!covered[i]) { covered[i] = 1; remaining--; }
        }
        ll c = pruneCost(chosen);
        return {c, chosen};
    };

    ll bestCost = LLONG_MAX;
    vector<int> best;
    // restart 0: pure greedy (jitter 0) then prune
    // remaining restarts: perturbed
    int restarts = 24;
    for (int r = 0; r < restarts; r++) {
        double jitter = (r == 0) ? 0.0 : (0.05 + 0.15 * ((double)(r % 5) / 4.0));
        auto res = oneRun(jitter);
        if (res.first < bestCost) { bestCost = res.first; best = res.second; }
    }

    printf("%d\n", (int)best.size());
    for (size_t k = 0; k < best.size(); k++)
        printf("%d%c", best[k] + 1, k + 1 == best.size() ? '\n' : ' ');
    return 0;
}
