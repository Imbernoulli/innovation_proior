// TIER: greedy
// Classic weighted greedy set cover: repeatedly activate the sensor with the best
// (newly-covered demand)/(cost) ratio until every station is covered.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int N, M;
    if (scanf("%d %d", &N, &M) != 2) return 0;
    vector<ll> dx(N), dy(N);
    for (int i = 0; i < N; i++) scanf("%lld %lld", &dx[i], &dy[i]);
    vector<ll> sx(M), sy(M), sr(M), sc(M);
    for (int j = 0; j < M; j++) scanf("%lld %lld %lld %lld", &sx[j], &sy[j], &sr[j], &sc[j]);

    // precompute coverage lists
    vector<vector<int>> cov(M);
    for (int j = 0; j < M; j++) {
        ll r2 = sr[j] * sr[j];
        for (int i = 0; i < N; i++) {
            ll ddx = sx[j] - dx[i], ddy = sy[j] - dy[i];
            if (ddx * ddx + ddy * ddy <= r2) cov[j].push_back(i);
        }
    }

    vector<char> covered(N, 0), used(M, 0);
    int remaining = N;
    vector<int> chosen;
    while (remaining > 0) {
        int best = -1; double bestEff = -1;
        for (int j = 0; j < M; j++) {
            if (used[j]) continue;
            int gain = 0;
            for (int i : cov[j]) if (!covered[i]) gain++;
            if (gain == 0) continue;
            double eff = (double)gain / (double)sc[j];
            if (eff > bestEff) { bestEff = eff; best = j; }
        }
        if (best < 0) break; // should not happen (all-on is feasible)
        used[best] = 1; chosen.push_back(best);
        for (int i : cov[best]) if (!covered[i]) { covered[i] = 1; remaining--; }
    }

    printf("%d\n", (int)chosen.size());
    for (size_t k = 0; k < chosen.size(); k++)
        printf("%d%c", chosen[k] + 1, k + 1 == chosen.size() ? '\n' : ' ');
    return 0;
}
