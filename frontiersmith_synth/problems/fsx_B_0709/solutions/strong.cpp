// TIER: strong
// Risk-aware allocation via a soft-minimax (multiplicative-weights-style) greedy:
// the insight is to reformulate F = min_g C(g) as a covering game and, at each
// step, weight each jammer position by how close it currently is to being the
// worst one (harmonic rank weighting -- scale-free, no tuning constant needed),
// then place the tower that most improves that WEIGHTED sum of coverage gains.
// A pure hard-minimax view undervalues any candidate that some single jammer can
// zero out even when it is highly valuable under every OTHER jammer; the soft
// (rank-weighted) view keeps crediting a candidate for the jammers it survives
// while still steadily shifting budget toward whichever jammer position is
// currently starving -- which is exactly what stops redundant coverage from
// piling up somewhere a single deletion disc can erase it all.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N, K, M, L, r, R;
vector<ll> px, py, pw; vector<int> pt;
vector<ll> gx, gy;

int main(){
    if (scanf("%d %d %d %d %d %d", &N, &K, &M, &L, &r, &R) != 6) return 0;
    px.resize(N); py.resize(N); pw.resize(N); pt.resize(N);
    for (int i = 0; i < N; i++)
        scanf("%lld %lld %lld %d", &px[i], &py[i], &pw[i], &pt[i]);
    gx.resize(M); gy.resize(M);
    for (int g = 0; g < M; g++) scanf("%lld %lld", &gx[g], &gy[g]);

    ll r2 = (ll)r * r, R2 = (ll)R * R;

    vector<vector<int>> coverList(N);
    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++){
            ll dx = px[j] - px[i], dy = py[j] - py[i];
            if (dx * dx + dy * dy <= r2) coverList[j].push_back(i);
        }

    // survJ[j][g]: does a tower placed at demand point j survive jammer g?
    vector<vector<char>> survJ(N, vector<char>(M));
    for (int j = 0; j < N; j++)
        for (int g = 0; g < M; g++){
            ll dx = px[j] - gx[g], dy = py[j] - gy[g];
            survJ[j][g] = (dx * dx + dy * dy > R2) ? 1 : 0;
        }

    vector<vector<int>> cnt(M, vector<int>(N, 0));   // cnt[g][i]: surviving-under-g count so far
    vector<double> covG(M, 0.0);                      // C(g) so far
    vector<double> weight(M, 1.0);

    for (int step = 0; step < K; step++){
        // harmonic rank weighting: the currently-worst jammer gets weight 1,
        // the second-worst 1/2, etc. -- scale-free, needs no tuning constant.
        vector<int> order(M);
        iota(order.begin(), order.end(), 0);
        sort(order.begin(), order.end(), [&](int a, int b){ return covG[a] < covG[b]; });
        for (int rk = 0; rk < M; rk++) weight[order[rk]] = 1.0 / (rk + 1.0);

        double best = -1.0; int bestJ = 0;
        for (int j = 0; j < N; j++){
            double score = 0.0;
            const vector<int>& cl = coverList[j];
            for (int g = 0; g < M; g++){
                if (!survJ[j][g]) continue;
                const vector<int>& cg = cnt[g];
                double inc = 0.0;
                for (int i : cl){
                    int c = cg[i];
                    if (c < pt[i]) inc += (double)pw[i] * (min(c + 1, pt[i]) - c) / (double)pt[i];
                }
                score += weight[g] * inc;
            }
            if (score > best){ best = score; bestJ = j; }
        }
        for (int g = 0; g < M; g++){
            if (!survJ[bestJ][g]) continue;
            for (int i : coverList[bestJ]){
                int& c = cnt[g][i];
                if (c < pt[i]){
                    covG[g] += (double)pw[i] * (min(c + 1, pt[i]) - c) / (double)pt[i];
                    c++;
                }
            }
        }
        printf("%lld %lld\n", px[bestJ], py[bestJ]);
    }
    return 0;
}
