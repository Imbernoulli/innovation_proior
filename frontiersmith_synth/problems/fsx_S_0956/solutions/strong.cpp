// TIER: strong
// Loop-aware knapsack: for every candidate module compute its EFFECTIVE
// ratio. If the module vents a byproduct into its own input type (an
// autocatalytic self-loop with gain g = byRatio/100 < 1), routing the
// byproduct back multiplies steady-state throughput by 1/(1-g), so the
// effective ratio is outRatio/100 / (1-g) instead of the raw outRatio/100.
// Run the same footprint-budget knapsack DP as "greedy" but over these
// effective ratios, then actually wire the discovered self-loops: solve
// v = ext + g*v for each looped instance and route v*g back into itself.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int K, M; ll S, R;
    if (scanf("%d %d %lld %lld", &K, &M, &S, &R) != 4) return 0;
    vector<int> inT(M), outT(M), outR(M), byT(M), byR(M), fp(M), cnt(M);
    for (int i = 0; i < M; i++)
        scanf("%d %d %d %d %d %d %d", &inT[i], &outT[i], &outR[i], &byT[i], &byR[i], &fp[i], &cnt[i]);

    int hops = K - 1;
    vector<vector<int>> groups(hops);
    for (int i = 0; i < M; i++) if (inT[i] >= 0 && inT[i] < hops && outT[i] == inT[i] + 1) groups[inT[i]].push_back(i);

    auto effRatio = [&](int id) -> double {
        double r = outR[id] / 100.0;
        if (byT[id] == inT[id]) {
            double g = byR[id] / 100.0;
            if (g < 0.999) r = r / (1.0 - g);
        }
        return r;
    };

    int cap = (int)S;
    const double NEG = -1e18;
    vector<double> dp(cap + 1, NEG);
    dp[0] = 0.0;
    vector<vector<int>> choiceAt(hops, vector<int>(cap + 1, -1));

    for (int t = 0; t < hops; t++) {
        vector<double> ndp(cap + 1, NEG);
        for (int b = 0; b <= cap; b++) {
            if (dp[b] == NEG) continue;
            for (int id : groups[t]) {
                int f = fp[id];
                int nb = b + f;
                if (nb > cap) continue;
                double val = dp[b] + log(effRatio(id));
                if (val > ndp[nb]) { ndp[nb] = val; choiceAt[t][nb] = id; }
            }
        }
        dp = ndp;
    }

    int bestB = 0;
    for (int b = 1; b <= cap; b++) if (dp[b] > dp[bestB]) bestB = b;

    vector<int> choice(hops, -1);
    int b = bestB;
    for (int t = hops - 1; t >= 0; t--) {
        int id = choiceAt[t][b];
        choice[t] = id;
        b -= fp[id];
    }

    printf("%d ", hops);
    // edge count: 1 (raw) + hops chain edges, +1 extra self-loop edge per
    // looped instance
    int extra = 0;
    for (int t = 0; t < hops; t++) if (byT[choice[t]] == inT[choice[t]]) extra++;
    printf("%d\n", hops + 1 + extra);
    for (int t = 0; t < hops; t++) printf("%d ", choice[t]);
    printf("\n");

    double ext = (double)R;
    printf("0 -1 0 1 0 %.6f\n", ext);
    for (int t = 0; t < hops; t++) {
        int id = choice[t];
        double v, mainOut;
        bool looped = (byT[id] == inT[id]);
        double g = looped ? byR[id] / 100.0 : 0.0;
        if (looped && g < 0.999) v = ext / (1.0 - g); else v = ext;
        mainOut = v * (outR[id] / 100.0);
        if (looped) {
            double byAmt = v * g;
            printf("1 %d 1 1 %d %.6f\n", t, t, byAmt); // self-loop: instance t's byproduct -> instance t's own input
        }
        if (t + 1 < hops) printf("1 %d 0 1 %d %.6f\n", t, t + 1, mainOut);
        else printf("1 %d 0 0 -1 %.6f\n", t, mainOut);
        ext = mainOut;
    }
    return 0;
}
