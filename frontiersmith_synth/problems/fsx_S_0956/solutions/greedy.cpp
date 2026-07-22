// TIER: greedy
// Ratio-greedy knapsack: for each hop, consider every catalog module that
// offers it, using ONLY its plain conversion ratio (byproduct fields are
// never inspected). Run a multiple-choice knapsack DP over the footprint
// budget to pick the best-ratio module per hop that fits S, then push the
// full raw supply through the resulting chain. This is the "obvious"
// pathfinding-style solution: it never notices that some byproduct feeds
// a module's own input.
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
                double val = dp[b] + log(outR[id] / 100.0);
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

    printf("%d %d\n", hops, hops + 1);
    for (int t = 0; t < hops; t++) printf("%d ", choice[t]);
    printf("\n");

    double amt = (double)R;
    printf("0 -1 0 1 0 %.6f\n", amt);
    for (int t = 0; t < hops; t++) {
        double outAmt = amt * (outR[choice[t]] / 100.0);
        if (t + 1 < hops) printf("1 %d 0 1 %d %.6f\n", t, t + 1, outAmt);
        else printf("1 %d 0 0 -1 %.6f\n", t, outAmt);
        amt = outAmt;
    }
    return 0;
}
