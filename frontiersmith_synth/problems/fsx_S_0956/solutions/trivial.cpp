// TIER: trivial
// Naive single chain: for each hop t->t+1 use the first (smallest-id) catalog
// module that offers it, ignore every byproduct field, push the full raw
// supply straight through. Matches the checker's own internal baseline B,
// so this scores ratio ~= 0.1.
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
    vector<int> choice(hops, -1);
    for (int t = 0; t < hops; t++) {
        for (int i = 0; i < M; i++) if (inT[i] == t && outT[i] == t + 1) { choice[t] = i; break; }
    }

    printf("%d %d\n", hops, hops + 1);
    for (int t = 0; t < hops; t++) printf("%d ", choice[t]);
    printf("\n");

    double amt = (double)R;
    // RAW -> instance 0
    printf("0 -1 0 1 0 %.6f\n", amt);
    for (int t = 0; t < hops; t++) {
        double outAmt = amt * (outR[choice[t]] / 100.0);
        if (t + 1 < hops) {
            printf("1 %d 0 1 %d %.6f\n", t, t + 1, outAmt);
        } else {
            printf("1 %d 0 0 -1 %.6f\n", t, outAmt);
        }
        amt = outAmt;
    }
    return 0;
}
