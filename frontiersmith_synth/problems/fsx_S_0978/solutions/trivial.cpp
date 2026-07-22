// TIER: trivial
// Block-and-singles baseline: keep trucks in input order (inbound get slots
// 1..T_in, outbound truck j gets slot T_in+j) and move every pallet as its own
// single-pallet trip. This is exactly the construction the checker uses for its
// internal baseline B, so this solution scores ratio ~0.1 on every test.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int Tin, Tout, M, K, cap;
    scanf("%d %d %d %d %d", &Tin, &Tout, &M, &K, &cap);
    vector<int> pi_(M), pj_(M), pf_(M);
    for (int e = 0; e < M; e++) scanf("%d %d %d", &pi_[e], &pj_[e], &pf_[e]);

    int D = Tin + Tout;
    for (int k = 1; k <= D; k++) printf("%d%c", k, k == D ? '\n' : ' ');

    for (int e = 0; e < M; e++) {
        int f = pf_[e];
        printf("%d", f);
        for (int t = 0; t < f; t++) printf(" 1");
        printf("\n");
    }
    return 0;
}
