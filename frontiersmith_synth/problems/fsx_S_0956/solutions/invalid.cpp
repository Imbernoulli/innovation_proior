// TIER: invalid
// Deliberately infeasible: build one instance of EVERY catalog module type
// at once. The catalog always contains far more footprint than the budget
// S allows (S only covers the cheap spine plus a couple of loop
// investments), so this blows the footprint budget and must score 0.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int K, M; ll S, R;
    if (scanf("%d %d %lld %lld", &K, &M, &S, &R) != 4) return 0;
    vector<int> inT(M), outT(M), outR(M), byT(M), byR(M), fp(M), cnt(M);
    for (int i = 0; i < M; i++)
        scanf("%d %d %d %d %d %d %d", &inT[i], &outT[i], &outR[i], &byT[i], &byR[i], &fp[i], &cnt[i]);

    printf("%d 0\n", M);
    for (int i = 0; i < M; i++) printf("%d ", i);
    printf("\n");
    return 0;
}
