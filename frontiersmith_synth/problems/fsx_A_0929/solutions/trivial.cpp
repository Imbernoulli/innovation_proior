// TIER: trivial
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Do-nothing plan: every cabinet uses stride 1 at every level (m_c = D_c).
// This reproduces the checker's own baseline B exactly.
int main() {
    int K; ll M;
    scanf("%d %lld", &K, &M);
    for (int c = 0; c < K; c++) {
        int D; ll W;
        scanf("%d %lld", &D, &W);
        for (int d = 0; d <= D; d++) { ll t; scanf("%lld", &t); }
        printf("%d", D);
        for (int j = 0; j < D; j++) printf(" 1");
        printf("\n");
    }
    return 0;
}
