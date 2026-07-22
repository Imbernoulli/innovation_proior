// TIER: invalid
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int N, K; ll T;
    scanf("%d %d %lld", &N, &K, &T);
    vector<int> lens(K);
    for (int p = 0; p < K; p++) scanf("%d", &lens[p]);

    vector<int> Kv(N + 1);
    for (int v = 1; v <= N; v++) {
        int kv; scanf("%d", &kv); Kv[v] = kv;
        for (int i = 0; i < kv; i++) { ll d, a; scanf("%lld %lld", &d, &a); }
    }
    for (int p = 0; p < K; p++) {
        int L = lens[p];
        for (int i = 0; i < L - 1; i++) { ll w, c; scanf("%lld %lld", &w, &c); }
    }

    // Deliberately infeasible: out-of-range variant index for station 1.
    printf("999999 ");
    for (int v = 2; v <= N; v++) printf("0 ");
    printf("\n");
    int M = N - K;
    for (int i = 0; i < M; i++) printf("0 ");
    printf("\n");
    return 0;
}
