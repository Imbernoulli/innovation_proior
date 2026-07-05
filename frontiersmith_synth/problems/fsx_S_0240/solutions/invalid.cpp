// TIER: invalid
// Deliberately infeasible: assign every task to step 1, violating windows and throughput.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int T, N, M;
    if (scanf("%d %d %d", &T, &N, &M) != 3) return 0;
    long long G; scanf("%lld", &G);
    for (int t = 0; t < T; t++) { long long a,b; int c; scanf("%lld %lld %d",&a,&b,&c); }
    for (int i = 0; i < N; i++) { int r,d; scanf("%d %d",&r,&d); }
    for (int i = 0; i < N; i++) printf("1\n");
    return 0;
}
