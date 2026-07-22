// TIER: invalid
// Deliberately infeasible: seed one of the rival's own hub nodes (already B) and
// repeat a duplicate id among the K outputs. Must score 0.
#include <bits/stdc++.h>
using namespace std;

int main(){
    int N, M, K, S;
    scanf("%d %d %d %d", &N, &M, &K, &S);
    for (int i = 1; i <= N; i++){ long long x; scanf("%lld", &x); }
    for (int i = 0; i < M; i++){ int u, v; scanf("%d %d", &u, &v); }
    int firstRival = -1;
    for (int i = 0; i < S; i++){ int b; scanf("%d", &b); if (firstRival == -1) firstRival = b; }
    // print the rival's hub id (illegal) followed by K-1 copies of node 1 (duplicates)
    printf("%d\n", firstRival);
    for (int i = 1; i < K; i++) printf("1\n");
    return 0;
}
