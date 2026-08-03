// TIER: trivial
// Do-nothing: put every zone on program 1. Matches the checker baseline -> ratio ~0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, M, K;
    if (scanf("%d %d %d", &N, &M, &K) != 3) return 0;
    // consume edges (not needed)
    for (int i = 0; i < M; i++) { int a,b,c; if(scanf("%d %d %d",&a,&b,&c)!=3) break; }
    for (int j = 0; j < N; j++) printf("1\n");
    return 0;
}
