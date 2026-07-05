// TIER: invalid
// Deliberately infeasible: emits program 0, which is out of the allowed range [1,K].
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, M, K;
    if (scanf("%d %d %d", &N, &M, &K) != 3) return 0;
    for (int i = 0; i < M; i++) { int a,b,c; if(scanf("%d %d %d",&a,&b,&c)!=3) break; }
    for (int j = 0; j < N; j++) printf("0\n");   // 0 < 1 -> checker rejects
    return 0;
}
