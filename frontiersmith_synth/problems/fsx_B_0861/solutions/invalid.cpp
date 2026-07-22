// TIER: invalid
// Deliberately infeasible: prints stop 2 twice (and omits some other required stop),
// which the checker's seen[] duplicate check must reject -> no Ratio -> scores 0.
#include <bits/stdc++.h>
using namespace std;
int main(){
    int N, M, K;
    scanf("%d %d %d", &N, &M, &K);
    for (int i = 0; i < M; i++){ long long u,v,w; scanf("%lld %lld %lld", &u,&v,&w); }
    for (int s = 0; s < K; s++){
        int B; long long w; scanf("%d %lld", &B, &w);
        for (int j = 0; j < B; j++){ int id; scanf("%d", &id); }
    }
    for (int i = 0; i < N-1; i++) printf("2%c", i+1<N-1?' ':'\n');
    return 0;
}
