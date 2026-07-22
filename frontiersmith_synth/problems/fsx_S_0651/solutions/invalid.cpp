// TIER: invalid
// Deliberately infeasible: every line claims a "stop" far outside [1,N], which
// the checker's bounded read must reject immediately.
#include <bits/stdc++.h>
using namespace std;

int main(){
    int N,M; scanf("%d %d", &N, &M);
    for(int i=0;i<M;i++){ int u,v; long long w; scanf("%d %d %lld", &u,&v,&w); }
    int L; long long BUDGET; scanf("%d %lld", &L, &BUDGET);
    for(int s=0;s<3;s++){
        int D; long long oracle; scanf("%d %lld", &D, &oracle);
        for(int i=0;i<D;i++){ long long u,v,w; scanf("%lld %lld %lld", &u,&v,&w); }
    }
    for(int i=0;i<L;i++) printf("2 1 999999\n");
    return 0;
}
