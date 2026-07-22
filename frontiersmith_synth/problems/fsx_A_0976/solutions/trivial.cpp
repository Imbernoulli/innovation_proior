// TIER: trivial
// Do-nothing baseline: take the first S candidate sites in the exact order they
// appear in the input (this mirrors the checker's own internal baseline B).
#include <bits/stdc++.h>
using namespace std;

int main(){
    int N,M,C,S,K;
    scanf("%d %d %d %d %d", &N,&M,&C,&S,&K);
    for (int i=0;i<N;i++){ long long x; scanf("%lld",&x); }
    for (int i=0;i<M;i++){ int u,v; long long c; scanf("%d %d %lld",&u,&v,&c); }
    // don't even need to read candidates/scenarios for this trivial pick
    for (int i=0;i<S;i++) printf("%d ", i);
    printf("\n");
    return 0;
}
