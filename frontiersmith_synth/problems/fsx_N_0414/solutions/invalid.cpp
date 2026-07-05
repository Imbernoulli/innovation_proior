// TIER: invalid
// Deliberately infeasible: assigns nutrient level 1 (forbidden; must be in [2,V])
// to every node. The checker's bounded read rejects it -> score 0.
#include <bits/stdc++.h>
using namespace std;
int main(){
    int N,V; long long L,U;
    if(scanf("%d %d %lld %lld",&N,&V,&L,&U)!=4) return 0;
    for(int i=0;i<N-1;i++){ int a,b; scanf("%d %d",&a,&b); }
    for(int i=0;i<N;i++){ long long x; scanf("%lld",&x); }
    for(int i=1;i<=N;i++) printf("1%c", i==N?'\n':' ');
    return 0;
}
