// TIER: trivial
// Reference dosing: root(node 1)=3, every other node=2. This is exactly the
// baseline B the checker measures, so it scores ratio ~= 0.1.
#include <bits/stdc++.h>
using namespace std;
int main(){
    int N,V; long long L,U;
    if(scanf("%d %d %lld %lld",&N,&V,&L,&U)!=4) return 0;
    for(int i=0;i<N-1;i++){ int a,b; scanf("%d %d",&a,&b); }
    for(int i=0;i<N;i++){ long long x; scanf("%lld",&x); }
    for(int i=1;i<=N;i++) printf("%d%c", i==1?3:2, i==N?'\n':' ');
    return 0;
}
