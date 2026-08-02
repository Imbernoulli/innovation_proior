// TIER: invalid
// deliberately infeasible: emits an out-of-range coefficient (|A_i|>1000) on the first
// machine line -- the checker's bounded ouf.readInt(-1000,1000,...) must reject this.
#include <bits/stdc++.h>
using namespace std;
int main(){
    int n,m;
    if(scanf("%d %d",&n,&m)!=2) return 0;
    for(int j=0;j<n;j++){
        for(int i=0;i<m;i++){ long long x; scanf("%lld",&x); }
        long long ww; scanf("%lld",&ww);
    }
    printf("999999 0\n");
    for(int i=1;i<m;i++) printf("1 0\n");
    return 0;
}
