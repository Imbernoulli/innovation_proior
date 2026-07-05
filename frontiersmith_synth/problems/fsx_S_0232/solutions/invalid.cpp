// TIER: invalid
// Deliberately infeasible: release order 1 without ever loading it (precedence
// violation). The checker must reject this -> score 0.
#include <bits/stdc++.h>
using namespace std;
int main(){
    int P,Q; if(scanf("%d %d",&P,&Q)!=2) return 0;
    long long x0,y0; scanf("%lld %lld",&x0,&y0);
    for(int i=0;i<P;i++){ long long a,b,c,d,q,w; scanf("%lld %lld %lld %lld %lld %lld",&a,&b,&c,&d,&q,&w); }
    // one event: release order 1 with no prior load
    printf("1\n");
    printf("1 1\n");
    return 0;
}
