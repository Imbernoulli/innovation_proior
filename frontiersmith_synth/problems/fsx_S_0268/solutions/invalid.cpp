// TIER: invalid
// Deliberately infeasible: deliver contract 1 with no prior pickup (violates
// precedence / a contract appearing exactly once). Must score 0.
#include <bits/stdc++.h>
using namespace std;
int main(){
    int P; long long Q,K;
    if(scanf("%d %lld %lld",&P,&Q,&K)!=3) return 0;
    long long x0,y0; scanf("%lld %lld",&x0,&y0);
    for(int i=0;i<P;i++){ long long a,b,c2,d,m,c,w; scanf("%lld %lld %lld %lld %lld %lld %lld",&a,&b,&c2,&d,&m,&c,&w); }
    printf("1\n1 1\n"); // delivery of contract 1 before any pickup -> infeasible
    return 0;
}
