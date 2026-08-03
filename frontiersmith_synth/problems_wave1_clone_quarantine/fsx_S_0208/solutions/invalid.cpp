// TIER: invalid
// Deliberately infeasible: reads task 1 without ever mounting it (violates
// precedence) -> checker must score 0.
#include <bits/stdc++.h>
using namespace std;
int main(){
    int P; long long Q;
    scanf("%d %lld",&P,&Q);
    long long x0,y0; scanf("%lld %lld",&x0,&y0);
    for(int i=0;i<P;i++){ long long a,b,cc,d,q,c,w;
        scanf("%lld %lld %lld %lld %lld %lld %lld",&a,&b,&cc,&d,&q,&c,&w); }
    // one event: read (t=1) of task 1 with no prior mount
    printf("1\n1 1\n");
    return 0;
}
