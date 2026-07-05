// TIER: invalid
// Deliberately infeasible: delivers job 1 before ever picking it up, violating
// the precedence rule. Must score 0.
#include <bits/stdc++.h>
using namespace std;
int main(){
    int N,Q; scanf("%d %d",&N,&Q);
    long long x0,y0; scanf("%lld %lld",&x0,&y0);
    for(int i=1;i<=N;i++){ long long a,b,c,d; int q; scanf("%lld %lld %lld %lld %d",&a,&b,&c,&d,&q); }
    printf("2\n");
    printf("1 1\n"); // deliver before pickup -> infeasible
    printf("0 1\n");
    return 0;
}
