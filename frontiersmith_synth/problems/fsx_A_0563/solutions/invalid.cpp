// TIER: invalid
// Deliberately infeasible: observes the same target twice. Must score 0.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main(){
    int N; ll P,k; int K; ll R,Tb,SCALE,Dmax;
    scanf("%d %lld %lld %d %lld %lld %lld %lld",&N,&P,&k,&K,&R,&Tb,&SCALE,&Dmax);
    printf("2\n");
    printf("1 1\n");
    printf("1 1\n");   // duplicate observation -> checker rejects -> ratio 0
    return 0;
}
