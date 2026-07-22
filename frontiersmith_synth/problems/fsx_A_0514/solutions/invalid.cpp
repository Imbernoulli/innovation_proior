// TIER: invalid
// Deliberately infeasible: declares a self-loop bridge (u == v), which the checker rejects. Must
// score exactly 0 -> confirms the checker validates feasibility.
#include <bits/stdc++.h>
using namespace std;
int main(){
    int n,m; long long B;
    if(scanf("%d %d %lld",&n,&m,&B)!=3) return 0;
    printf("1\n1 1\n");
    return 0;
}
