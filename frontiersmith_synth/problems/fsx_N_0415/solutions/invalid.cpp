// TIER: invalid
// Deliberately infeasible: prints m copies of the same cell (0,0) -> duplicate pods,
// which the checker rejects (and coverage is broken too). Must score 0.
#include <bits/stdc++.h>
using namespace std;
int main(){
    int W,H,m,R,D,a,b,K,r0;
    if(scanf("%d %d %d %d %d %d %d %d %d",&W,&H,&m,&R,&D,&a,&b,&K,&r0)!=9) return 0;
    for(int i=0;i<m;i++) printf("0 0\n");
    return 0;
}
