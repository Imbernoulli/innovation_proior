// TIER: invalid
// Deliberately infeasible: installs one copy of type 0 whose cells all sit on (0,0),
// so cells are duplicated/overlapping and not a legal footprint. Must score 0.
#include <bits/stdc++.h>
using namespace std;
int main(){
    int W,H,P; scanf("%d %d",&W,&H); scanf("%d",&P);
    vector<int> wval(P),rreq(P),stock(P),sz(P);
    for(int t=0;t<P;t++){scanf("%d %d %d %d",&wval[t],&rreq[t],&stock[t],&sz[t]);for(int i=0;i<sz[t];i++){int a,b;scanf("%d %d",&a,&b);}}
    printf("1\n");
    printf("0");
    for(int i=0;i<sz[0];i++)printf(" 0 0");   // all cells at origin -> duplicate/illegal shape
    printf("\n");
    return 0;
}
