// TIER: trivial
// Reproduces the checker's reference layout B: greedy cover in input order
// (hubs first) then fill the remaining pods with demand cells in input order.
// Scores ~0.1 by the baseline convention.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
static inline ll L1(ll ax,ll ay,ll bx,ll by){ return llabs(ax-bx)+llabs(ay-by); }

int main(){
    int W,H,m,R,D,a,b,K,r0;
    if(scanf("%d %d %d %d %d %d %d %d %d",&W,&H,&m,&R,&D,&a,&b,&K,&r0)!=9) return 0;
    vector<int> dx(D),dy(D);
    for(int i=0;i<D;i++) scanf("%d %d",&dx[i],&dy[i]);

    vector<ll> mind(D,LLONG_MAX);
    vector<char> used(D,0);
    vector<int> ox,oy; ox.reserve(m); oy.reserve(m);
    int placed=0;
    auto place=[&](int idx){
        used[idx]=1; placed++; ox.push_back(dx[idx]); oy.push_back(dy[idx]);
        for(int j=0;j<D;j++){ ll dd=L1(dx[idx],dy[idx],dx[j],dy[j]); if(dd<mind[j])mind[j]=dd; }
    };
    for(int i=0;i<D && placed<m;i++) if(mind[i]>R) place(i);
    while(placed<m){
        int best=-1; ll bv=-1;
        for(int j=0;j<D;j++){ if(used[j])continue; if(mind[j]>bv){bv=mind[j];best=j;} }
        if(best<0) break;
        place(best);
    }

    for(int i=0;i<placed;i++) printf("%d %d\n",ox[i],oy[i]);
    return 0;
}
