// TIER: greedy
// Obvious first instinct: absorb sound "where you hear it" -- install panels on
// the wall cells nearest the listener seats, cheapest-affordable first. Never
// looks at the phi2 coupling table at all, so it cannot tell a mode's antinode
// from its node -- it just trusts physical proximity to the ears.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(){
    int W,H,sx,sy; scanf("%d %d",&W,&H); scanf("%d %d",&sx,&sy);
    int NL; scanf("%d",&NL);
    vector<int> lx(NL), ly(NL);
    for(int i=0;i<NL;i++) scanf("%d %d",&lx[i],&ly[i]);
    int NM; scanf("%d",&NM);
    for(int k=0;k<NM;k++){ ll a,b,c,d; scanf("%lld %lld %lld %lld",&a,&b,&c,&d); }
    ll Budget; scanf("%lld",&Budget);
    int M; scanf("%d",&M);
    vector<int> X(M+1), Y(M+1); vector<ll> cost(M+1), alpha(M+1);
    for(int i=1;i<=M;i++){
        int id,x,y; ll c,a; scanf("%d %d %d %lld %lld",&id,&x,&y,&c,&a);
        X[i]=x; Y[i]=y; cost[i]=c; alpha[i]=a;
        for(int k=0;k<NM;k++){ ll v; scanf("%lld",&v); }
    }

    vector<int> order(M);
    for (int i=1;i<=M;i++) order[i-1]=i;
    auto distToNearestListener=[&](int i)->double{
        double best=1e18;
        for(int j=0;j<NL;j++){
            double dx=X[i]-lx[j], dy=Y[i]-ly[j];
            best=min(best, dx*dx+dy*dy);
        }
        return best;
    };
    sort(order.begin(), order.end(), [&](int a,int b){
        double da=distToNearestListener(a), db=distToNearestListener(b);
        if (da!=db) return da<db;
        return cost[a]<cost[b];
    });

    ll spent=0; vector<int> chosen;
    for (int id : order){
        if (spent + cost[id] <= Budget){ chosen.push_back(id); spent += cost[id]; }
    }
    printf("%d\n", (int)chosen.size());
    for (int id : chosen) printf("%d ", id);
    printf("\n");
    return 0;
}
