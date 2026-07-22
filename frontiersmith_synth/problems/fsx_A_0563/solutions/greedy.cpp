// TIER: greedy
// The obvious recipe: run the shortest tour (coordinate sweep) and spend the
// recalibration budget UNIFORMLY along it (a reset every N/(K+1) targets).
// This spreads the drift budget evenly instead of aligning it to value -> it is
// the trap the far high-value clusters are built to punish.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(){
    int N; ll P,k; int K; ll R,Tb,SCALE,Dmax;
    scanf("%d %lld %lld %d %lld %lld %lld %lld",&N,&P,&k,&K,&R,&Tb,&SCALE,&Dmax);
    vector<ll> px(N+1), pv(N+1);
    for(int i=1;i<=N;i++) scanf("%lld %lld",&px[i],&pv[i]);
    vector<int> ord(N); for(int i=0;i<N;i++) ord[i]=i+1;
    sort(ord.begin(),ord.end(),[&](int a,int b){ if(px[a]!=px[b]) return px[a]<px[b]; return a<b;});

    // reset boundaries: before sorted position round(v*N/(K+1)), v=1..K
    set<int> resetAt;
    for(int v=1; v<=K; v++){
        int pos = (int)llround((double)v * (double)N / (double)(K+1));
        if(pos>=1 && pos<N) resetAt.insert(pos);
    }
    // build actions
    vector<pair<int,int>> acts; // (type, id) type1=visit id, type0=reset
    for(int j=0;j<N;j++){
        if(resetAt.count(j)) acts.push_back({0,0});
        acts.push_back({1, ord[j]});
    }
    printf("%d\n",(int)acts.size());
    for(auto&a: acts){ if(a.first==0) printf("0\n"); else printf("1 %d\n", a.second); }
    return 0;
}
