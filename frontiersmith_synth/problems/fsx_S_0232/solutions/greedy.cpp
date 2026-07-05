// TIER: greedy
// Standalone-benefit greedy: serve every order whose penalty exceeds its solo
// load-weighted depot round-trip cost, each as an isolated load-then-release,
// ordered by benefit. Returns to the depot between orders (no batching).
#include <bits/stdc++.h>
using namespace std;
static inline long long manh(long long x1,long long y1,long long x2,long long y2){
    return llabs(x1-x2)+llabs(y1-y2);
}
int main(){
    int P,Q; if(scanf("%d %d",&P,&Q)!=2) return 0;
    long long x0,y0; scanf("%lld %lld",&x0,&y0);
    vector<long long> ax(P),ay(P),bx(P),by(P),q(P),w(P);
    for(int i=0;i<P;i++) scanf("%lld %lld %lld %lld %lld %lld",&ax[i],&ay[i],&bx[i],&by[i],&q[i],&w[i]);

    vector<pair<long long,int>> pick; // (benefit desc via negative), index
    for(int i=0;i<P;i++){
        long long solo = manh(x0,y0,ax[i],ay[i])
                       + manh(ax[i],ay[i],bx[i],by[i])*(1+q[i])
                       + manh(bx[i],by[i],x0,y0);
        long long benefit = w[i]-solo;
        if(benefit>0) pick.push_back({-benefit,i});
    }
    sort(pick.begin(),pick.end());

    printf("%d\n",(int)pick.size()*2);
    for(auto&pr:pick){
        int i=pr.second;
        printf("0 %d\n",i+1);
        printf("1 %d\n",i+1);
    }
    return 0;
}
