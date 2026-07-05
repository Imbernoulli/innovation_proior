// TIER: greedy
// Standalone-benefit selection: perform contract i iff its SOLO depot round-trip
// fuel + handling < omission penalty w_i. Route each accepted contract as an
// immediate pickup-then-deliver block, in input order. No batching, no reordering.
#include <bits/stdc++.h>
using namespace std;
static long long manh(long long x1,long long y1,long long x2,long long y2){
    return llabs(x1-x2)+llabs(y1-y2);
}
int main(){
    int P; long long Q,K;
    if(scanf("%d %lld %lld",&P,&Q,&K)!=3) return 0;
    long long x0,y0; scanf("%lld %lld",&x0,&y0);
    vector<long long> ax(P),ay(P),bx(P),by(P),m(P),c(P),w(P);
    for(int i=0;i<P;i++)
        scanf("%lld %lld %lld %lld %lld %lld %lld",&ax[i],&ay[i],&bx[i],&by[i],&m[i],&c[i],&w[i]);
    vector<int> sel;
    for(int i=0;i<P;i++){
        long long solo = manh(x0,y0,ax[i],ay[i])*K
                       + manh(ax[i],ay[i],bx[i],by[i])*(K+m[i])
                       + manh(bx[i],by[i],x0,y0)*K
                       + c[i];
        if(solo < w[i]) sel.push_back(i);
    }
    printf("%d\n", (int)sel.size()*2);
    for(int idx: sel){
        printf("0 %d\n", idx+1);
        printf("1 %d\n", idx+1);
    }
    return 0;
}
