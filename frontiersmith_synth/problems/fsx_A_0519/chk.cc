// checker for Pottery Kiln Firing Queue
#include "testlib.h"
#include <vector>
#include <algorithm>
using namespace std;
typedef long long ll;

static ll num_, den_;

// apply floor(T*num/den) k times (T>=0), short-circuit at 0
static ll coolT(ll T, ll k){
    while(k>0 && T>0){ T = (T*num_)/den_; k--; }
    return T;
}
// minimal idle so that cooled temp <= H
static ll minIdleTo(ll T, ll H){
    ll g=0;
    while(T>H){ T=(T*num_)/den_; g++; if(g>4000000) break; }
    return g;
}

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);
    int N   = inf.readInt(1, 2000, "N");
    ll cheat= inf.readLong(1LL, 10LL, "cheat");
    num_    = inf.readLong(1LL, 9LL, "cool_num");
    den_    = inf.readLong(2LL, 10LL, "cool_den");
    vector<ll> lo(N+1), hi(N+1), d(N+1), D(N+1), w(N+1);
    for(int i=1;i<=N;i++){
        lo[i]=inf.readLong(1LL,1000LL,"lo");
        hi[i]=inf.readLong(lo[i],1000LL,"hi");
        d[i] =inf.readLong(1LL,15LL,"d");
        D[i] =inf.readLong(1LL,10000000LL,"D");
        w[i] =inf.readLong(1LL,8LL,"w");
    }

    // ---- internal baseline B: input order, minimal idle into window, heat only to lo_i ----
    {
        ll T=0, t=0, cost=0;
        for(int i=1;i<=N;i++){
            ll g=minIdleTo(T, hi[i]);
            T=coolT(T,g); t+=g;
            ll F=max(lo[i], T);            // T<=hi[i]; lo<=hi so F<=hi; F>=T
            cost += cheat*(F-T);
            t += d[i];
            if(t>D[i]) cost += w[i]*(t-D[i]);
            T=F;
        }
        // ---- participant output ----
        ll pT=0, pt=0, pcost=0;
        vector<char> used(N+1,0);
        for(int j=0;j<N;j++){
            int id=ouf.readInt(1,N,"job_id");
            if(used[id]) quitf(_wa,"job %d fired twice",id);
            used[id]=1;
            ll idle=ouf.readLong(0LL,1000000000LL,"idle");
            ll F   =ouf.readLong(1LL,1000000LL,"F");
            pT=coolT(pT, idle); pt+=idle;
            if(pT>hi[id]) quitf(_wa,"job %d: temp %lld exceeds hi %lld after idling (idle more)",id,pT,hi[id]);
            if(F<lo[id]||F>hi[id]) quitf(_wa,"job %d: fire temp %lld outside window [%lld,%lld]",id,F,lo[id],hi[id]);
            if(F<pT) quitf(_wa,"job %d: fire temp %lld below current %lld (cannot cool without idling)",id,F,pT);
            pcost += cheat*(F-pT);
            pt += d[id];
            if(pt>D[id]) pcost += w[id]*(pt-D[id]);
            pT=F;
        }
        if(!ouf.seekEof()) quitf(_wa,"trailing output");
        for(int i=1;i<=N;i++) if(!used[i]) quitf(_wa,"job %d never fired",i);

        ll B=cost, Fp=pcost;
        if(B<1) B=1;
        // Smooth open-ceiling score: ratio = B / (B + 9*COST).
        //   trivial (COST==B) -> exactly 0.1; asymptotes toward 1 as COST->0 so the
        //   ceiling stays open above the reference solutions (no hard cap saturation).
        double denom = (double)B + 9.0*(double)Fp;
        double ratio = (denom>0.0) ? (double)B/denom : 0.0;
        if(ratio<0.0) ratio=0.0; if(ratio>1.0) ratio=1.0;
        quitp(ratio, "OK COST=%lld B=%lld Ratio: %.6f", Fp, B, ratio);
    }
}
