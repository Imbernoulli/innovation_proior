#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

static inline long long manh(long long x1,long long y1,long long x2,long long y2){
    return llabs(x1-x2)+llabs(y1-y2);
}

int main(int argc,char**argv){
    registerTestlibCmd(argc,argv);

    // ----- read input -----
    int P = inf.readInt();
    int Q = inf.readInt();
    long long x0 = inf.readInt();
    long long y0 = inf.readInt();
    vector<long long> ax(P+1),ay(P+1),bx(P+1),by(P+1),q(P+1),w(P+1);
    long long B=0;
    for(int i=1;i<=P;i++){
        ax[i]=inf.readInt(); ay[i]=inf.readInt();
        bx[i]=inf.readInt(); by[i]=inf.readInt();
        q[i]=inf.readInt();  w[i]=inf.readInt();
        B += w[i];
    }
    if(B<1) B=1; // safety; generator guarantees B>=P>=1

    // ----- read participant output -----
    long long L = ouf.readLong(0, (long long)2*P, "L");

    vector<char> seenP(P+1,0), seenD(P+1,0);
    long long cur = 0;          // on-board load
    long long px=x0, py=y0;     // current position
    long long D = 0;            // load-weighted travel

    for(long long e=0;e<L;e++){
        int t = ouf.readInt(0,1,"t");
        int i = ouf.readInt(1,P,"i");
        long long tx, ty;
        if(t==0){ tx=ax[i]; ty=ay[i]; } else { tx=bx[i]; ty=by[i]; }
        // leg to this point at the load carried BEFORE the event
        D += manh(px,py,tx,ty)*(1+cur);
        if(t==0){
            if(seenP[i]) quitf(_wa,"order %d loaded twice",i);
            seenP[i]=1;
            cur += q[i];
            if(cur>Q) quitf(_wa,"capacity exceeded at event %lld: load %lld > Q %d",e+1,cur,Q);
        }else{
            if(seenD[i]) quitf(_wa,"order %d released twice",i);
            if(!seenP[i]) quitf(_wa,"order %d released before loaded (precedence)",i);
            seenD[i]=1;
            cur -= q[i];
            if(cur<0) quitf(_wa,"internal: negative load"); // unreachable
        }
        px=tx; py=ty;
    }
    if(!ouf.seekEof()) quitf(_wa,"trailing output after %lld events",L);

    // an order loaded but never released -> infeasible
    for(int i=1;i<=P;i++){
        if(seenP[i]!=seenD[i]) quitf(_wa,"order %d appears exactly once (unbalanced)",i);
    }
    // final leg back to depot (load must be 0 here)
    if(cur!=0) quitf(_wa,"internal: nonzero load at end"); // unreachable given balance
    D += manh(px,py,x0,y0)*(1+cur);

    // ----- objective -----
    long long F = D;
    for(int i=1;i<=P;i++) if(!seenP[i]) F += w[i]; // unserved penalties
    if(F<1) F=1;

    double sc = min(1000.0, 100.0*(double)B/(double)max(1LL,F));
    quitp(sc/1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc/1000.0);
    return 0;
}
