#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

static inline long long man(long long ax,long long ay,long long bx,long long by){
    return llabs(ax-bx)+llabs(ay-by);
}

int main(int argc,char*argv[]){
    registerTestlibCmd(argc,argv);

    long long Xd=inf.readInt(), Yd=inf.readInt();
    int m=inf.readInt();
    vector<long long> px(m),py(m),dx(m),dy(m),w(m);
    long long B=0;
    for(int j=0;j<m;j++){
        px[j]=inf.readInt(); py[j]=inf.readInt();
        dx[j]=inf.readInt(); dy[j]=inf.readInt();
        w[j]=inf.readInt();
        B+=w[j];
    }
    if(B<1) B=1; // penalties are positive so this holds; guard anyway

    int k=ouf.readInt(0, 2*m, "k");
    vector<char> seenP(m,0), seenD(m,0);
    long long prevX=Xd, prevY=Yd, dist=0;
    for(int i=0;i<k;i++){
        int t=ouf.readInt(1,m,"t"); t--;
        int s=ouf.readInt(0,1,"s");
        long long cx,cy;
        if(s==0){
            if(seenP[t]) quitf(_wa,"task %d picked up twice", t+1);
            seenP[t]=1; cx=px[t]; cy=py[t];
        } else {
            if(seenD[t]) quitf(_wa,"task %d delivered twice", t+1);
            if(!seenP[t]) quitf(_wa,"task %d delivered before pickup", t+1);
            seenD[t]=1; cx=dx[t]; cy=dy[t];
        }
        dist += man(prevX,prevY,cx,cy);
        prevX=cx; prevY=cy;
    }
    dist += man(prevX,prevY,Xd,Yd); // return to depot
    if(!ouf.seekEof()) quitf(_wa,"trailing output");

    long long unpaid=0;
    for(int j=0;j<m;j++){
        if(seenP[j] && !seenD[j]) quitf(_wa,"task %d picked up but never delivered", j+1);
        bool served = seenP[j] && seenD[j];
        if(!served) unpaid += w[j];
    }

    long long F = dist + unpaid;
    if(F<1) F=1;

    double sc = min(1000.0, 100.0*(double)B/(double)max(1LL,F));
    quitp(sc/1000.0, "OK F=%lld B=%lld dist=%lld Ratio: %.6f", F, B, dist, sc/1000.0);
    return 0;
}
