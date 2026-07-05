#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

static long long manh(long long x1,long long y1,long long x2,long long y2){
    return llabs(x1-x2)+llabs(y1-y2);
}

int main(int argc,char*argv[]){
    registerGen(argc,argv,1);
    int tid = atoi(argv[1]);

    // ---- size ladder: tid 1 tiny (example scale) -> tid 10 max ----
    int P;
    if(tid<=1) P = 4;
    else P = min(120, 4 + (tid-1)*13);

    // capacity / base burn
    int Q = rnd.next(6, 30);
    long long K = rnd.next(1, 8);

    // depot
    long long x0 = rnd.next(0,1600), y0 = rnd.next(0,1600);

    // spatial mode: 0 uniform, 1 clustered around a few hubs, 2 axis-skewed band
    int mode = tid % 3;
    int nHubs = rnd.next(2, 5);
    vector<pair<long long,long long>> hubs(nHubs);
    for(auto&h:hubs){ h.first=rnd.next(0,1600); h.second=rnd.next(0,1600); }

    auto samplePt=[&](void)->pair<long long,long long>{
        if(mode==1){
            auto&h=hubs[rnd.next(0,nHubs-1)];
            long long x = min(1600LL,max(0LL, h.first + rnd.next(-180,180)));
            long long y = min(1600LL,max(0LL, h.second + rnd.next(-180,180)));
            return {x,y};
        } else if(mode==2){
            // a wide band: y skewed toward a stripe
            long long x = rnd.next(0,1600);
            long long y = min(1600LL,max(0LL, 800LL + rnd.next(-250,250)));
            return {x,y};
        } else {
            return {rnd.next(0,1600), rnd.next(0,1600)};
        }
    };

    struct Task{ long long ax,ay,bx,by,m,c,w; };
    vector<Task> ts(P);
    for(int i=0;i<P;i++){
        auto a=samplePt(); auto b=samplePt();
        long long m = rnd.next(1,6);
        long long c = rnd.next(0,200);
        // solo depot round-trip fuel + handling
        long long solo = manh(x0,y0,a.first,a.second)*K
                       + manh(a.first,a.second,b.first,b.second)*(K+m)
                       + manh(b.first,b.second,x0,y0)*K
                       + c;
        // omission penalty: multiplier in [40,350]% of solo -> mix of profitable/unprofitable
        long long mult = rnd.next(40,350);
        long long w = max(1LL, solo*mult/100);
        ts[i] = {a.first,a.second,b.first,b.second,m,c,w};
    }

    // shuffle task order so input order carries no free routing signal
    shuffle(ts.begin(), ts.end());

    printf("%d %d %lld\n", P, Q, K);
    printf("%lld %lld\n", x0, y0);
    for(int i=0;i<P;i++){
        printf("%lld %lld %lld %lld %lld %lld %lld\n",
               ts[i].ax, ts[i].ay, ts[i].bx, ts[i].by, ts[i].m, ts[i].c, ts[i].w);
    }
    return 0;
}
