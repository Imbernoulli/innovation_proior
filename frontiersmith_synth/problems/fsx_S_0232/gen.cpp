#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Reservoir Cascade: selective load-weighted silt ferry generator.
// testId is a difficulty/structure ladder (1 tiny .. 10 large/adversarial).

static inline long long manh(long long x1,long long y1,long long x2,long long y2){
    return llabs(x1-x2)+llabs(y1-y2);
}

int main(int argc,char**argv){
    registerGen(argc,argv,1);
    int T=atoi(argv[1]);

    // ----- size ladder -----
    int P = 4 + 4*T;            // T=1 -> 8 ... T=10 -> 44
    if(P>50) P=50;
    if(T==1) P=6;               // tiny example-scale case

    // hull capacity varies by test to change how much batching is possible
    int Qchoices[4]={6,10,16,26};
    int Q = Qchoices[T%4];

    // depot near the middle of the grid
    int x0 = rnd.next(300,700);
    int y0 = rnd.next(300,700);

    // geometry mode: 0 clustered (batching pays off), 1 uniform, 2 mixed
    int mode = T%3;

    // penalty-scale mode: tune how profitable serving tends to be
    // factor drawn per order in [flo, fhi]; >1 tends profitable, <1 tends skip
    double flo, fhi;
    if(T%2==0){ flo=0.45; fhi=1.9; } else { flo=0.6; fhi=2.4; }

    // cluster centers (used in clustered/mixed modes)
    int nc = 3 + T%4;
    vector<pair<int,int>> cen(nc);
    for(int c=0;c<nc;c++) cen[c]={rnd.next(0,1000),rnd.next(0,1000)};

    auto samplePt=[&](bool clustered)->pair<int,int>{
        if(clustered){
            int c=rnd.next(0,nc-1);
            int spread=rnd.next(30,140);
            int x=cen[c].first + rnd.next(-spread,spread);
            int y=cen[c].second+ rnd.next(-spread,spread);
            x=max(0,min(1000,x)); y=max(0,min(1000,y));
            return {x,y};
        }else{
            return {rnd.next(0,1000),rnd.next(0,1000)};
        }
    };

    int qmax = min(Q,8);

    vector<array<int,6>> ord(P); // ax ay bx by q w
    for(int i=0;i<P;i++){
        bool clus;
        if(mode==0) clus=true;
        else if(mode==1) clus=false;
        else clus = (rnd.next(0,1)==0);
        auto a=samplePt(clus);
        auto b=samplePt(clus);
        int q=rnd.next(1,qmax);
        // solo load-weighted round-trip estimate: depot->a (empty) + a->b (loaded q) + b->depot (empty)
        long long solo = manh(x0,y0,a.first,a.second)
                       + manh(a.first,a.second,b.first,b.second)*(1+q)
                       + manh(b.first,b.second,x0,y0);
        double f = rnd.next(flo,fhi);
        long long w = (long long)llround((double)solo*f);
        if(w<1) w=1;
        if(w>200000) w=200000;
        ord[i]={a.first,a.second,b.first,b.second,q,(int)w};
    }

    // ----- emit -----
    printf("%d %d\n",P,Q);
    printf("%d %d\n",x0,y0);
    for(int i=0;i<P;i++){
        printf("%d %d %d %d %d %d\n",ord[i][0],ord[i][1],ord[i][2],ord[i][3],ord[i][4],ord[i][5]);
    }
    return 0;
}
