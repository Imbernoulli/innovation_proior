// Generator for "Gorge Bridge from Rationed Steel" (frame-truss-load-ceiling).
// Emits: H W Budget  /  H per-level rock ceilings C[0..H-1].
// Structure ladder: size grows with testId; most cases PLANT a weak rock level (a
// deep or middle stratum with a low ceiling) that is the true load bottleneck.  The
// obvious "uniform truss" heuristic leaves the weak level under-reinforced; a min-cut
// aware builder pours its spare steel into exactly that stratum.
#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc,char**argv){
    registerGen(argc,argv,1);
    int t=atoi(argv[1]);

    int Hs[10]={2,2,3,3,4,4,5,5,6,6};
    int Ws[10]={2,3,3,4,4,5,5,6,6,7};
    int H=Hs[t-1], W=Ws[t-1];

    // per-level ceilings
    vector<long long> C(H);
    int Chi = 44 + (t%3)*4;              // strong-rock ceiling ~44..52
    int Clo = 18 + (t%2)*3;              // weak-rock ceiling ~18..21
    int weak = -1;
    if(t>=3){
        if(t%2==1) weak = H-1;           // deep stratum
        else       weak = H/2;           // mid stratum
    }
    for(int i=0;i<H;i++){
        long long v = Chi + rnd.next(0,6); // jitter so it isn't perfectly flat
        C[i]=v;
    }
    if(weak>=0) C[weak]=Clo;             // the planted bottleneck

    // full single-diagonal triangulated sheet cost (reference span the solver should cover)
    long long sheet = (long long)(W+1)*H + (long long)(H+1)*W + 2LL*H*W;
    // spare steel = enough to X-brace roughly ~1.3 rock levels (water-filling budget)
    long long slack = (long long)llround(2.0*W*1.3) + 1;
    long long Budget = sheet + slack;

    printf("%d %d %lld\n", H, W, Budget);
    for(int i=0;i<H;i++) printf("%lld%c", C[i], i+1<H?' ':'\n');
    return 0;
}
