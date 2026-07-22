#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// gen.cpp -- "Reinforcing Footbridges across a Branching Canyon System".
// Plants a NESTED-BOTTLENECK hierarchy: leaf clusters are high-diameter ring lattices whose
// inter-cluster bridges attach only inside a small PORT WINDOW (ring positions [0,pw)), so
// deep nodes (ring pos ~csize/2) are hop-FAR from every gateway. Cuts INCREASE with depth so
// the TOP bisection is the sparsest bottleneck (base lambda_2), and progressively deeper cuts
// bind only after it is repaired -> the bottleneck migrates across scales.
// testId is a difficulty ladder: 1 tiny (example scale) .. 10 full envelope.

int LV; int CS; int PW; int KR; long long BUD;
vector<int> CUTS;

void setparams(int t){
    switch(t){
        case 1:  LV=2; CS=14;  KR=3; PW=4; CUTS={2,3};    BUD=45;   break; // tiny (example)
        case 2:  LV=2; CS=40;  KR=3; PW=5; CUTS={2,5};    BUD=250;  break;
        case 3:  LV=2; CS=60;  KR=3; PW=6; CUTS={2,6};    BUD=480;  break;
        case 4:  LV=3; CS=40;  KR=3; PW=6; CUTS={2,4,8};  BUD=440;  break; // trap (3 scales)
        case 5:  LV=3; CS=48;  KR=3; PW=6; CUTS={2,4,8};  BUD=640;  break; // trap
        case 6:  LV=3; CS=44;  KR=3; PW=6; CUTS={2,4,8};  BUD=560;  break; // trap
        case 7:  LV=3; CS=64;  KR=3; PW=6; CUTS={2,5,10}; BUD=820;  break; // trap
        case 8:  LV=3; CS=80;  KR=4; PW=8; CUTS={3,6,12}; BUD=1150; break; // trap
        case 9:  LV=3; CS=100; KR=3; PW=6; CUTS={2,5,10}; BUD=1450; break; // trap
        default: LV=3; CS=128; KR=4; PW=8; CUTS={2,6,12}; BUD=2000; break; // full envelope
    }
}

int main(int argc, char** argv){
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);
    setparams(t);

    int leafcount = 1 << LV;
    int n = leafcount * CS;
    set<pair<int,int>> E;
    auto add=[&](int u,int v)->bool{
        if(u==v) return false;
        int a=min(u,v), b=max(u,v);
        if(E.count({a,b})) return false;
        E.insert({a,b}); return true;
    };

    // ---- ring-lattice leaf clusters (circulant: neighbours +-1..+-KR) ----
    for(int b=0;b<leafcount;b++){
        int base=b*CS;
        for(int i=0;i<CS;i++)
            for(int d=1;d<=KR;d++)
                add(base+i, base+(i+d)%CS);
    }

    // ---- hierarchical planted bridges (ports live in [0,PW) of each leaf) ----
    // recursively bisect the leaf range; connect the two halves' port pools with CUTS[level]
    // random bridges; deeper levels get MORE edges (top cut stays sparsest).
    function<void(int,int,int)> rec=[&](int lo,int hi,int level){
        if(hi-lo<=1) return;
        int mid=(lo+hi)/2;
        rec(lo,mid,level+1);
        rec(mid,hi,level+1);
        vector<int> Lp, Rp;
        for(int lf=lo;lf<mid;lf++) for(int j=0;j<PW;j++) Lp.push_back(lf*CS+j);
        for(int lf=mid;lf<hi;lf++) for(int j=0;j<PW;j++) Rp.push_back(lf*CS+j);
        int c = CUTS[level];
        for(int e=0;e<c;e++){
            for(int tryc=0;tryc<40;tryc++){
                int u=Lp[rnd.next((int)Lp.size())];
                int v=Rp[rnd.next((int)Rp.size())];
                if(add(u,v)) break;
            }
        }
    };
    rec(0,leafcount,0);

    // ---- emit ----
    printf("%d %d %lld\n", n, (int)E.size(), BUD);
    for(auto &pr:E) printf("%d %d\n", pr.first+1, pr.second+1);
    return 0;
}
