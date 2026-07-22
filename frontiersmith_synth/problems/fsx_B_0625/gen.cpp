#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Avalanche Sculptor"  (generator)  family: abelian-sandpile-sculptor
//
// An N x N abelian sandpile. A cell with >=4 grains TOPPLES: it loses 4 grains and
// sends one to each orthogonal neighbour. Grains that leave the grid, or that fall
// onto a SINK cell, vanish. The boundary and the internal sink cells drain the pile,
// so any drop stabilizes to a unique config (values 0..3) independent of order.
//
// The instance publishes: the sink cells, a fixed set of K DROP SITES (sources), a
// per-cell match WEIGHT, and a TARGET stable configuration T. The solver chooses how
// many grains to drop at each source (total <= budget); the pile is stabilized and
// scored by weighted agreement with T.
//
// PLANTED STRUCTURE (the trap):  T is itself the stabilization of grains dropped at
// SEVERAL spread-out sources at once. Because toppling is abelian the joint pile is a
// linear superposition of the individual drops -- so the honest way to recover T is to
// solve for a combination across MANY sources. Dumping everything on the single best
// pile ("tune one pile") only ever reproduces one localized fractal lobe and misses the
// rest of the target. Internal sink walls (planted on some tests) split avalanches so a
// single source physically cannot reach the far side.
//
// Output format:
//   N K B
//   nSink
//   (nSink lines)  r c            internal sink cells
//   (K lines)      r c            source / drop sites (distinct, non-sink, interior)
//   (N lines)      N ints 0..3    target stable configuration T (sinks printed 0)
//   (N lines)      N ints         per-cell weight (sinks printed 0)
// -----------------------------------------------------------------------------

int N;
static inline int IDX(int r,int c){return r*N+c;}

void stabilize(vector<int>&g, const vector<char>&sink){
    int NN=N*N;
    vector<char> inq(NN,0);
    deque<int> q;
    for(int i=0;i<NN;i++) if(!sink[i]&&g[i]>=4){q.push_back(i);inq[i]=1;}
    static const int dr[4]={-1,1,0,0}, dc[4]={0,0,-1,1};
    while(!q.empty()){
        int c=q.front();q.pop_front();inq[c]=0;
        if(sink[c]||g[c]<4) continue;
        int t=g[c]/4; g[c]-=4*t;
        int r=c/N, cc=c%N;
        for(int d=0;d<4;d++){
            int nr=r+dr[d], nc=cc+dc[d];
            if(nr<0||nr>=N||nc<0||nc>=N) continue;   // off-grid: drain
            int ni=nr*N+nc;
            if(sink[ni]) continue;                    // sink: drain
            g[ni]+=t;
            if(g[ni]>=4 && !inq[ni]){inq[ni]=1;q.push_back(ni);}
        }
    }
}

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    double f = (testId - 1) / 9.0;

    N = 6 + (int)llround(f * 30.0);          // 6 .. 36
    int NN = N*N;
    vector<char> sink(NN, 0);

    // ---- internal sinks: light random scatter ----
    int nrand = (int)llround(0.035 * NN);
    for(int t=0;t<nrand;t++){
        int r = rnd.next(1, N-2), c = rnd.next(1, N-2);
        sink[IDX(r,c)] = 1;
    }
    // ---- planted sink WALL (boundary-shaping): splits avalanches on some tests ----
    if(testId % 3 == 0 && N >= 10){
        int col = N/2;
        int gap = rnd.next(1, N-2);               // one gap so the grid stays drained/connected
        for(int r=1;r<N-1;r++) if(r!=gap) sink[IDX(r,col)] = 1;
    }
    if(testId % 3 == 1 && N >= 14){
        int row = N/3 + rnd.next(0, N/3);
        int gap = rnd.next(1, N-2);
        for(int c=1;c<N-1;c++) if(c!=gap) sink[IDX(row,c)] = 1;
    }

    // ---- source / drop sites: spread on a lattice, snapped off sinks ----
    int Kwant = 2 + (int)llround(f * 18.0);       // 2 .. 20
    int side = max(2, (int)ceil(sqrt((double)(Kwant*2))));
    vector<int> cand;
    for(int i=0;i<side;i++) for(int j=0;j<side;j++){
        int r = (int)((i + 0.5) / side * N);
        int c = (int)((j + 0.5) / side * N);
        r = min(max(r,1), N-2);
        c = min(max(c,1), N-2);
        cand.push_back(IDX(r,c));
    }
    sort(cand.begin(), cand.end());
    cand.erase(unique(cand.begin(), cand.end()), cand.end());
    shuffle(cand.begin(), cand.end());
    vector<int> src;
    vector<char> used(NN,0);
    for(int id : cand){
        if((int)src.size() >= Kwant) break;
        if(sink[id] || used[id]) continue;
        src.push_back(id); used[id]=1;
    }
    // guarantee at least 2 sources even on tiny/over-sinked grids
    for(int id=0; id<NN && (int)src.size()<2; id++){
        if(!sink[id] && !used[id]){
            int r=id/N,c=id%N;
            if(r>=1&&r<=N-2&&c>=1&&c<=N-2){ src.push_back(id); used[id]=1; }
        }
    }
    int K = (int)src.size();

    // ---- true drops: cover the grid; skewed on "needle" tests ----
    ll perCover = max(6LL, (ll)llround(1.9 * NN / (double)K));
    bool needle = (testId % 4 == 2);              // one dominant pile amid small ones
    vector<int> g(NN, 0);
    ll total = 0;
    for(int i=0;i<K;i++){
        ll amt;
        if(needle && i==0)      amt = perCover * (4 + rnd.next(0,3));
        else if(needle)         amt = max(6LL, perCover/2 + rnd.next(0,(int)(perCover/3+1)));
        else                    amt = perCover + rnd.next(-(int)(perCover/3), (int)(perCover/3));
        amt = max(6LL, amt);
        g[src[i]] += (int)amt;
        total += amt;
    }
    stabilize(g, sink);                            // g is now the target T

    // ---- budget: generous headroom above the true drop mass ----
    ll B = (ll)llround(total * 2.2) + 10;

    // ---- per-cell match weights (hidden coefficients) ----
    // Matching a ZERO cell is cheap (weight 1); matching a raised cell is what the
    // sculpting is about (weight 5..9). This keeps the empty-grid baseline small so the
    // score has real headroom above a reconstruction.
    vector<int> w(NN, 0);
    for(int i=0;i<NN;i++) if(!sink[i]) w[i] = (g[i]==0) ? 1 : (5 + rnd.next(0, 4));

    // ---- emit ----
    vector<pair<int,int>> sinks;
    for(int i=0;i<NN;i++) if(sink[i]) sinks.push_back({i/N, i%N});

    printf("%d %d %lld\n", N, K, B);
    printf("%d\n", (int)sinks.size());
    for(auto &p : sinks) printf("%d %d\n", p.first, p.second);
    for(int i=0;i<K;i++) printf("%d %d\n", src[i]/N, src[i]%N);
    for(int r=0;r<N;r++){
        for(int c=0;c<N;c++) printf("%d%c", sink[IDX(r,c)]?0:g[IDX(r,c)], c+1<N?' ':'\n');
    }
    for(int r=0;r<N;r++){
        for(int c=0;c<N;c++) printf("%d%c", w[IDX(r,c)], c+1<N?' ':'\n');
    }
    return 0;
}
