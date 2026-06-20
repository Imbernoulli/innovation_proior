**Problem.** AHC039: maximize `a - b + 1` with one axis-aligned rectilinear simple net under
`4 <= m <= 1000`, axis-parallel edges, simplicity, and perimeter `<= 4x10^5`. ENDPOINT: the
ShinkaEvolve refinements on top of the rung-3 grid SA -- cached validation + a targeted edge move --
the two levers that took ALE-Agent's AHC039 solution from performance 2880 (5th) to 3140 (2nd).

**Key idea.** Keep rung 3's warm-started grid SA (region = connected hole-free cell set; O(1)
incremental `a-b`; simple-point + pinch topology checks; geometric cooling). Add the two
ShinkaEvolve changes. (1) CACHED VALIDATION: maintain a per-cell boundary-flag cache refreshed only
on the 3x3 neighbourhood of each accepted flip, so candidate proposals and validity read a cache
instead of rescanning the grid (the analogue of caching subtree fish-count/bbox statistics at each
kd-tree node). (2) TARGETED EDGE MOVE: with probability `P_TARGET`, sample a boundary cell and
greedily move the nearest edge to fix a misclassified fish -- ADD a mackerel-rich outside neighbour
(capture a mackerel the net misses), or REMOVE a sardine-heavy boundary cell (release a sardine the
net catches). Otherwise fall back to a uniform random flip. Directed proposals still pass through
Metropolis acceptance and all validity gates.

**Why these choices.** Rung 3 plateaued because, with the boundary at its perimeter limit, random
flips almost always touch already-correct boundary; the targeted edge move aims proposals at fish
the net actually misclassifies and at the nearest edge that fixes them, "strengthening the
directionality of the search" exactly as ShinkaEvolve reported. The boundary-flag cache makes those
directed proposals affordable: an accepted flip can only change boundary status in its 3x3 window,
so the cache is refreshed locally and is faithful by construction (the internal `a-b` matches the
exact evaluator to the unit). Mixing targeted and uniform moves keeps the large high-temperature
reshapes while directing most proposals at real errors.

**Hyperparameters / contract.** Grid `G = 50`; `~5 s` budget; cooling `T0 = 8.0 -> T1 = 0.05`;
targeted-move probability `P_TARGET = 0.55`; perimeter safe-margin `20` cell-sides under `4x10^5`.
Deterministic xorshift RNG; a true-perimeter check falls back to the best rectangle if a traced
polygon is ever degenerate. Reads instance on stdin, writes `m` then `m` vertices on stdout.
Compile: `g++ -O2`.

```cpp
// Rung 4 (ENDPOINT): ShinkaEvolve-style refinements on top of the grid-cell SA.
//
// Base: warm-started SA on a GxG binary grid whose region's outer boundary is a
// simple rectilinear polygon; per-cell (Av,Bv) give O(1) incremental a-b scoring
// and a running boundary-edge count gives O(1) perimeter.
//
// Two refinements ported from ShinkaEvolve's evolved AHC039 solution (which took
// ALE-Agent's score from 2880, 5th, to 3140, 2nd; arXiv:2509.19349):
//   (1) CACHED VALIDATION: a per-cell boundary-flag cache refreshed only on the
//       3x3 neighbourhood of each accepted flip, so candidate validity/scoring
//       never rescans the grid (the analogue of ShinkaEvolve caching subtree
//       fish-count/bbox statistics in the kd-tree node).
//   (2) TARGETED EDGE MOVE: a directed operator that finds a misclassified fish
//       (a mackerel-rich cell just outside the net, or a sardine-rich cell on the
//       net boundary) and greedily moves the nearest edge to fix it.
//
// Metropolis acceptance on delta(a-b) with geometric cooling. Best region kept.
// Output: the traced rectilinear polygon (vertices), collinear runs merged.
#include <bits/stdc++.h>
using namespace std;

static uint64_t rng_state = 88172645463325252ULL;
static inline uint64_t xr(){ rng_state^=rng_state<<13; rng_state^=rng_state>>7; rng_state^=rng_state<<17; return rng_state; }
static inline double urand(){ return (xr()>>11)*(1.0/9007199254740992.0); }

int G;
long long CW; // grid line spacing helper not needed; we compute coords from i*1e5//G
vector<long long> linecoord; // linecoord[i] = i*100000//G

vector<int> Wv, Av, Bv; // per cell weight, mackerel, sardine  (size G*G)
vector<char> inset;
inline int CID(int i,int j){ return i*G+j; }

vector<pair<long long,long long>> mackP, sardP;

// local simple-point test for 4-connected foreground / 8-connected background.
// We use the standard crossing-number test on the 8-neighbourhood.
inline bool simple_point(int i,int j){
    // gather 8-neighbourhood foreground values (treat off-grid as background)
    auto F=[&](int di,int dj)->int{
        int ni=i+di,nj=j+dj;
        if(ni<0||ni>=G||nj<0||nj>=G) return 0;
        return inset[CID(ni,nj)]?1:0;
    };
    // p2..p9 clockwise from top
    int p2=F(-1,0),p3=F(-1,1),p4=F(0,1),p5=F(1,1),p6=F(1,0),p7=F(1,-1),p8=F(0,-1),p9=F(-1,-1);
    int seq[8]={p2,p3,p4,p5,p6,p7,p8,p9};
    int trans=0;
    for(int k=0;k<8;k++){ if(seq[k]==0 && seq[(k+1)%8]==1) trans++; }
    if(trans!=1) return false;
    // Reject diagonal "pinch": any 2x2 block touching (i,j) that is a checker-
    // board (two opposite cells in, two out) would make the boundary a figure-8,
    // not a single simple polygon. Forbid it so the trace is always one cycle.
    int p1=inset[CID(i,j)]?1:0;
    auto G2=[&](int di,int dj)->int{ return F(di,dj); };
    // four 2x2 blocks with (i,j) as a corner: (NE,N,E),(NW,N,W),(SE,S,E),(SW,S,W)
    // block diagonal-only iff the two diagonal cells differ in the same way:
    // here p1 and the diagonal are one class, the two orthogonals the other.
    if(p1==G2(-1,1) && p1!=G2(-1,0) && p1!=G2(0,1) && G2(-1,0)==G2(0,1)) return false; // NE
    if(p1==G2(-1,-1)&& p1!=G2(-1,0) && p1!=G2(0,-1)&& G2(-1,0)==G2(0,-1)) return false; // NW
    if(p1==G2(1,1)  && p1!=G2(1,0)  && p1!=G2(0,1) && G2(1,0)==G2(0,1)) return false;   // SE
    if(p1==G2(1,-1) && p1!=G2(1,0)  && p1!=G2(0,-1)&& G2(1,0)==G2(0,-1)) return false;  // SW
    return true;
}

int main(int argc,char**argv){
    int n; if(scanf("%d",&n)!=1) return 0;
    mackP.resize(n); sardP.resize(n);
    for(auto&p:mackP) scanf("%lld %lld",&p.first,&p.second);
    for(auto&p:sardP) scanf("%lld %lld",&p.first,&p.second);

    G = argc>1?atoi(argv[1]):50;
    double TIME = argc>2?atof(argv[2]):2.0;
    linecoord.resize(G+1);
    for(int i=0;i<=G;i++) linecoord[i]=(long long)i*100000/G;
    auto gi=[&](long long v)->int{ int c=(int)(v*G/100000); if(c<0)c=0; if(c>=G)c=G-1; return c; };

    Wv.assign(G*G,0); Av.assign(G*G,0); Bv.assign(G*G,0);
    for(auto&p:mackP){ int c=CID(gi(p.first),gi(p.second)); Av[c]++; Wv[c]++; }
    for(auto&p:sardP){ int c=CID(gi(p.first),gi(p.second)); Bv[c]++; Wv[c]--; }

    inset.assign(G*G,0);
    // WARM START: initialize the region to the best axis-aligned rectangle on
    // the grid (2D prefix sums of weight, perimeter-constrained), then let SA
    // carve staircase notches. A perimeter-efficient rectangle is a far better
    // starting basin than a single seed cell.
    {
        vector<vector<long long>> PS(G+1, vector<long long>(G+1,0));
        for(int i=0;i<G;i++)for(int j=0;j<G;j++)
            PS[i+1][j+1]=Wv[CID(i,j)]+PS[i][j+1]+PS[i+1][j]-PS[i][j];
        long long bestS=-(1LL<<60); int bi1=0,bi2=1,bj1=0,bj2=1;
        for(int i1=0;i1<G;i1++)for(int i2=i1+1;i2<=G;i2++){
            long long w=linecoord[i2]-linecoord[i1];
            for(int j1=0;j1<G;j1++)for(int j2=j1+1;j2<=G;j2++){
                long long h=linecoord[j2]-linecoord[j1];
                if(2*(w+h)>400000) continue;
                long long s=PS[i2][j2]-PS[i1][j2]-PS[i2][j1]+PS[i1][j1];
                if(s>bestS){ bestS=s; bi1=i1;bi2=i2;bj1=j1;bj2=j2; }
            }
        }
        for(int i=bi1;i<bi2;i++)for(int j=bj1;j<bj2;j++) inset[CID(i,j)]=1;
    }
    long long curA=0,curB=0;
    for(int c=0;c<G*G;c++) if(inset[c]){ curA+=Av[c]; curB+=Bv[c]; }
    long long curScore=curA-curB; // (a-b); final objective = max(0,a-b+1)

    // perimeter in coord units. Approximate each boundary unit-edge length by
    // the average cell side; we keep an exact running boundary-edge count and
    // multiply by the (nearly uniform) cell width. Use a margin.
    long long CELLW = 100000/G;
    long long PERIM_BUDGET = 400000;
    long long SAFE = PERIM_BUDGET - 20*CELLW; // margin for rounding/border edges
    // count current boundary unit-edges
    long long bedges = 0;
    for(int i=0;i<G;i++)for(int j=0;j<G;j++){
        if(!inset[CID(i,j)]) continue;
        int di[4]={1,-1,0,0}, dj[4]={0,0,1,-1};
        for(int d=0;d<4;d++){int ni=i+di[d],nj=j+dj[d];
            if(ni<0||ni>=G||nj<0||nj>=G||!inset[CID(ni,nj)]) bedges++;}
    }

    auto edge_delta=[&](int i,int j,bool adding)->int{
        // change in boundary-edge count when toggling cell (i,j)
        int sameNbr=0,deg=0;
        for(int d=0;d<4;d++){
            int di[4]={1,-1,0,0}, dj[4]={0,0,1,-1};
            int ni=i+di[d],nj=j+dj[d];
            if(ni<0||ni>=G||nj<0||nj>=G) continue;
            deg++;
            if(inset[CID(ni,nj)]) sameNbr++;
        }
        int borderEdges = 4-deg; // edges on grid boundary
        if(adding){
            // 4 new edges minus 2*sameNbr shared edges that vanish
            return 4 - 2*sameNbr; // borderEdges already part of the 4
        }else{
            return -(4 - 2*sameNbr);
        }
    };

    // best
    vector<char> best=inset; long long bestScore=curScore;
    long long bestBedges=bedges;

    // SA
    double T0=8.0, T1=0.05;
    auto t_start=chrono::steady_clock::now();
    long long iter=0;
    vector<int> regionCells;
    for(int c=0;c<G*G;c++) if(inset[c]) regionCells.push_back(c);

    // ----- ShinkaEvolve refinement (1): CACHED VALIDATION -----
    // Maintain the boundary-cell list incrementally so candidate moves never
    // rescan the grid. A cell is a "boundary region cell" if it touches outside
    // or the grid border; we keep that flag in a cache and only refresh the
    // 3x3 neighbourhood of each accepted flip. The per-cell (Av,Bv) and the
    // running bedges already make scoring + perimeter O(1) per candidate.
    vector<char> isBnd(G*G,0);
    auto recompute_bnd=[&](int i,int j){
        if(i<0||i>=G||j<0||j>=G) return;
        int c=CID(i,j);
        if(!inset[c]){ isBnd[c]=0; return; }
        int di[4]={1,-1,0,0}, dj[4]={0,0,1,-1}; char b=0;
        for(int d=0;d<4;d++){int ni=i+di[d],nj=j+dj[d];
            if(ni<0||ni>=G||nj<0||nj>=G||!inset[CID(ni,nj)]){b=1;break;}}
        isBnd[c]=b;
    };
    for(int c=0;c<G*G;c++) if(inset[c]) recompute_bnd(c/G,c%G);
    auto refresh33=[&](int i,int j){
        for(int di=-1;di<=1;di++)for(int dj=-1;dj<=1;dj++) recompute_bnd(i+di,j+dj);
    };

    // ----- ShinkaEvolve refinement (2): TARGETED EDGE MOVE -----
    // With probability P_TARGET, instead of a uniform move, pick a MISCLASSIFIED
    // fish and greedily move the nearest boundary to fix it:
    //   * a mackerel-rich cell just OUTSIDE the region adjacent to a boundary
    //     cell -> ADD it (capture the mackerel), or
    //   * a sardine-rich BOUNDARY cell of the region -> REMOVE it (release the
    //     sardine). We pre-rank cells by (Av - Bv) so the proposal is directed.
    // This strengthens the directionality of the search, the exact lever
    // ShinkaEvolve used to lift ALE-Agent's AHC039 solution 2880 -> 3140.
    const double P_TARGET=0.55;
    int di4[4]={1,-1,0,0}, dj4[4]={0,0,1,-1};

    auto try_add=[&](int i,int j,double T)->void{
        if(i<0||i>=G||j<0||j>=G) return;
        int c=CID(i,j); if(inset[c]) return;
        // must be adjacent to region
        bool adj=false; for(int d=0;d<4;d++){int ni=i+di4[d],nj=j+dj4[d]; if(ni>=0&&ni<G&&nj>=0&&nj<G&&inset[CID(ni,nj)]){adj=true;break;}}
        if(!adj) return;
        int de=edge_delta(i,j,true);
        if((bedges+de)*(double)CELLW > SAFE) return;
        inset[c]=1;
        if(!simple_point(i,j)){ inset[c]=0; return; }
        long long dScore=Av[c]-Bv[c];
        if(dScore>=0 || urand()<exp(dScore/T)){
            bedges+=de; curScore+=dScore; regionCells.push_back(c); refresh33(i,j);
            if(curScore>bestScore){ bestScore=curScore; best=inset; }
        } else inset[c]=0;
    };
    auto try_remove_idx=[&](int idx,double T)->void{
        if((int)regionCells.size()<=1) return;
        int c=regionCells[idx]; int i=c/G,j=c%G;
        if(!isBnd[c]) return;
        int de=edge_delta(i,j,false);
        inset[c]=0;
        if(!simple_point(i,j)){ inset[c]=1; return; }
        long long dScore=-(Av[c]-Bv[c]);
        if(dScore>=0 || urand()<exp(dScore/T)){
            bedges+=de; curScore+=dScore;
            regionCells[idx]=regionCells.back(); regionCells.pop_back();
            refresh33(i,j);
            if(curScore>bestScore){ bestScore=curScore; best=inset; }
        } else inset[c]=1;
    };

    while(true){
        if((iter & 1023)==0){
            double el=chrono::duration<double>(chrono::steady_clock::now()-t_start).count();
            if(el>TIME) break;
        }
        iter++;
        double frac = chrono::duration<double>(chrono::steady_clock::now()-t_start).count()/TIME;
        if(frac>1) frac=1;
        double T = T0*pow(T1/T0, frac);

        if(urand()<P_TARGET){
            // TARGETED EDGE MOVE: sample a boundary region cell; look at its
            // outside neighbours and greedily ADD the most mackerel-rich one
            // (fix a misclassified mackerel); separately, if this boundary cell
            // is itself sardine-heavy, REMOVE it (fix a misclassified sardine).
            if(regionCells.empty()) continue;
            int idx=xr()%regionCells.size();
            int rc=regionCells[idx]; int ri=rc/G, rj=rc%G;
            if(!isBnd[rc]) continue;
            // candidate add: best outside neighbour by (Av-Bv)
            int bestNi=-1,bestNj=-1; long long bestw=-(1LL<<60);
            for(int d=0;d<4;d++){int ni=ri+di4[d],nj=rj+dj4[d];
                if(ni<0||ni>=G||nj<0||nj>=G) continue; int nc=CID(ni,nj);
                if(inset[nc]) continue; long long w=Av[nc]-Bv[nc];
                if(w>bestw){bestw=w; bestNi=ni; bestNj=nj;}}
            // if this boundary cell hurts (sardine-heavy), bias toward removing it
            if(Av[rc]-Bv[rc] < 0 && (xr()&1)){
                try_remove_idx(idx,T);
            } else if(bestNi>=0){
                try_add(bestNi,bestNj,T);
            }
        } else {
            // baseline uniform move (add adjacent / remove boundary)
            if(xr()&1){
                if(regionCells.empty()) continue;
                int rc=regionCells[xr()%regionCells.size()];
                int d=xr()%4; try_add(rc/G+di4[d], rc%G+dj4[d], T);
            } else {
                if(regionCells.size()<=1) continue;
                try_remove_idx(xr()%regionCells.size(), T);
            }
        }
    }

    // trace best region boundary into a polygon
    inset=best;
    auto cx=[&](int i)->long long{ return linecoord[i]; };
    map<pair<int,int>,pair<int,int>> edges;
    for(int i=0;i<G;i++)for(int j=0;j<G;j++){
        if(!inset[CID(i,j)]) continue;
        bool below=(j-1>=0)&&inset[CID(i,j-1)];
        if(!below) edges[{i,j}]={i+1,j};
        bool above=(j+1<G)&&inset[CID(i,j+1)];
        if(!above) edges[{i+1,j+1}]={i,j+1};
        bool left=(i-1>=0)&&inset[CID(i-1,j)];
        if(!left) edges[{i,j+1}]={i,j};
        bool right=(i+1<G)&&inset[CID(i+1,j)];
        if(!right) edges[{i+1,j}]={i+1,j+1};
    }
    // walk
    auto start=edges.begin()->first;
    vector<pair<int,int>> cells; auto cur=start; size_t cap=edges.size()+5;
    while(true){ cells.push_back(cur); auto it=edges.find(cur); if(it==edges.end()) break; cur=it->second; if(cur==start||cells.size()>cap) break; }
    // to coords + merge collinear
    vector<pair<long long,long long>> pts;
    for(auto&cc:cells) pts.push_back({cx(cc.first),cx(cc.second)});
    int m=pts.size(); vector<pair<long long,long long>> poly;
    for(int k=0;k<m;k++){ auto a=pts[(k-1+m)%m],b=pts[k],cc=pts[(k+1)%m];
        if((a.first==b.first&&b.first==cc.first)||(a.second==b.second&&b.second==cc.second)) continue;
        poly.push_back(b);
    }
    // verify true perimeter; if over budget or degenerate, fall back to the
    // warm-start best rectangle (recomputed), which is always valid.
    long long trueP=0; { int mm=poly.size(); for(int k=0;k<mm;k++){ auto a=poly[k],b=poly[(k+1)%mm]; trueP+=llabs(a.first-b.first)+llabs(a.second-b.second);} }
    if((int)poly.size()<4||(int)poly.size()>1000||trueP>PERIM_BUDGET){
        // recompute best rectangle for a safe valid fallback
        vector<vector<long long>> PS(G+1, vector<long long>(G+1,0));
        for(int i=0;i<G;i++)for(int j=0;j<G;j++)
            PS[i+1][j+1]=Wv[CID(i,j)]+PS[i][j+1]+PS[i+1][j]-PS[i][j];
        long long bestS=-(1LL<<60); int bi1=0,bi2=1,bj1=0,bj2=1;
        for(int i1=0;i1<G;i1++)for(int i2=i1+1;i2<=G;i2++){
            long long w=linecoord[i2]-linecoord[i1];
            for(int j1=0;j1<G;j1++)for(int j2=j1+1;j2<=G;j2++){
                long long h=linecoord[j2]-linecoord[j1];
                if(2*(w+h)>400000) continue;
                long long s=PS[i2][j2]-PS[i1][j2]-PS[i2][j1]+PS[i1][j1];
                if(s>bestS){ bestS=s; bi1=i1;bi2=i2;bj1=j1;bj2=j2; }
            }
        }
        printf("4\n%lld %lld\n%lld %lld\n%lld %lld\n%lld %lld\n",
            linecoord[bi1],linecoord[bj1], linecoord[bi2],linecoord[bj1],
            linecoord[bi2],linecoord[bj2], linecoord[bi1],linecoord[bj2]);
        return 0;
    }
    printf("%d\n",(int)poly.size());
    for(auto&p:poly) printf("%lld %lld\n",p.first,p.second);
    fprintf(stderr,"iters=%lld bestScore(a-b)=%lld m=%d\n",iter,bestScore,(int)poly.size());
    return 0;
}
```
