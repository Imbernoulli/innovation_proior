The greedy region averaged $2872$, beating the rectangle by $168$, and it taught me two things at once: that a rectilinear net built from grid cells is the right object, and that growing it by a one-shot forward pass is the wrong engine. The greedy cannot take back a cell it regrets, it spends its perimeter budget on whatever ragged boundary it stumbles into, and it lives or dies by a single grid resolution. Every one of those failures is a failure of *irreversibility*. So I keep the representation — a connected, hole-free set of grid cells whose outer boundary traces a simple rectilinear polygon — and replace the greedy with local search that can both add *and* remove cells, and that is willing to step downhill to escape the traps the greedy fell into. This is exactly the engine the benchmark's strong baseline uses: simulated annealing on the net, with cheap incremental scoring, which took ALE-Agent to fifth place at performance $2880$.

I propose **warm-started simulated annealing on a grid-cell region with $O(1)$ incremental scoring and local topology checks**. The move set is the natural one for a cell region: at each step I either *add* an outside cell adjacent to the region or *remove* a boundary cell of the region — a single-cell flip that adds or shaves one staircase step. Each flip changes the catch by exactly that cell's $(\#\text{mackerel}-\#\text{sardine})$, so if I precompute once the mackerel count `Av[c]` and sardine count `Bv[c]` in every cell, the change in $a-b$ for any candidate flip is an $O(1)$ lookup. That is the whole game for speed — I never recount fish during the search, only read stored counts and add or subtract them — and it is the incremental scoring the benchmark relies on, done here against a grid index instead of a kd-tree, with the same payoff of millions of candidate moves per second.

Three things must stay legal under every flip, and each needs a cheap incremental check. Perimeter: I keep a running boundary-edge count `bedges`; adding or removing a cell changes it by $\pm(4-2\cdot\#\text{inside neighbors})$, computable from the four neighbors, so I reject any flip that would push the traced perimeter over $4\times10^5$ without retracing the polygon. Connectivity and hole-freeness: a flip must not disconnect the region or open or close a hole, or the boundary stops being a single simple polygon. Recomputing global connectivity per flip would destroy the speed, so I use the classical digital-topology *simple-point* test — a cell flip preserves topology iff the foreground transitions exactly once as you walk the eight-cell ring around the cell, i.e. the crossing/transition count over the 8-neighbourhood equals $1$, a constant-time check on a $3\times3$ window. The simple-point test alone misses one trap: two cells touching only at a corner make the boundary a figure-eight, not a simple polygon, so I also forbid any flip that would create a diagonal pinch in a $2\times2$ block (the four checkerboard guards in `simple_point`). With both checks the traced boundary is always a single simple rectilinear cycle.

The search itself is Metropolis annealing: propose a random legal flip; if it improves $a-b$, take it; if it worsens $a-b$ by $\Delta<0$, take it anyway with probability $\exp(\Delta/T)$ for a temperature $T$ cooled geometrically from $T_0=8.0$ to $T_1=0.05$ as wall-clock fraction advances. Early, high $T$ lets the region wander — accepting cells that locally cost catch — so it reshapes out of the basins the greedy got stuck in; late, low $T$ keeps only improving flips and the region settles. I keep the best region ever seen and emit that.

Two design choices are load-bearing. First, *where to start*. The instinct is to grow from a seed cell the way the greedy did, but that fights the binding constraint: perimeter is the hard limit, and a rectangle is the most perimeter-efficient shape — most area per unit boundary. Growing from a dot spends almost the entire move budget just inflating a blob, paying boundary cost the whole way, and barely reaches the interesting part, carving notches. So I *warm-start* the region to the best perimeter-constrained rectangle, computed exactly by the rung-1 prefix-sum sweep, and let SA spend its whole budget on what it is uniquely good at: cutting staircase notches to release sardine pockets and extending the boundary to grab outlying mackerel. Starting in a strong basin instead of the empty grid is the difference between SA refining a good net and reconstructing a mediocre one. Second, the *grid resolution* $G=50$ (cell side $2000$, dividing the grid exactly). Finer cells hug the fish more closely but, under a fixed perimeter budget, each unit-edge is shorter so the region cannot grow large enough to hold the mackerel; coarser cells cover more area per edge but blunt the notches. $G=50$ is the balance: fine enough to carve useful notches, coarse enough that the budget still wraps a large region. A final true-perimeter check on the traced polygon falls back to the best rectangle if the trace is ever degenerate or over budget, so the rung never emits an illegal net and never regresses below rung 1.

This should clearly beat the one-shot greedy: on overlapping seeds it can do what the greedy cannot — start from the best box, then *remove* sardine-heavy boundary cells and *add* mackerel cells just outside, trading the net's shape against catch in both directions; on easy seeds where the box is near-optimal it holds the box. And I should be honest about the ceiling. The single-cell flip is a *local* move while the perimeter budget is *global* and binding: once the boundary is near its length limit, every useful notch must be paid for by shaving length somewhere else, and a blind random flip rarely proposes that coordinated trade. The search plateaus not because the idea is wrong but because the proposals are *undirected* — most random flips touch boundary that is already correct, and the few that would fix a genuinely misclassified fish are drowned out. That undirectedness, and the cost of revalidating each candidate, are exactly the two levers the next rung pulls.

```cpp
// Rung 3: simulated annealing on a grid-cell region (rectilinear polygon).
//
// Representation: a GxG binary grid; the net is the union of selected cells,
// whose outer boundary is a simple rectilinear polygon. Each cell precomputes
// w[c] = (#mackerel - #sardine), a[c]=#mackerel, b[c]=#sardine inside it, so the
// objective a-b changes by O(1) per cell flip (INCREMENTAL SCORING).
//
// SA moves: pick a boundary cell and toggle it (add an outside cell adjacent to
// the region, or remove a region cell on the boundary). A move is admissible iff
// it keeps the region (a) non-empty, (b) 4-connected, (c) hole-free, and (d)
// within the perimeter budget. Connectivity/hole checks are done locally on the
// 3x3 neighbourhood for speed (a cell flip preserves connectivity/simple-ness iff
// its local "transition count" around the 8-neighbourhood is exactly 1 -- the
// standard digital-topology simple-point test), with a cheap global guard.
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
    // frontier maintained implicitly: pick random cell that is a boundary cell
    // (in-region cell adjacent to outside, or outside cell adjacent to region)
    // We sample candidate boundary cells by random walk over a maintained list.
    // For simplicity & correctness, sample a random region cell or random
    // outside-adjacent cell each step.
    vector<int> regionCells;
    for(int c=0;c<G*G;c++) if(inset[c]) regionCells.push_back(c);

    while(true){
        if((iter & 1023)==0){
            double el=chrono::duration<double>(chrono::steady_clock::now()-t_start).count();
            if(el>TIME) break;
        }
        iter++;
        double frac = chrono::duration<double>(chrono::steady_clock::now()-t_start).count()/TIME;
        if(frac>1) frac=1;
        double T = T0*pow(T1/T0, frac);

        bool addMove = (xr()&1);
        int i,j,c;
        if(addMove){
            // pick a random region cell, then a random outside neighbour
            if(regionCells.empty()) continue;
            int rc=regionCells[xr()%regionCells.size()];
            int ri=rc/G, rj=rc%G;
            int d=xr()%4; int di[4]={1,-1,0,0}, dj[4]={0,0,1,-1};
            i=ri+di[d]; j=rj+dj[d];
            if(i<0||i>=G||j<0||j>=G) continue;
            c=CID(i,j);
            if(inset[c]) continue;
            int de=edge_delta(i,j,true);
            if((bedges+de)*(double)CELLW > SAFE) continue;
            // tentatively add, check simple-point (preserves topology)
            inset[c]=1;
            bool ok=simple_point(i,j);
            if(!ok){ inset[c]=0; continue; }
            long long dScore=Av[c]-Bv[c];
            if(dScore>=0 || urand()<exp(dScore/T)){
                // accept
                bedges+=de; curScore+=dScore; regionCells.push_back(c);
                if(curScore>bestScore){ bestScore=curScore; best=inset; bestBedges=bedges; }
            }else{
                inset[c]=0;
            }
        }else{
            // remove a random region cell on the boundary
            if(regionCells.size()<=1) continue;
            int idx=xr()%regionCells.size();
            c=regionCells[idx]; i=c/G; j=c%G;
            // must be a boundary cell (has an outside neighbour or grid border)
            bool bnd=false; { int di[4]={1,-1,0,0}, dj[4]={0,0,1,-1};
                for(int d=0;d<4;d++){int ni=i+di[d],nj=j+dj[d]; if(ni<0||ni>=G||nj<0||nj>=G||!inset[CID(ni,nj)]){bnd=true;break;}}}
            if(!bnd) continue;
            int de=edge_delta(i,j,false);
            // removing reduces perimeter normally; always within budget
            inset[c]=0;
            bool ok=simple_point(i,j); // simple-point test symmetric for removal
            if(!ok){ inset[c]=1; continue; }
            long long dScore=-(Av[c]-Bv[c]);
            if(dScore>=0 || urand()<exp(dScore/T)){
                bedges+=de; curScore+=dScore;
                // swap-remove from regionCells
                regionCells[idx]=regionCells.back(); regionCells.pop_back();
                if(curScore>bestScore){ bestScore=curScore; best=inset; bestBedges=bedges; }
            }else{
                inset[c]=1;
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
