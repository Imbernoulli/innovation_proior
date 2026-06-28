// Grid Wire Routing (ale-v2-07) -- maximize the number of vertex-disjoint nets routed.
//
// OBJECTIVE: route as many nets as possible; each net is a simple 4-connected path
// of free ('.') cells between its two terminals; routed nets must be pairwise
// VERTEX-DISJOINT (no shared cell => no overlap and no crossing). Score = #routed.
//
// HEURISTIC (the innovation): negotiated-congestion RIP-UP-AND-REROUTE (PathFinder).
// A trivial baseline routes nets one at a time with A* through already-claimed cells
// hard-blocked, skipping a net when it is boxed in -- the order it picks is arbitrary
// and it strands nets that a small reroute of an earlier net would have freed. Instead
// we let ALL nets route through a SHARED grid with soft "congestion" costs, then
// iteratively rip up nets that share cells and reroute them via A* on a cost field that
// PENALISES cells in proportion to how contested they are (present occupancy) and how
// chronically contested they have been (accumulated history). Over rounds the nets
// negotiate away from each other; cells that everyone wants get expensive, so most nets
// find private detours. We then COMMIT a hard vertex-disjoint subset from that layout
// and, with the freed space, greedily route any remaining nets. The committed solution
// is ALWAYS disjoint by construction, so the output is always feasible.
//
// Single file, C++17, reads stdin, writes stdout. Time budget ~1.8s wall.
#include <bits/stdc++.h>
using namespace std;

static uint64_t rng_state = 0x9e3779b97f4a7c15ULL;
static inline uint64_t xr(){ rng_state^=rng_state<<13; rng_state^=rng_state>>7; rng_state^=rng_state<<17; return rng_state; }
static inline double urand(){ return (xr()>>11)*(1.0/9007199254740992.0); }

static chrono::steady_clock::time_point T0;
static inline double elapsed(){ return chrono::duration<double>(chrono::steady_clock::now()-T0).count(); }
static const double TIME_LIMIT = 1.80;

int H, W, K;
int N;                       // H*W
vector<char> blk;            // blocked cell? (obstacle)
vector<int> sr, sc, tr, tc;  // net terminals
inline int ID(int r,int c){ return r*W+c; }

const int DR[4]={-1,1,0,0};
const int DC[4]={0,0,-1,1};

// ---------- A* on a cost field -----------------------------------------------
// Find a min-cost path from s to t over free cells, where entering cell v costs
// cellCost[v] (>=1). 'hardBlocked' cells are impassable (used by the committer).
// Returns the path (cell ids) including both endpoints, or empty if unreachable.
struct AStar {
    vector<double> dist;
    vector<int> prev;
    vector<int> visToken;     // which "generation" last touched a cell
    int token = 0;
    void init(){ dist.assign(N, 0); prev.assign(N,-1); visToken.assign(N,-1); }
    // generic A*; predicate passable(v) decides if cell v may be entered.
    template<class Pass>
    vector<int> run(int s,int t, const vector<double>& cellCost, Pass passable){
        token++;
        // priority queue of (f = g + heuristic, cell)
        typedef pair<double,int> PDI;
        priority_queue<PDI, vector<PDI>, greater<PDI>> pq;
        int tr_=t/W, tc_=t%W;
        auto h=[&](int v)->double{ int r=v/W,c=v%W; return abs(r-tr_)+abs(c-tc_); };
        dist[s]=0; visToken[s]=token; prev[s]=-1;
        pq.push({h(s), s});
        while(!pq.empty()){
            auto [f,u]=pq.top(); pq.pop();
            if(u==t) break;
            double gu=dist[u];
            if(f - h(u) > gu + 1e-9) continue;   // stale entry
            int r=u/W,c=u%W;
            for(int d=0;d<4;d++){
                int nr=r+DR[d], nc=c+DC[d];
                if(nr<0||nr>=H||nc<0||nc>=W) continue;
                int v=nr*W+nc;
                if(!passable(v)) continue;
                double ng=gu+cellCost[v];
                if(visToken[v]!=token || ng < dist[v]-1e-9){
                    dist[v]=ng; visToken[v]=token; prev[v]=u;
                    pq.push({ng+h(v), v});
                }
            }
        }
        if(visToken[t]!=token) return {};
        vector<int> path;
        for(int v=t; v!=-1; v=prev[v]) path.push_back(v);
        reverse(path.begin(), path.end());
        return path;
    }
} astar;

// ---------- committer: hard vertex-disjoint greedy ----------------------------
// Given an ORDER of nets and (optionally) a congestion field to bias detours,
// route each net via A* through cells not yet claimed by a committed net. This
// ALWAYS yields a vertex-disjoint set. Returns (#routed, paths-by-net).
int commit(const vector<int>& order, const vector<double>& bias,
           vector<vector<int>>& outPath){
    vector<char> claimed(N,0);
    outPath.assign(K, {});
    int routed=0;
    // cost = 1 + small bias so ties prefer less-contested cells; >=1 keeps A* admissible-ish.
    vector<double> cost(N);
    for(int v=0; v<N; v++) cost[v] = 1.0 + bias[v];
    for(int idx : order){
        int s=ID(sr[idx],sc[idx]), t=ID(tr[idx],tc[idx]);
        if(claimed[s] || claimed[t]) continue;   // a terminal got eaten by another net
        auto pass=[&](int v){ return !blk[v] && !claimed[v]; };
        // endpoints must be passable too (they are free by construction, just not claimed)
        if(blk[s]||blk[t]) continue;
        auto path = astar.run(s,t,cost,pass);
        if(path.empty()) continue;
        for(int v: path) claimed[v]=1;
        outPath[idx]=path;
        routed++;
    }
    return routed;
}

int main(){
    T0 = chrono::steady_clock::now();
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if(!(cin>>H>>W>>K)){ cout<<0<<"\n"; return 0; }
    N=H*W;
    blk.assign(N,0);
    {
        string row;
        for(int r=0;r<H;r++){
            cin>>row;
            for(int c=0;c<W;c++) blk[ID(r,c)] = (row[c]=='#');
        }
    }
    sr.resize(K); sc.resize(K); tr.resize(K); tc.resize(K);
    for(int i=0;i<K;i++){ cin>>sr[i]>>sc[i]>>tr[i]>>tc[i]; }
    astar.init();

    // ---- BASELINE & FIRST COMMIT: sequential A* in a sensible order ----------
    // Order nets by Manhattan distance ascending: short nets are easiest to place
    // and least likely to block others; this alone beats arbitrary order.
    vector<int> baseOrder(K);
    iota(baseOrder.begin(), baseOrder.end(), 0);
    sort(baseOrder.begin(), baseOrder.end(), [&](int a,int b){
        int da=abs(sr[a]-tr[a])+abs(sc[a]-tc[a]);
        int db=abs(sr[b]-tr[b])+abs(sc[b]-tc[b]);
        return da<db;
    });
    vector<double> zeroBias(N, 0.0);
    vector<vector<int>> bestPath;
    int bestRouted = commit(baseOrder, zeroBias, bestPath);

    // ---- NEGOTIATED-CONGESTION RIP-UP-AND-REROUTE (PathFinder) ---------------
    // Soft layout: every net keeps a path on a SHARED grid; cells may be shared.
    // occ[v]   = how many net-paths currently use cell v   (present congestion)
    // hist[v]  = accumulated history penalty for chronically contested cells
    // Entering v costs:  base(1) + presentFactor*occ[v] + hist[v].
    // Each round: for nets touching any over-used cell, rip up and reroute on the
    // updated cost field. Then bump hist on still-shared cells. Nets negotiate apart.
    vector<int> occ(N,0);
    vector<double> hist(N,0.0);
    vector<vector<int>> soft(K);

    auto cellCostField = [&](vector<double>& cost, double presentFactor){
        for(int v=0; v<N; v++)
            cost[v] = 1.0 + presentFactor*occ[v] + hist[v];
    };
    auto addPath = [&](int i, const vector<int>& p){
        soft[i]=p; for(int v:p) occ[v]++;
    };
    auto removePath = [&](int i){
        for(int v: soft[i]) occ[v]--;
        soft[i].clear();
    };

    vector<double> cost(N,1.0);
    // initial soft routing: shortest path ignoring congestion (occ all 0)
    {
        cellCostField(cost, 0.0);
        for(int i=0;i<K;i++){
            int s=ID(sr[i],sc[i]), t=ID(tr[i],tc[i]);
            auto pass=[&](int v){ return !blk[v]; };
            auto p = astar.run(s,t,cost,pass);
            if(!p.empty()) addPath(i,p);
            if(elapsed()>TIME_LIMIT*0.45) break;
        }
    }

    double presentFactor = 0.5;       // grows each round (negotiation pressure)
    int round=0;
    while(elapsed() < TIME_LIMIT*0.78){
        round++;
        cellCostField(cost, presentFactor);
        // Reroute every net (full PathFinder sweep) on the current field, in a
        // shuffled order so no net is permanently advantaged.
        vector<int> ord(K); iota(ord.begin(),ord.end(),0);
        for(int i=K-1;i>0;i--){ int j=xr()%(i+1); swap(ord[i],ord[j]); }
        for(int i : ord){
            int s=ID(sr[i],sc[i]), t=ID(tr[i],tc[i]);
            // rip up this net so it does not pay congestion against itself
            if(!soft[i].empty()) removePath(i);
            cellCostField(cost, presentFactor); // refresh after rip-up (cheap: O(N))
            auto pass=[&](int v){ return !blk[v]; };
            auto p = astar.run(s,t,cost,pass);
            if(!p.empty()) addPath(i,p);
            if(elapsed()>TIME_LIMIT*0.78) break;
        }
        // bump history on cells that are still shared (occ>1): negotiated congestion
        for(int v=0; v<N; v++) if(occ[v]>1) hist[v] += 0.5*(occ[v]-1);
        presentFactor += 0.7;

        // ---- derive a hard-disjoint commit from this soft layout -------------
        // Order nets by how "clean" their soft path is (fewest contested cells
        // first): those routes are most likely to survive as private corridors.
        // Use the soft occ field as the detour bias for the committer too.
        vector<int> order(K); iota(order.begin(),order.end(),0);
        vector<double> contest(K,1e18);
        for(int i=0;i<K;i++){
            if(soft[i].empty()) continue;
            double sh=0; for(int v: soft[i]) sh += (occ[v]>1)? (occ[v]-1):0;
            contest[i]=sh;
        }
        sort(order.begin(), order.end(), [&](int a,int b){
            if(contest[a]!=contest[b]) return contest[a]<contest[b];
            int da=(int)soft[a].size(), db=(int)soft[b].size();
            return da<db;
        });
        // bias the committer's A* toward currently-popular corridors lightly,
        // away from heavily contested ones: bias = small * hist (proven hot spots)
        vector<double> bias(N,0.0);
        for(int v=0; v<N; v++) bias[v] = 0.15*hist[v];
        vector<vector<int>> cand;
        int r = commit(order, bias, cand);
        if(r > bestRouted){ bestRouted=r; bestPath=cand; }
    }

    // ---- LNS POLISH on the committed solution --------------------------------
    // Try to squeeze in more nets: repeatedly RIP UP a random committed net (and
    // its neighbours) and REROUTE the unrouted ones plus the ripped ones through
    // the freed space, keeping any change that does not lose nets. This is the
    // large-neighbourhood-search refinement on top of the disjoint commit.
    {
        // Reconstruct claimed grid + routed flags from bestPath.
        auto buildClaimed=[&](const vector<vector<int>>& paths, vector<char>& claimed){
            claimed.assign(N,0);
            for(int i=0;i<K;i++) for(int v: paths[i]) claimed[v]=1;
        };
        vector<double> cost1(N,1.0);
        while(elapsed() < TIME_LIMIT){
            vector<vector<int>> paths = bestPath;
            // pick a few random ROUTED nets to rip up
            vector<int> routedNets;
            for(int i=0;i<K;i++) if(!paths[i].empty()) routedNets.push_back(i);
            if(routedNets.empty()) break;
            int ripCount = 1 + (int)(xr()% (1 + min((size_t)3, routedNets.size())));
            vector<int> ripped;
            for(int k=0;k<ripCount;k++){
                int i = routedNets[xr()%routedNets.size()];
                paths[i].clear();
                ripped.push_back(i);
            }
            vector<char> claimed; buildClaimed(paths, claimed);
            // candidate set to (re)route: the ripped nets + all currently unrouted nets
            vector<int> todo;
            for(int i=0;i<K;i++) if(paths[i].empty()) todo.push_back(i);
            // random order
            for(int i=(int)todo.size()-1;i>0;i--){ int j=xr()%(i+1); swap(todo[i],todo[j]); }
            for(int idx: todo){
                int s=ID(sr[idx],sc[idx]), t=ID(tr[idx],tc[idx]);
                if(claimed[s]||claimed[t]) continue;
                auto pass=[&](int v){ return !blk[v] && !claimed[v]; };
                auto p = astar.run(s,t,cost1,pass);
                if(p.empty()) continue;
                for(int v:p) claimed[v]=1;
                paths[idx]=p;
            }
            int cnt=0; for(int i=0;i<K;i++) if(!paths[i].empty()) cnt++;
            if(cnt >= bestRouted){ bestRouted=cnt; bestPath=paths; }
            if(elapsed()>TIME_LIMIT) break;
        }
    }

    // ---- SAFETY RE-COMMIT: guarantee the emitted set is vertex-disjoint -------
    // bestPath is built only by disjoint committers, but re-verify and drop any
    // accidental overlap so the output is NEVER infeasible.
    {
        vector<char> claimed(N,0);
        for(int i=0;i<K;i++){
            if(bestPath[i].empty()) continue;
            bool ok=true;
            // validate path: endpoints, adjacency, free, unclaimed, simple
            auto& p=bestPath[i];
            int s=ID(sr[i],sc[i]), t=ID(tr[i],tc[i]);
            if(!((p.front()==s&&p.back()==t)||(p.front()==t&&p.back()==s))) ok=false;
            vector<char> local;
            if(ok){
                for(size_t k=0;k<p.size() && ok;k++){
                    int v=p[k];
                    if(blk[v]||claimed[v]) { ok=false; break; }
                    if(k){ int a=p[k-1]; int ar=a/W,ac=a%W,br=v/W,bc=v%W;
                           if(abs(ar-br)+abs(ac-bc)!=1){ ok=false; break; } }
                }
            }
            if(ok){
                // ensure simple (no repeat) via a local set
                unordered_set<int> seen; seen.reserve(p.size()*2);
                for(int v:p){ if(!seen.insert(v).second){ ok=false; break; } }
            }
            if(ok){ for(int v:p) claimed[v]=1; }
            else bestPath[i].clear();
        }
    }

    // ---- OUTPUT --------------------------------------------------------------
    vector<int> routedIdx;
    for(int i=0;i<K;i++) if(!bestPath[i].empty()) routedIdx.push_back(i);
    // build output buffer
    {
        string out;
        out.reserve(routedIdx.size()*16);
        out += to_string(routedIdx.size()); out += '\n';
        for(int i: routedIdx){
            auto& p=bestPath[i];
            out += to_string(i); out += ' ';
            out += to_string(p.size());
            for(int v: p){ out += ' '; out += to_string(v/W); out += ' '; out += to_string(v%W); }
            out += '\n';
        }
        cout<<out;
    }
    return 0;
}
