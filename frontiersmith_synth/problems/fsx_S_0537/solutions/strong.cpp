// TIER: strong
// Network surgery. The insight: a toll's only job is to change the settled
// equilibrium, so evaluate every road by its COUNTERFACTUAL -- how much total
// travel time drops if a prohibitive toll effectively deletes it. Decompose the
// city into independent districts, and for each road binary-search the minimum
// toll that drives its flow to zero, measuring the resulting travel-time change.
// Roads whose deletion helps (the Braess shortcuts) get positive benefit; the
// genuinely useful shortcut-shaped roads get negative benefit and are left
// alone. Then spend the scarce budget knapsack-style on the best benefit/cost
// deletions. This concentrates the budget to actually remove paradox edges,
// instead of smearing it thinly over the visibly-busy roads.
#include <bits/stdc++.h>
using namespace std;

struct E { int u, v; long long a; int k; long long b; };
static const long long CAP = (long long)4e15;
static long long latency(const E& e, long long x){
    long long xp = 1;
    for (int i = 0; i < e.k; i++){ xp *= x; if (xp > CAP){ xp = CAP; break; } }
    long long r = e.a * xp + e.b; if (r > CAP) r = CAP; return r;
}
// local settle over a subgraph (already-remapped nodes/edges). returns F and flow.
static long long settle(int N, const vector<E>& edg, const vector<vector<int>>& adj,
                        const vector<int>& cs, const vector<int>& ct, int R,
                        const vector<long long>& toll, vector<int>* flowOut){
    int M = edg.size(), D = cs.size();
    vector<int> flow(M, 0); vector<vector<int>> path(D);
    auto ec = [&](int e){ long long c = latency(edg[e], (long long)flow[e]+1)+toll[e]; return c>CAP?CAP:c; };
    auto shortest = [&](int s, int t){
        vector<long long> dist(N, LLONG_MAX); vector<int> pe(N, -1);
        priority_queue<pair<long long,int>, vector<pair<long long,int>>, greater<pair<long long,int>>> pq;
        dist[s]=0; pq.push({0,s});
        while(!pq.empty()){ auto tp=pq.top(); pq.pop(); long long d=tp.first; int u=tp.second;
            if(d>dist[u]) continue;
            for(int e: adj[u]){ int v=edg[e].v; long long nd=d+ec(e);
                if(nd<dist[v]){ dist[v]=nd; pe[v]=e; pq.push({nd,v}); } } }
        vector<int> p; int cur=t; while(cur!=s){ int e=pe[cur]; if(e<0){p.clear();return p;} p.push_back(e); cur=edg[e].u; }
        reverse(p.begin(),p.end()); return p;
    };
    auto assign=[&](int i){ auto p=shortest(cs[i],ct[i]); path[i]=p; for(int e:p) flow[e]++; };
    auto rem=[&](int i){ for(int e:path[i]) flow[e]--; path[i].clear(); };
    for(int i=0;i<D;i++) assign(i);
    for(int r=0;r<R;r++) for(int i=0;i<D;i++){ rem(i); assign(i); }
    long long F=0; for(int e=0;e<M;e++) F+=(long long)flow[e]*latency(edg[e],(long long)flow[e]);
    if(flowOut)*flowOut=flow; return F;
}

struct DSU { vector<int> p; DSU(int n):p(n){ iota(p.begin(),p.end(),0);}
    int f(int x){ return p[x]==x?x:p[x]=f(p[x]); }
    void u(int a,int b){ p[f(a)]=f(b);} };

int main(){
    int N,M,D,R; long long T;
    if(!(cin>>N>>M>>D>>T>>R)) return 0;
    vector<E> edg(M);
    vector<vector<int>> gadj(N);
    for(int e=0;e<M;e++){ int u,v,k; long long a,b; cin>>u>>v>>a>>k>>b; u--; v--; edg[e]={u,v,a,k,b}; gadj[u].push_back(e); }
    vector<int> cs(D), ct(D);
    for(int i=0;i<D;i++){ cin>>cs[i]>>ct[i]; cs[i]--; ct[i]--; }

    // ---- connected components (undirected) ----
    DSU dsu(N);
    for(int e=0;e<M;e++) dsu.u(edg[e].u, edg[e].v);

    // group edges & commuters by component root
    map<int,vector<int>> compEdges, compComm;
    for(int e=0;e<M;e++) compEdges[dsu.f(edg[e].u)].push_back(e);
    for(int i=0;i<D;i++) compComm[dsu.f(cs[i])].push_back(i);

    struct Cand { long long benefit, cost; int gedge; }; // gedge = global edge idx
    vector<Cand> cands;

    for(auto& kv : compEdges){
        int root = kv.first;
        vector<int>& ges = kv.second;                 // global edge indices
        vector<int>& gis = compComm[root];            // global commuter indices
        if(gis.empty()) continue;

        // remap nodes appearing in this component
        map<int,int> nid;
        auto getn = [&](int g)->int{ auto it=nid.find(g); if(it!=nid.end()) return it->second; int id=nid.size(); nid[g]=id; return id; };
        vector<E> LE; vector<int> Lg2local(M,-1); // global edge -> local edge
        for(int ge : ges){ int lu=getn(edg[ge].u), lv=getn(edg[ge].v); Lg2local[ge]=(int)LE.size(); LE.push_back({lu,lv,edg[ge].a,edg[ge].k,edg[ge].b}); }
        int LN = nid.size();
        vector<vector<int>> Ladj(LN);
        for(int le=0; le<(int)LE.size(); le++) Ladj[LE[le].u].push_back(le);
        vector<int> Lcs, Lct;
        for(int gi : gis){ Lcs.push_back(getn(cs[gi])); Lct.push_back(getn(ct[gi])); }

        int LM = LE.size();
        int Dloc = Lcs.size();
        vector<long long> ztoll(LM, 0);
        vector<int> f0;
        long long F0 = settle(LN, LE, Ladj, Lcs, Lct, R, ztoll, &f0);

        // deletion-toll upper bound: enough to price any single road out
        long long H = 1;
        for(int le=0; le<LM; le++) H += latency(LE[le], (long long)Dloc) + LE[le].b + 1;
        if(H > T) H = T + 1; // no point searching above budget cap (except the boundary)
        if(H > (long long)4e15) H = (long long)4e15;

        auto flowOfWithToll = [&](int le, long long tau)->long long{
            vector<long long> tt(LM, 0); tt[le]=tau; vector<int> ff; settle(LN, LE, Ladj, Lcs, Lct, R, tt, &ff); return ff[le];
        };
        auto FWithToll = [&](int le, long long tau)->long long{
            vector<long long> tt(LM, 0); tt[le]=tau; return settle(LN, LE, Ladj, Lcs, Lct, R, tt, nullptr);
        };

        for(int le=0; le<LM; le++){
            if(f0[le] == 0) continue;                 // road carries no flow -> nothing to delete
            if(flowOfWithToll(le, H) != 0) continue;  // cannot be fully priced out within reach
            // binary search minimal tau that zeroes its flow
            long long lo=0, hi=H;
            while(lo < hi){ long long mid=lo+(hi-lo)/2; if(flowOfWithToll(le, mid)==0) hi=mid; else lo=mid+1; }
            long long tau = lo;
            if(tau <= 0 || tau > T) continue;
            long long Fd = FWithToll(le, tau);
            long long benefit = F0 - Fd;
            if(benefit > 0) cands.push_back({benefit, tau, ges[le]});
        }
    }

    // ---- knapsack-greedy by benefit/cost within the budget ----
    sort(cands.begin(), cands.end(), [](const Cand&a, const Cand&b){
        return (__int128)a.benefit * b.cost > (__int128)b.benefit * a.cost;
    });
    vector<long long> toll(M, 0);
    long long remain = T;
    for(auto& c : cands){ if(c.cost <= remain){ toll[c.gedge] = c.cost; remain -= c.cost; } }

    // ---- spend the leftover budget as marginal-cost pricing of the surviving
    //      congestible roads, nudging the settled flow toward the social optimum
    //      (a genuine extra edge over pure deletion). ----
    if(remain > 0){
        vector<int> fs;
        settle(N, edg, gadj, cs, ct, R, toll, &fs);
        vector<long long> wm(M, 0); long long swm = 0;
        for(int e=0;e<M;e++){
            if(edg[e].a > 0 && toll[e] == 0){
                long long mg = latency(edg[e], (long long)fs[e]) - latency(edg[e], (long long)max(0, fs[e]-1));
                wm[e] = mg * (long long)fs[e];
                swm += wm[e];
            }
        }
        if(swm > 0){
            long long spent = 0;
            for(int e=0;e<M;e++){
                long long g = (long long)((__int128)remain * wm[e] / swm);
                if(g > remain - spent) g = remain - spent;
                toll[e] += g; spent += g;
            }
        }
    }

    for(int e=0;e<M;e++){ if(e) putchar(' '); printf("%lld", toll[e]); }
    printf("\n");
    return 0;
}
