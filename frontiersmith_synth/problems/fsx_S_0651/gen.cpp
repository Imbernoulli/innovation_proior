#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Bus lines for weekdays, weekends, and stadium nights"  (generator)
// family: regret-bounded-transit-lines
//
// A street graph (N stops, M streets) + a shared line-length BUDGET / line-count L.
// THREE demand scenarios (0=weekday,1=weekend,2=stadium night) each ship a
// precomputed ORACLE cost (a strong scenario-DEDICATED construction using the
// SAME full L/BUDGET but only that scenario's own demand). The solver must build
// ONE shared network (L paths in the street graph, total length <= BUDGET) that is
// good in the worst case: minimize max_s (your_cost_s / oracle_s).
//
// PLANTED STRUCTURE: a "stadium spur" -- a long pendant chain attached to the
// weekday-busiest core hub, leading to a stadium node ST that is otherwise
// unreachable except through that chain. Weekday/weekend demand lives entirely in
// the core+suburbs; stadium-night demand is a handful of LOW-weight pairs from
// core/suburb nodes to ST. Because minimax REGRET (ratio to each scenario's own
// oracle) is what's scored -- not raw cost -- ST's small demand volume still makes
// stadium-night the most REGRET-CRITICAL scenario: its oracle is cheap (one direct
// line), so ANY shortfall blows its ratio up far more than an equal shortfall hurts
// the well-covered weekday/weekend scenarios.
// -----------------------------------------------------------------------------

static const ll WALK_MULT = 5;

int N;
vector<array<ll,3>> edgeList;              // u,v,w  (1-indexed)
vector<vector<pair<int,ll>>> adjv;         // adjacency

void addEdge(int u,int v,ll w){
    edgeList.push_back({(ll)u,(ll)v,w});
    adjv[u].push_back({v,w});
    adjv[v].push_back({u,w});
}

// shortest path (by real edge weight) from s to t; returns edge list (u,v,w) along the path
vector<array<ll,3>> shortestPathEdges(int s,int t){
    vector<ll> dist(N+1, LLONG_MAX/4);
    vector<int> par(N+1, -1);
    vector<char> done(N+1, 0);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<>> pq;
    dist[s]=0; pq.push({0,s});
    while(!pq.empty()){
        auto [d,u] = pq.top(); pq.pop();
        if(done[u]) continue; done[u]=1;
        if(u==t) break;
        for(auto&e: adjv[u]){
            int v=e.first; ll w=e.second;
            if(dist[u]+w < dist[v]){ dist[v]=dist[u]+w; par[v]=u; pq.push({dist[v],v}); }
        }
    }
    vector<array<ll,3>> path;
    if(dist[t] > LLONG_MAX/8) return path; // unreachable (shouldn't happen, graph connected)
    int cur=t;
    while(cur!=s){
        int p = par[cur];
        ll w=-1;
        for(auto&e: adjv[p]) if(e.first==cur){ w=e.second; break; }
        path.push_back({(ll)p,(ll)cur,w});
        cur=p;
    }
    reverse(path.begin(), path.end());
    return path;
}

// hybrid distance from src to all nodes: edges in `service` cost real weight, others cost WALK_MULT*weight
vector<ll> hybridDist(int src, const set<pair<int,int>>& service){
    vector<ll> dist(N+1, LLONG_MAX/4);
    vector<char> done(N+1,0);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<>> pq;
    dist[src]=0; pq.push({0,src});
    while(!pq.empty()){
        auto [d,u]=pq.top(); pq.pop();
        if(done[u]) continue; done[u]=1;
        for(auto&e: adjv[u]){
            int v=e.first; ll w=e.second;
            int a=min(u,v), b=max(u,v);
            ll eff = service.count({a,b}) ? w : WALK_MULT*w;
            if(dist[u]+eff < dist[v]){ dist[v]=dist[u]+eff; pq.push({dist[v],v}); }
        }
    }
    return dist;
}

// "stop at first unaffordable pair" greedy: process pairsSorted in order, add shortest
// path if it fits in the remaining budget/line count, else STOP.
set<pair<int,int>> greedyConstruct(const vector<array<ll,3>>& pairsSorted, int L, ll budget){
    set<pair<int,int>> es;
    int linesUsed=0; ll remaining=budget;
    for(auto&p: pairsSorted){
        if(linesUsed>=L) break;
        int u=(int)p[0], v=(int)p[1];
        auto path = shortestPathEdges(u,v);
        ll length=0; for(auto&e:path) length+=e[2];
        if(length<=remaining){
            remaining-=length; linesUsed++;
            for(auto&e:path) es.insert({(int)min(e[0],e[1]), (int)max(e[0],e[1])});
        } else break;
    }
    return es;
}

ll scenarioCost(const vector<array<ll,3>>& pairs, const set<pair<int,int>>& service){
    ll total=0;
    for(auto&p: pairs){
        int u=(int)p[0], v=(int)p[1]; ll w=p[2];
        auto dist = hybridDist(u, service);
        total += w*dist[v];
    }
    return total;
}

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    double f = (testId - 1) / 9.0;

    int C    = 6  + (int)llround(f*14.0);   // core stops
    int Rsub = 4  + (int)llround(f*16.0);   // suburb stops
    int spurLen = 3 + (testId % 4);          // 3..6 spur edges
    int spurNodes = spurLen - 1;
    N = C + Rsub + spurNodes + 1;
    int stadium = N;

    adjv.assign(N+1, {});

    vector<int> coreNodes(C);
    for(int i=0;i<C;i++) coreNodes[i]=i+1;
    for(int i=C-1;i>0;i--) swap(coreNodes[i], coreNodes[rnd.next(0,i)]);

    // core spanning tree
    vector<int> connected; connected.push_back(coreNodes[0]);
    set<pair<int,int>> haveEdge;
    auto markEdge=[&](int u,int v){ haveEdge.insert({min(u,v),max(u,v)}); };
    for(int i=1;i<C;i++){
        int nd = coreNodes[i];
        int u = connected[rnd.next(0,(int)connected.size()-1)];
        ll w = rnd.next(3,10);
        addEdge(u,nd,w); markEdge(u,nd);
        connected.push_back(nd);
    }
    // extra chords
    int extra = C/2;
    for(int i=0;i<extra;i++){
        int u = coreNodes[rnd.next(0,C-1)];
        int v = coreNodes[rnd.next(0,C-1)];
        if(u!=v && !haveEdge.count({min(u,v),max(u,v)})){
            addEdge(u,v, rnd.next(3,10));
            markEdge(u,v);
        }
    }
    // suburb pendants
    vector<int> suburbNodes(Rsub);
    for(int i=0;i<Rsub;i++) suburbNodes[i] = C+1+i;
    for(int s : suburbNodes){
        int u = coreNodes[rnd.next(0,C-1)];
        addEdge(u, s, rnd.next(5,15));
    }

    vector<int> coreSuburb = coreNodes;
    for(int s: suburbNodes) coreSuburb.push_back(s);
    int poolSize = (int)coreSuburb.size();

    auto randPairs=[&](int n, int loW, int hiW, int destFixed)->vector<array<ll,3>>{
        vector<array<ll,3>> pairs;
        for(int i=0;i<n;i++){
            int u = coreSuburb[rnd.next(0,poolSize-1)];
            int v;
            if(destFixed>0) v=destFixed;
            else { do { v = coreSuburb[rnd.next(0,poolSize-1)]; } while(v==u); }
            ll w = rnd.next(loW,hiW);
            pairs.push_back({(ll)u,(ll)v,w});
        }
        return pairs;
    };

    int nWeekday = 10 + (int)llround(f*20.0);
    int nWeekend = 6  + (int)llround(f*10.0);
    int nEvent   = 3  + (int)llround(f*4.0);

    vector<array<ll,3>> weekday = randPairs(nWeekday, 25, 60, -1);
    vector<array<ll,3>> weekend = randPairs(nWeekend, 20, 50, -1);

    // hub = core node with the largest total weekday weight incident to it
    map<int,ll> wdeg;
    for(int c: coreNodes) wdeg[c]=0;
    for(auto&p: weekday){
        int u=(int)p[0], v=(int)p[1]; ll w=p[2];
        if(wdeg.count(u)) wdeg[u]+=w;
        if(wdeg.count(v)) wdeg[v]+=w;
    }
    int hub = coreNodes[0];
    for(int c: coreNodes) if(wdeg[c] > wdeg[hub]) hub=c;

    // spur chain hub -> ... -> stadium
    int prev = hub;
    ll Sspur = 0;
    for(int i=0;i<spurNodes;i++){
        int sn = C + Rsub + 1 + i;
        ll w = rnd.next(4,8);
        addEdge(prev, sn, w); Sspur += w; prev = sn;
    }
    { ll w = rnd.next(4,8); addEdge(prev, stadium, w); Sspur += w; }

    vector<array<ll,3>> event = randPairs(nEvent, 8, 20, stadium);

    int L = 5 + (testId % 3);

    // BUDGET: 1.1 * MST(core+suburb) + 0.5 * spur length
    {
        set<int> nodesSet(coreNodes.begin(), coreNodes.end());
        for(int s: suburbNodes) nodesSet.insert(s);
        // Prim's MST restricted to nodesSet using only edges among nodesSet (core tree/chords/suburb pendants)
        int start = *nodesSet.begin();
        set<int> visited={start};
        priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<>> pq;
        for(auto&e: adjv[start]) if(nodesSet.count(e.first)) pq.push({e.second, e.first});
        ll mstCost=0;
        while((int)visited.size() < (int)nodesSet.size() && !pq.empty()){
            auto [w,u] = pq.top(); pq.pop();
            if(visited.count(u)) continue;
            visited.insert(u); mstCost += w;
            for(auto&e: adjv[u]) if(nodesSet.count(e.first) && !visited.count(e.first)) pq.push({e.second, e.first});
        }
        ll BUDGET = (ll)llround(mstCost*1.1) + (ll)llround(Sspur*0.5);
        if(BUDGET < 10) BUDGET = 10;

        // ---- oracle per scenario: dedicate FULL L & FULL BUDGET to that scenario alone ----
        vector<array<ll,3>>* scen[3] = {&weekday, &weekend, &event};
        ll oracle[3];
        for(int s=0;s<3;s++){
            vector<array<ll,3>> sorted = *scen[s];
            std::sort(sorted.begin(), sorted.end(), [](const array<ll,3>&a, const array<ll,3>&b){ return a[2] > b[2]; });
            auto es = greedyConstruct(sorted, L, BUDGET);
            oracle[s] = scenarioCost(*scen[s], es);
            if(oracle[s] < 1) oracle[s] = 1;
        }

        // ---- print ----
        printf("%d %d\n", N, (int)edgeList.size());
        for(auto&e: edgeList) printf("%lld %lld %lld\n", e[0], e[1], e[2]);
        printf("%d %lld\n", L, BUDGET);
        for(int s=0;s<3;s++){
            printf("%d %lld\n", (int)scen[s]->size(), oracle[s]);
            for(auto&p: *scen[s]) printf("%lld %lld %lld\n", p[0], p[1], p[2]);
        }
    }
    return 0;
}
