#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Bus lines for weekdays, weekends, and stadium nights".
//
// Input:  N M ; M lines (u v w) ; L BUDGET ; then for s=0,1,2 (weekday/weekend/
//         stadium-night): "D_s ORACLE_s" then D_s lines (u v w) demand pairs.
// Output: L lines. Line i: "k s_1 s_2 ... s_k" (k>=1 stops, 1-indexed). Consecutive
//         stops must be joined by a real street edge. Total length (sum of the
//         traversed edge weights over ALL lines) must be <= BUDGET.
//
// Objective (MIN): let the SERVICE set = union of all edges appearing in any line.
// For every demand pair (u,v,w) of scenario s, its travel time is the shortest
// path in the FULL graph where a service edge costs its real weight and a
// non-service edge costs WALK_MULT*weight (walking/no-service penalty). cost_s =
// sum w*time. regret_s = cost_s / oracle_s (oracle_s given in input). F = max_s
// regret_s (worst-case regret against each scenario's own dedicated reference).
//
// Baseline B (checker-built "trivial" reference): the SAME pooled-by-weight
// "stop at the first unaffordable pair" construction the naive/greedy strategy
// uses, but starved to a single line and 3% of BUDGET -- a genuinely minimal-
// effort network. Because both B and any real submission use variants of the
// same monotone construction rule, B <= any construction with more lines/budget
// on the same sorted order (a strictly weaker resource envelope can only serve a
// prefix of what a richer envelope serves), so real submissions reliably beat it.
// Score (min): sc = min(1000, 100*B/max(eps,F)); ratio = sc/1000.
// -----------------------------------------------------------------------------

static const ll WALK_MULT = 5;
static const double EPS = 1e-9;

int N;
vector<vector<pair<int,ll>>> adjv;

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
    if(dist[t] > LLONG_MAX/8) return path;
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
    registerTestlibCmd(argc, argv);

    int M;
    N = inf.readInt();
    M = inf.readInt();
    adjv.assign(N+1, {});
    map<pair<int,int>, ll> edgeW;
    for(int i=0;i<M;i++){
        int u = inf.readInt(1,N);
        int v = inf.readInt(1,N);
        ll w = inf.readLong(1, (ll)1e9);
        adjv[u].push_back({v,w});
        adjv[v].push_back({u,w});
        edgeW[{min(u,v),max(u,v)}] = w;
    }
    int L = inf.readInt();
    ll BUDGET = inf.readLong();

    vector<vector<array<ll,3>>> scen(3);
    vector<ll> oracle(3);
    for(int s=0;s<3;s++){
        int D = inf.readInt();
        oracle[s] = inf.readLong();
        if(oracle[s] < 1) oracle[s] = 1;
        for(int i=0;i<D;i++){
            int u=inf.readInt(1,N), v=inf.readInt(1,N);
            ll w=inf.readLong(1,(ll)1e9);
            scen[s].push_back({(ll)u,(ll)v,w});
        }
    }

    // ---- replay participant's L lines ----
    set<pair<int,int>> service;
    ll totalLength = 0;
    const int MAXK = 2*N + 60;
    for(int i=0;i<L;i++){
        int k = ouf.readInt(1, MAXK, "line_len");
        int prevNode = ouf.readInt(1, N, "stop");
        for(int j=1;j<k;j++){
            int cur = ouf.readInt(1, N, "stop");
            auto it = edgeW.find({min(prevNode,cur), max(prevNode,cur)});
            if(it == edgeW.end())
                quitf(_wa, "line %d: no street edge between stop %d and stop %d", i+1, prevNode, cur);
            totalLength += it->second;
            if(totalLength > BUDGET)
                quitf(_wa, "total line length exceeds BUDGET: %lld > %lld", totalLength, BUDGET);
            service.insert({min(prevNode,cur), max(prevNode,cur)});
            prevNode = cur;
        }
    }
    if(!ouf.seekEof()) quitf(_wa, "trailing output tokens after %d lines", L);

    // ---- objective F = max regret across the 3 scenarios ----
    double F = -1e18;
    for(int s=0;s<3;s++){
        ll cost = scenarioCost(scen[s], service);
        double regret = (double)cost / (double)oracle[s];
        F = max(F, regret);
    }

    // ---- internal baseline B: minimal-effort pooled-by-weight construction ----
    vector<array<ll,3>> pooled;
    for(int s=0;s<3;s++) for(auto&p: scen[s]) pooled.push_back(p);
    std::sort(pooled.begin(), pooled.end(), [](const array<ll,3>&a, const array<ll,3>&b){ return a[2] > b[2]; });
    ll budgetTriv = max((ll)1, (ll)llround(BUDGET*0.03));
    auto esTriv = greedyConstruct(pooled, 1, budgetTriv);
    double B = -1e18;
    for(int s=0;s<3;s++){
        ll cost = scenarioCost(scen[s], esTriv);
        double regret = (double)cost / (double)oracle[s];
        B = max(B, regret);
    }
    if(B <= 0) B = 1.0;

    double sc = min(1000.0, 100.0 * B / max(EPS, F));
    quitp(sc/1000.0, "OK F=%.6f B=%.6f Ratio: %.6f", F, B, sc/1000.0);
    return 0;
}
