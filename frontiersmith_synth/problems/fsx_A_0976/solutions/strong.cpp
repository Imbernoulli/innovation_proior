// TIER: strong
// Insight: the objective is max-over-scenarios of ceil(P / maxflow), and by
// max-flow/min-cut, maxflow to the chosen shelters IS the surviving cut capacity
// of that scenario. Distance is irrelevant; what matters is choosing a set of
// sites whose surviving cut stays healthy in EVERY published scenario at once.
// We build the set by marginal-gain (each addition picked to most improve the
// worst-scenario max-flow, evaluated with real Dinic per scenario -- this is the
// min-cut reasoning, not a distance heuristic), then locally improve by swaps.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

struct Dinic {
    struct E { int to; ll cap; };
    vector<E> es;
    vector<vector<int>> g;
    vector<int> level, it;
    int n;
    Dinic(int n_) : n(n_), g(n_) {}
    void addEdge(int u, int v, ll cap) {
        g[u].push_back((int)es.size()); es.push_back({v, cap});
        g[v].push_back((int)es.size()); es.push_back({u, 0});
    }
    bool bfs(int s, int t) {
        level.assign(n, -1);
        queue<int> q; level[s] = 0; q.push(s);
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int id : g[u]) if (es[id].cap > 0 && level[es[id].to] < 0) { level[es[id].to] = level[u] + 1; q.push(es[id].to); }
        }
        return level[t] >= 0;
    }
    ll dfs(int u, int t, ll f) {
        if (u == t) return f;
        for (int &i = it[u]; i < (int)g[u].size(); i++) {
            int id = g[u][i]; int v = es[id].to;
            if (es[id].cap > 0 && level[v] == level[u] + 1) {
                ll d = dfs(v, t, min(f, es[id].cap));
                if (d > 0) { es[id].cap -= d; es[id ^ 1].cap += d; return d; }
            }
        }
        return 0;
    }
    ll maxflow(int s, int t) {
        ll flow = 0;
        while (bfs(s, t)) { it.assign(n, 0); ll f; while ((f = dfs(s, t, LLONG_MAX)) > 0) flow += f; }
        return flow;
    }
};

int N,M,C,S,K;
vector<ll> pop_;
vector<int> eu, ev; vector<ll> ecap;
vector<int> candNode; vector<ll> candCap;
vector<vector<int>> scen;
ll P_total;

ll evalObjective(const vector<int>& sel){
    int SRC=0, SINK=N+1;
    ll worst=0;
    for (int s=0;s<K;s++){
        vector<char> blocked(M,0);
        for (int e: scen[s]) blocked[e]=1;
        Dinic din(N+2);
        for (int d=1; d<=N; d++) if (pop_[d]>0) din.addEdge(SRC, d, pop_[d]);
        for (int e=0;e<M;e++){
            if (blocked[e]) continue;
            din.addEdge(eu[e], ev[e], ecap[e]);
            din.addEdge(ev[e], eu[e], ecap[e]);
        }
        for (int idx: sel) din.addEdge(candNode[idx], SINK, candCap[idx]);
        ll Fs = din.maxflow(SRC, SINK);
        ll denom = max(Fs, 1LL);
        ll Ts = (P_total + denom - 1) / denom;
        worst = max(worst, Ts);
    }
    return worst;
}

// One run of marginal-gain construction (visiting candidates in `order`) followed
// by best-improvement swap local search. Returns the final objective.
ll runOnce(const vector<int>& order, vector<int>& sel){
    sel.clear();
    vector<char> chosen(C, 0);
    for (int iter=0; iter<S; iter++){
        int best=-1; ll bestObj=LLONG_MAX;
        for (int c : order){
            if (chosen[c]) continue;
            sel.push_back(c);
            ll obj = evalObjective(sel);
            sel.pop_back();
            if (obj < bestObj){ bestObj=obj; best=c; }
        }
        sel.push_back(best); chosen[best]=1;
    }

    ll curObj = evalObjective(sel);
    for (int round=0; round<8; round++){
        bool improved=false;
        for (int pi=0; pi<(int)sel.size(); pi++){
            int old = sel[pi];
            int bestC=-1; ll bestObj=curObj;
            for (int c : order){
                if (chosen[c]) continue;
                sel[pi] = c;
                ll obj = evalObjective(sel);
                if (obj < bestObj){ bestObj=obj; bestC=c; }
            }
            sel[pi] = old;
            if (bestC != -1){
                sel[pi] = bestC;
                chosen[old]=0; chosen[bestC]=1;
                curObj = bestObj;
                improved = true;
            }
        }
        if (!improved) break;
    }
    return curObj;
}

int main(){
    scanf("%d %d %d %d %d", &N,&M,&C,&S,&K);
    pop_.assign(N+1,0);
    for (int i=1;i<=N;i++) scanf("%lld", &pop_[i]);
    eu.resize(M); ev.resize(M); ecap.resize(M);
    for (int e=0;e<M;e++) scanf("%d %d %lld", &eu[e], &ev[e], &ecap[e]);
    candNode.resize(C); candCap.resize(C);
    for (int i=0;i<C;i++) scanf("%d %lld", &candNode[i], &candCap[i]);
    scen.assign(K, {});
    for (int s=0;s<K;s++){
        int b; scanf("%d",&b);
        scen[s].resize(b);
        for (int j=0;j<b;j++) scanf("%d", &scen[s][j]);
    }
    P_total = 0;
    for (int i=1;i<=N;i++) P_total += pop_[i];
    if (P_total<=0) P_total=1;

    // Try a few different candidate-visiting orders (breaks tie-break-order
    // dependence of the greedy construction / local search) and keep the best.
    vector<int> ascOrder(C), descOrder(C), capOrder(C);
    for (int i=0;i<C;i++) ascOrder[i]=i;
    descOrder = ascOrder; reverse(descOrder.begin(), descOrder.end());
    capOrder = ascOrder;
    sort(capOrder.begin(), capOrder.end(), [&](int a,int b){ return candCap[a] > candCap[b]; });

    vector<int> bestSel; ll bestObj = LLONG_MAX;
    for (auto &order : {ascOrder, descOrder, capOrder}){
        vector<int> sel;
        ll obj = runOnce(order, sel);
        if (obj < bestObj){ bestObj = obj; bestSel = sel; }
    }

    for (int idx: bestSel) printf("%d ", idx);
    printf("\n");
    return 0;
}
