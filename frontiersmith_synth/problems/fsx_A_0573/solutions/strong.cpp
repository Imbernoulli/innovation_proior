// TIER: strong
// Insight: the score is the PEAK of a discrete integer-tick interference sum, so
// what matters is not a reflector's raw gain nor its continuous distance but its
// exact integer arrival tick A(f,R)=D(S,R)+D(R,f) under the octile lattice metric,
// together with its facet parity. For each listener we bucket the CONSTRUCTIVE
// candidates (those with (A+p) even) by exact tick, find the tick whose gains sum
// the highest (the iso-chronal cluster -- a lattice staircase that sits OFF the
// smooth Euclidean focal ring), and take that set. Then, because the budget K is
// scarce and a reflector re-emits toward every listener, we allocate greedily:
// install whole clusters in decreasing cluster-value order, filling partial budget
// with the loudest members. This reads the simulator's true metric instead of
// importing continuous-optics geometry.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static inline ll octile(ll ax, ll ay, ll bx, ll by){
    ll dx = llabs(ax-bx), dy = llabs(ay-by);
    ll mx = max(dx,dy), mn = min(dx,dy);
    return 2*mx + mn;
}

int main(){
    ll G, sx, sy, g0; int M, N; ll K;
    if (scanf("%lld %lld %lld %lld %d %d %lld", &G,&sx,&sy,&g0,&M,&N,&K) != 7) return 0;
    vector<ll> fx(M+1), fy(M+1);
    for (int i = 1; i <= M; i++) scanf("%lld %lld",&fx[i],&fy[i]);
    vector<ll> cx(N+1), cy(N+1), cg(N+1), cp(N+1);
    for (int j = 1; j <= N; j++)
        scanf("%lld %lld %lld %lld",&cx[j],&cy[j],&cg[j],&cp[j]);

    // Per listener: best constructive iso-chronal cluster.
    struct Cluster { ll value; vector<int> ids; };
    vector<Cluster> clusters;
    for (int f = 1; f <= M; f++){
        unordered_map<ll, pair<ll,vector<int>>> byTick; // tick -> (sumgain, ids)
        byTick.reserve(N*2 + 8);
        for (int j = 1; j <= N; j++){
            ll A = octile(sx,sy,cx[j],cy[j]) + octile(cx[j],cy[j],fx[f],fy[f]);
            if (((A + cp[j]) & 1LL) != 0) continue;      // destructive -> skip
            auto &e = byTick[A];
            e.first += cg[j];
            e.second.push_back(j);
        }
        ll bestV = 0; vector<int> bestIds;
        for (auto &kv : byTick){
            if (kv.second.first > bestV){ bestV = kv.second.first; bestIds = kv.second.second; }
        }
        if (!bestIds.empty()) clusters.push_back({bestV, bestIds});
    }
    // Allocate scarce budget: richest clusters first; loudest members within a cluster.
    sort(clusters.begin(), clusters.end(),
         [](const Cluster&a, const Cluster&b){ return a.value > b.value; });
    vector<char> used(N+1, 0);
    vector<int> install;
    for (auto &cl : clusters){
        if ((ll)install.size() >= K) break;
        vector<int> ids = cl.ids;
        sort(ids.begin(), ids.end(), [&](int a, int b){ return cg[a] > cg[b]; });
        for (int j : ids){
            if ((ll)install.size() >= K) break;
            if (used[j]) continue;
            used[j] = 1;
            install.push_back(j);
        }
    }
    printf("%lld\n", (ll)install.size());
    for (size_t i = 0; i < install.size(); i++)
        printf("%d%c", install[i], (i+1==install.size()?'\n':' '));
    if (install.empty()) printf("\n");
    return 0;
}
