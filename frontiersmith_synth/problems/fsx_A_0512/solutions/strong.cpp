// TIER: strong
// Firebreak hardening. The insight: capacity is routing, not armor.
//   1. SACRIFICE ground-sink pockets (out-degree 0, low value): give them cap 0 so
//      they trip early and shed their load harmlessly to ground -- a firebreak.
//   2. Cover every kept line's base load so nothing trips at rest.
//   3. Pour the freed budget into SURGE margin on the lines that, when they fail,
//      cascade the most value away (the backbone entrances), sized to the worst
//      single incoming surge over the scenario sweep. Priority = value-at-risk.
// So every cascade in the sweep dies in a sacrificed pocket instead of crossing a
// backbone.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(){
    ll L, M, S, B;
    if (scanf("%lld %lld %lld %lld", &L, &M, &S, &B) != 4) return 0;
    vector<ll> w(L), v(L);
    ll maxV = 0;
    for (ll i = 0; i < L; i++){ scanf("%lld %lld", &w[i], &v[i]); maxV = max(maxV, v[i]); }
    vector<vector<pair<int,int>>> out(L);
    vector<int> outdeg(L, 0);
    for (ll e = 0; e < M; e++){
        int u, j, num; scanf("%d %d %d", &u, &j, &num);
        out[u].push_back({j, num}); outdeg[u]++;
    }
    vector<int> scen(S);
    for (ll i = 0; i < S; i++) scanf("%d", &scen[i]);

    const ll DEN = 1000;

    // value at risk if line i trips: its own value + value it cascades onward to.
    // (reroute graph is acyclic; memoize, guard against re-entry just in case.)
    vector<ll> reach(L, -1);
    vector<char> onstk(L, 0);
    // iterative post-order to avoid deep recursion on long chains
    {
        vector<int> order; order.reserve(L);
        vector<char> vis(L, 0);
        for (int s = 0; s < L; s++){
            if (vis[s]) continue;
            vector<pair<int,int>> st; st.push_back({s, 0});
            while (!st.empty()){
                auto& top = st.back();
                int u = top.first;
                if (top.second == 0){ vis[u] = 1; }
                if (top.second < (int)out[u].size()){
                    int j = out[u][top.second].first;
                    top.second++;
                    if (!vis[j]) st.push_back({j, 0});
                } else { order.push_back(u); st.pop_back(); }
            }
        }
        for (int u : order){
            ll r = v[u];
            for (auto& e : out[u]) if (reach[e.first] >= 0) r += reach[e.first];
            reach[u] = r;
        }
    }

    // worst single incoming surge each line sees over the sweep (one trigger trips).
    vector<ll> peak(L);
    for (ll i = 0; i < L; i++) peak[i] = w[i];
    for (int t : scen)
        for (auto& e : out[t]){
            ll cand = w[t] * (ll)e.second / DEN;
            peak[e.first] = max(peak[e.first], w[e.first] + cand);
        }

    // 1+2: sacrifice ground-sink low-value pockets, cover base for the rest.
    vector<ll> cap(L, 0);
    ll used = 0;
    for (ll i = 0; i < L; i++){
        bool sacrifice = (outdeg[i] == 0 && v[i] * 2 < maxV);   // pockets
        if (sacrifice) cap[i] = 0;
        else { cap[i] = w[i]; used += w[i]; }
    }

    // 3: surge boosts, highest value-at-risk first.
    vector<int> cand;
    for (ll i = 0; i < L; i++)
        if (cap[i] > 0 || w[i] > 0){   // a kept line
            if (peak[i] > cap[i]) cand.push_back((int)i);
        }
    sort(cand.begin(), cand.end(), [&](int a, int b){ return reach[a] > reach[b]; });
    for (int i : cand){
        ll need = peak[i] - cap[i];
        if (need <= 0) continue;
        if (used + need <= B){ cap[i] += need; used += need; }
    }

    // any remaining budget: pad the very highest value-at-risk line for safety.
    if (used < B && !cand.empty()){
        cap[cand[0]] += (B - used);
    }

    for (ll i = 0; i < L; i++) printf("%lld%c", cap[i], i + 1 < L ? ' ' : '\n');
    return 0;
}
