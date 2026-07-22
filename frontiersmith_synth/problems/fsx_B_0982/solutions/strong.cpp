// TIER: strong
// The insight: under multiplicative per-segment loss, the right notion of
// "shortest route" is the LOG-LOSS length (Dijkstra with weight -ln(retention)),
// not ditch length -- this gives every field its straightest route in the loss
// metric. Then, since only the WORST field's moisture counts, don't split flow
// equally: solve a water-filling equalization (binary search on a common target
// moisture tau; a field needing X=tau/R to hit tau) across every field's chosen
// path, respecting shared-segment capacity and the spring's total discharge --
// this deliberately gives the farthest/lossiest field a LARGER raw flow share
// to compensate for its heavier compounding loss, so every field converges to
// nearly the same delivered amount instead of the near fields hoarding it.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N, M, F; ll D, CAP, L;
vector<int> eu, ev; vector<ll> elen; vector<int> eret;
vector<vector<pair<int,int>>> adj;

vector<int> bfsPath(int target){
    vector<int> dist(N, -1), parentEdge(N, -1);
    queue<int> q; q.push(0); dist[0] = 0;
    while (!q.empty()){
        int u = q.front(); q.pop();
        if (u == target) break;
        for (auto &pr : adj[u]){
            int v = pr.first, eid = pr.second;
            if (dist[v] == -1){ dist[v] = dist[u] + 1; parentEdge[v] = eid; q.push(v); }
        }
    }
    vector<int> p; int v = target; p.push_back(v);
    while (v != 0){ int eid = parentEdge[v]; v = (eu[eid] == v) ? ev[eid] : eu[eid]; p.push_back(v); }
    reverse(p.begin(), p.end());
    return p;
}

// Dijkstra minimizing sum of -ln(retention); returns path + full retention product.
pair<vector<int>, double> retentionPath(int target, const vector<char>* forbidEdge = nullptr){
    vector<double> dist(N, 1e18);
    vector<int> parentEdge(N, -1);
    priority_queue<pair<double,int>, vector<pair<double,int>>, greater<>> pq;
    dist[0] = 0; pq.push({0.0, 0});
    while (!pq.empty()){
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u] + 1e-12) continue;
        for (auto &pr : adj[u]){
            int v = pr.first, eid = pr.second;
            if (forbidEdge && (*forbidEdge)[eid]) continue;
            double w = -log(eret[eid] / 1000.0);
            double nd = d + w;
            if (nd < dist[v] - 1e-12){ dist[v] = nd; parentEdge[v] = eid; pq.push({nd, v}); }
        }
    }
    vector<int> p; int v = target; double retProd = 1.0;
    if (dist[target] > 1e17){ return { {}, 0.0 }; }
    p.push_back(v);
    while (v != 0){
        int eid = parentEdge[v];
        retProd *= eret[eid] / 1000.0;
        v = (eu[eid] == v) ? ev[eid] : eu[eid];
        p.push_back(v);
    }
    reverse(p.begin(), p.end());
    return { p, retProd };
}

double pathRetention(const vector<int> &p){
    double r = 1.0;
    for (size_t i = 0; i + 1 < p.size(); i++){
        int a = min(p[i], p[i+1]), b = max(p[i], p[i+1]);
        for (auto &pr : adj[p[i]]) if (pr.first == p[i+1]){ r *= eret[pr.second] / 1000.0; break; }
    }
    return r;
}

ll edgeIdOf(int a, int b){
    for (auto &pr : adj[a]) if (pr.first == b) return pr.second;
    return -1;
}

ll pathNewLen(const vector<int> &p, const vector<char> &built){
    ll s = 0;
    for (size_t i = 0; i + 1 < p.size(); i++){
        int eid = (int)edgeIdOf(p[i], p[i+1]);
        if (!built[eid]) s += elen[eid];
    }
    return s;
}

int main(){
    scanf("%d %d %d %lld %lld %lld", &N, &M, &F, &D, &CAP, &L);
    vector<int> fieldId(F);
    for (int i = 0; i < F; i++) scanf("%d", &fieldId[i]);
    eu.resize(M); ev.resize(M); elen.resize(M); eret.resize(M); adj.assign(N, {});
    for (int i = 0; i < M; i++){
        scanf("%d %d %lld %d", &eu[i], &ev[i], &elen[i], &eret[i]);
        adj[eu[i]].push_back({ev[i], i});
        adj[ev[i]].push_back({eu[i], i});
    }

    // 1) log-loss shortest path per field.
    vector<vector<int>> ideal(F); vector<double> idealR(F);
    for (int i = 0; i < F; i++){
        auto pr = retentionPath(fieldId[i]);
        ideal[i] = pr.first; idealR[i] = pr.second;
    }

    // 2) priority order: worst retention first (the farthest/lossiest field).
    vector<int> order(F);
    for (int i = 0; i < F; i++) order[i] = i;
    sort(order.begin(), order.end(), [&](int a, int b){ return idealR[a] < idealR[b]; });

    vector<char> built(M, 0);
    ll usedLen = 0;
    vector<vector<int>> chosenPath(F);
    vector<double> R(F);
    for (int idx : order){
        ll newLen = pathNewLen(ideal[idx], built);
        if (usedLen + newLen <= L){
            chosenPath[idx] = ideal[idx];
            R[idx] = idealR[idx];
        } else {
            // fallback: fewest-hops path (always fits -- generator guarantees the
            // fewest-hops union alone is within L).
            vector<int> bp = bfsPath(fieldId[idx]);
            chosenPath[idx] = bp;
            R[idx] = pathRetention(bp);
            newLen = pathNewLen(bp, built);
        }
        for (size_t i = 0; i + 1 < chosenPath[idx].size(); i++){
            int eid = (int)edgeIdOf(chosenPath[idx][i], chosenPath[idx][i+1]);
            if (!built[eid]){ built[eid] = 1; usedLen += elen[eid]; }
        }
    }

    // 3) water-filling: binary search the largest common target moisture tau
    //    that is jointly feasible for every field's chosen path.
    auto feasible = [&](double tau) -> bool {
        vector<double> edgeFocus(M, 0.0);
        double springTotal = 0.0;
        for (int i = 0; i < F; i++){
            double X = tau / R[i];
            springTotal += X;
            if (springTotal > (double)D + 1e-6) return false;
            double cur = X;
            for (size_t k = 0; k + 1 < chosenPath[i].size(); k++){
                int eid = (int)edgeIdOf(chosenPath[i][k], chosenPath[i][k+1]);
                edgeFocus[eid] += cur;
                if (edgeFocus[eid] > (double)CAP + 1e-6) return false;
                cur *= eret[eid] / 1000.0;
            }
        }
        return true;
    };

    double lo = 0.0, hi = 1.0;
    while (feasible(hi)) hi *= 2.0;
    for (int it = 0; it < 60; it++){
        double mid = (lo + hi) / 2.0;
        if (feasible(mid)) lo = mid; else hi = mid;
    }
    double tau = lo * 0.999; // tiny safety margin against fp boundary rejection

    printf("%d\n", F);
    for (int i = 0; i < F; i++){
        double X = tau / R[i];
        printf("%d %.9f %d", fieldId[i], X, (int)chosenPath[i].size());
        for (int x : chosenPath[i]) printf(" %d", x);
        printf("\n");
    }
    return 0;
}
