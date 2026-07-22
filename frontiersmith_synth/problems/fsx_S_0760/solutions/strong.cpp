// TIER: strong
// Insight: the ice is a SHARED, DECAYING asset, so the right question for a convoy is not
// "what's my cheapest route right now" but "is it worth me PAYING to break a corridor that
// is currently expensive (full/virgin), because the discount it hands to the near-future
// convoys that fall inside its regrow window outweighs that one-time superlinear premium?"
//
// Precompute all-pairs shortest-path distances/trees under the VIRGIN (fully refrozen)
// cost f+C^1.5 once. Then replay convoys in ready-time order maintaining the REAL per-edge
// break/regrow state:
//   (a) DEFAULT: a fresh Dijkstra using each edge's CURRENT real dynamic cost. This alone
//       makes a convoy automatically discover and reuse any corridor a PRIOR convoy just
//       warmed up -- no extra bookkeeping needed for "followers".
//   (b) INVESTMENT: for every candidate edge e, estimate the net gain of detouring THIS
//       convoy through e right now (paying e's real current premium) plus the estimated
//       benefit to near-future convoys that fall inside e's regrow window and would also
//       want e once it is warm. If the best such net gain is positive, take that detour
//       instead of the default path -- i.e. deliberately be the "trailblazer" even though
//       it looks locally worse, because the group benefits.
// Commit whichever of (a)/(b) is chosen, update the real state along it exactly as the
// checker will (chronological per-edge time), and move to the next convoy.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N, M, K;
vector<int> eu, ev, eL, eC, er, ef;
vector<vector<pair<int,int>>> adj; // node -> (neighbor, edgeId)

vector<vector<ll>>  Dstatic;      // Dstatic[s][v]
vector<vector<int>> parE;         // parE[s][v] = edge id used to reach v on shortest tree from s
vector<vector<int>> parN;         // parN[s][v] = predecessor node

static inline ll virginCost(int e) { return (ll)ef[e] + (ll)llround(pow((double)eC[e], 1.5)); }

void dijkstraFrom(int s) {
    vector<ll>& dist = Dstatic[s];
    vector<int>& pe = parE[s];
    vector<int>& pn = parN[s];
    fill(dist.begin(), dist.end(), LLONG_MAX);
    dist[s] = 0;
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<>> pq;
    pq.push({0, s});
    while (!pq.empty()) {
        auto [du, u] = pq.top(); pq.pop();
        if (du > dist[u]) continue;
        for (auto& pr : adj[u]) {
            int v = pr.first, eid = pr.second;
            ll nd = du + virginCost(eid);
            if (nd < dist[v]) { dist[v] = nd; pe[v] = eid; pn[v] = u; pq.push({nd, v}); }
        }
    }
}

// reconstruct the path (edge ids, in order) from s to t on the tree rooted at s
vector<int> pathFrom(int s, int t) {
    vector<int> rev;
    int cur = t;
    while (cur != s) { rev.push_back(parE[s][cur]); cur = parN[s][cur]; }
    reverse(rev.begin(), rev.end());
    return rev;
}

int main() {
    if (scanf("%d %d %d", &N, &M, &K) != 3) return 0;
    eu.assign(M + 1, 0); ev.assign(M + 1, 0); eL.assign(M + 1, 0);
    eC.assign(M + 1, 0); er.assign(M + 1, 0); ef.assign(M + 1, 0);
    adj.assign(N + 1, {});
    for (int i = 1; i <= M; i++) {
        scanf("%d %d %d %d %d %d", &eu[i], &ev[i], &eL[i], &eC[i], &er[i], &ef[i]);
        adj[eu[i]].push_back({ev[i], i});
        adj[ev[i]].push_back({eu[i], i});
    }
    vector<int> co(K + 1), cd(K + 1), cready(K + 1);
    for (int i = 1; i <= K; i++) scanf("%d %d %d", &co[i], &cd[i], &cready[i]);

    Dstatic.assign(N + 1, vector<ll>(N + 1));
    parE.assign(N + 1, vector<int>(N + 1));
    parN.assign(N + 1, vector<int>(N + 1));
    for (int s = 1; s <= N; s++) dijkstraFrom(s);

    vector<ll> costStatic(K + 1);
    for (int i = 1; i <= K; i++) costStatic[i] = Dstatic[co[i]][cd[i]];

    vector<int> order(K);
    for (int i = 0; i < K; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (cready[a] != cready[b]) return cready[a] < cready[b];
        return a < b;
    });

    vector<char> broken(M + 1, 0);
    vector<ll> lastBreak(M + 1, 0);
    const ll WARM_THICK = 15;
    const int LOOKAHEAD = 60;

    vector<ll> outT(K + 1);
    vector<vector<int>> outPath(K + 1);

    for (int idx = 0; idx < K; idx++) {
        int i = order[idx];
        ll t = cready[i];

        // (a) default: Dijkstra using current REAL dynamic weights
        vector<ll> dist(N + 1, LLONG_MAX);
        vector<int> pe(N + 1), pn(N + 1);
        dist[co[i]] = 0;
        priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<>> pq;
        pq.push({0, co[i]});
        while (!pq.empty()) {
            auto [du, u] = pq.top(); pq.pop();
            if (du > dist[u]) continue;
            if (u == cd[i]) break;
            for (auto& pr : adj[u]) {
                int v = pr.first, eidx = pr.second;
                ll th;
                if (!broken[eidx]) th = (ll)eC[eidx];
                else { ll elapsed = max(0LL, t - lastBreak[eidx]); th = min((ll)eC[eidx], (ll)er[eidx] * elapsed); }
                ll w = (ll)ef[eidx] + (ll)llround(pow((double)th, 1.5));
                ll nd = du + w;
                if (nd < dist[v]) { dist[v] = nd; pe[v] = eidx; pn[v] = u; pq.push({nd, v}); }
            }
        }
        vector<int> defaultPath;
        { int cur = cd[i]; while (cur != co[i]) { defaultPath.push_back(pe[cur]); cur = pn[cur]; } reverse(defaultPath.begin(), defaultPath.end()); }

        // (b) investment scan
        ll bestGain = 0;
        int bestEdge = -1;
        vector<int> bestPath;
        for (int e = 1; e <= M; e++) {
            ll realTh;
            if (!broken[e]) realTh = (ll)eC[e];
            else { ll elapsed = max(0LL, t - lastBreak[e]); realTh = min((ll)eC[e], (ll)er[e] * elapsed); }
            ll realPremium = (ll)ef[e] + (ll)llround(pow((double)realTh, 1.5));
            int u = eu[e], v = ev[e];
            ll optA = Dstatic[co[i]][u] + realPremium + Dstatic[v][cd[i]];
            ll optB = Dstatic[co[i]][v] + realPremium + Dstatic[u][cd[i]];
            bool useA = optA <= optB;
            ll detourReal = useA ? optA : optB;
            ll gain = costStatic[i] - detourReal;

            ll windowE = (ll)eC[e] / max(1, er[e]);
            ll warmCost = (ll)ef[e] + (ll)llround(pow((double)WARM_THICK, 1.5));
            for (int k = idx + 1; k < K && k <= idx + LOOKAHEAD; k++) {
                int j = order[k];
                if (cready[j] - t > windowE) break;
                ll oA = Dstatic[co[j]][u] + warmCost + Dstatic[v][cd[j]];
                ll oB = Dstatic[co[j]][v] + warmCost + Dstatic[u][cd[j]];
                ll detourWarm = min(oA, oB);
                ll bj = costStatic[j] - detourWarm;
                if (bj > 0) gain += bj;
            }
            if (gain > bestGain) {
                bestGain = gain; bestEdge = e;
                vector<int> p1 = useA ? pathFrom(co[i], u) : pathFrom(co[i], v);
                vector<int> p2 = useA ? pathFrom(v, cd[i]) : pathFrom(u, cd[i]);
                bestPath = p1;
                bestPath.push_back(e);
                for (int x : p2) bestPath.push_back(x);
            }
        }

        vector<int>& chosen = (bestEdge != -1) ? bestPath : defaultPath;
        if (chosen.empty() && co[i] != cd[i]) chosen = defaultPath; // safety net

        outT[i] = t;
        outPath[i] = chosen;

        ll cur_time = t;
        for (int eidx : chosen) {
            broken[eidx] = 1;
            lastBreak[eidx] = cur_time;
            cur_time += eL[eidx];
        }
    }

    for (int i = 1; i <= K; i++) {
        printf("%lld %d", outT[i], (int)outPath[i].size());
        for (int eidx : outPath[i]) printf(" %d", eidx);
        printf("\n");
    }
    return 0;
}
