// TIER: strong
// Insight: on any single spoke (primitive direction from the tower) at most
// one selected mirror can ever have positive value -- every farther mirror
// on that spoke is blocked by a nearer one -- so restrict to (at most) one
// representative per spoke, and treat WHICH member of a spoke to use as
// free (a lower-quality point on the same spoke is never blocked either).
//
// Which spokes to spend the M-mirror budget on is then a marginal-value
// selection under interference: every mirror's realized value can only ever
// DROP as more neighbours get built (shading only ever removes sun-steps,
// never adds them) -- a diminishing-returns / submodular structure. So
// instead of committing to the M raw-highest-quality spokes up front (the
// geometric sweet spot), run a LAZY marginal-gain greedy: repeatedly take
// the spoke whose best member currently has the highest REALIZED value
// given everything already built, re-scoring on the fly as earlier picks
// shade later candidates. A spoke that looked great in isolation but has
// since been shaded flat by a higher-priority neighbour drops down the
// queue, and the marginal mirror lands instead at whichever spoke is still
// clean -- the interference-free frontier -- even if its raw quality is
// lower.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static ll igcd(ll a, ll b) {
    a = llabs(a); b = llabs(b);
    while (b) { ll t = a % b; a = b; b = t; }
    return a;
}

int P, M, K;
vector<ll> X, Y, Q, DX, DY, E;

static inline ll encPos(ll x, ll y) { return (x + 2000000000LL) * 4000000001LL + (y + 2000000000LL); }

int main() {
    if (scanf("%d %d %d", &P, &M, &K) != 3) return 0;
    X.assign(P + 1, 0); Y.assign(P + 1, 0); Q.assign(P + 1, 0);
    for (int i = 1; i <= P; i++) scanf("%lld %lld %lld", &X[i], &Y[i], &Q[i]);
    DX.assign(K + 1, 0); DY.assign(K + 1, 0); E.assign(K + 1, 0);
    ll Esum = 0;
    for (int k = 1; k <= K; k++) { scanf("%lld %lld %lld", &DX[k], &DY[k], &E[k]); Esum += E[k]; }

    // ---- bucket candidates by spoke direction (each spoke competes for one slot) ----
    map<pair<ll, ll>, vector<int>> groups;
    for (int i = 1; i <= P; i++) {
        ll g = igcd(X[i], Y[i]);
        groups[{X[i] / g, Y[i] / g}].push_back(i);
    }
    const int TOPK = 8;
    vector<vector<int>> spokeOpts;
    spokeOpts.reserve(groups.size());
    for (auto& kv : groups) {
        vector<int>& v = kv.second;
        sort(v.begin(), v.end(), [&](int a, int b) {
            if (Q[a] != Q[b]) return Q[a] > Q[b];
            return a < b;
        });
        if ((int)v.size() > TOPK) v.resize(TOPK);
        spokeOpts.push_back(v);
    }
    int NS = (int)spokeOpts.size();

    unordered_map<ll, int> occupied;
    vector<int> chosen;

    auto realizedValue = [&](int id) -> ll {
        ll val = 0;
        for (int k = 1; k <= K; k++) {
            ll tx = X[id] + DX[k], ty = Y[id] + DY[k];
            auto it = occupied.find(encPos(tx, ty));
            if (it != occupied.end() && it->second != id) continue;  // shaded
            val += E[k];
        }
        return Q[id] * val;
    };
    auto bestOfSpoke = [&](int s) -> pair<ll, int> {
        ll bestVal = -1; int best = -1;
        for (int id : spokeOpts[s]) {
            ll v = realizedValue(id);
            if (v > bestVal) { bestVal = v; best = id; }
        }
        return {bestVal, best};
    };

    // lazy marginal-gain greedy (values only ever decrease as the selection grows)
    priority_queue<pair<ll, int>> pq;
    for (int s = 0; s < NS; s++) {
        auto [v, id] = bestOfSpoke(s);
        if (v > 0) pq.push({v, s});
    }
    vector<int> chosenSpoke;   // spoke index each chosen mirror came from
    while (!pq.empty() && (int)chosen.size() < M) {
        auto [val, s] = pq.top(); pq.pop();
        auto [actualVal, actualId] = bestOfSpoke(s);
        if (actualVal <= 0) continue;
        if (pq.empty() || actualVal >= pq.top().first) {
            chosen.push_back(actualId);
            chosenSpoke.push_back(s);
            occupied[encPos(X[actualId], Y[actualId])] = actualId;
        } else {
            pq.push({actualVal, s});
        }
    }

    // ---- local-search refinement: swap the currently weakest picks for a
    // leftover spoke's candidate if that raises total realized value. The
    // forward-only greedy above never revisits a decision once a later pick
    // has shaded it; this pass cleans up exactly that residual damage. ----
    vector<int> leftoverSpokes;
    while (!pq.empty()) { leftoverSpokes.push_back(pq.top().second); pq.pop(); }
    int rounds = 6;
    for (int round = 0; round < rounds; round++) {
        vector<int> order(chosen.size());
        iota(order.begin(), order.end(), 0);
        sort(order.begin(), order.end(), [&](int a, int b) {
            return realizedValue(chosen[a]) < realizedValue(chosen[b]);
        });
        bool improved = false;
        int limit = min((int)order.size(), 900);
        for (int oi = 0; oi < limit; oi++) {
            int idx = order[oi];
            ll curVal = realizedValue(chosen[idx]);
            occupied.erase(encPos(X[chosen[idx]], Y[chosen[idx]]));
            ll bestGain = 0; int bestLeftoverPos = -1, bestId = -1;
            for (int lp = 0; lp < (int)leftoverSpokes.size(); lp++) {
                auto [v, id] = bestOfSpoke(leftoverSpokes[lp]);
                if (v - curVal > bestGain) { bestGain = v - curVal; bestLeftoverPos = lp; bestId = id; }
            }
            if (bestLeftoverPos >= 0) {
                int newSpoke = leftoverSpokes[bestLeftoverPos];
                leftoverSpokes[bestLeftoverPos] = chosenSpoke[idx];   // old spoke returns to the pool
                chosen[idx] = bestId;
                chosenSpoke[idx] = newSpoke;
                occupied[encPos(X[bestId], Y[bestId])] = bestId;
                improved = true;
            } else {
                occupied[encPos(X[chosen[idx]], Y[chosen[idx]])] = chosen[idx];
            }
        }
        if (!improved) break;
    }

    printf("%d\n", (int)chosen.size());
    for (int id : chosen) printf("%d ", id);
    printf("\n");
    return 0;
}
