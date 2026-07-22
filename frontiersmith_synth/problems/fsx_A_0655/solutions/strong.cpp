// TIER: strong
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Insight: (1) a star cell's order choice is FREE (every star cell is always
// self-consistent under every palette order here) -- so pick, per cell, an
// order that first maximizes the canonical-phase bonus, tie-broken by the
// LEAST-used order so far (spreads usage -> maximizes order diversity D
// instead of collapsing to one order). (2) an adapter cell's incompatible
// neighboring angles can only be reconciled by an order whose angle
// vocabulary is a COMMON REFINEMENT of both -- compute the minimal-cost
// valid order via gcd(angle differences, 24) instead of always paying for
// the universal (and expensive) order; only build the adapter when its
// strand-length payoff covers its (now minimal) cost.

static ll gcdll(ll a, ll b){ while (b){ a %= b; swap(a,b);} return a; }

int main(){
    int N, M;
    scanf("%d %d", &N, &M);
    vector<int> type(N);
    for (int i = 0; i < N; i++) scanf("%d", &type[i]);
    int p; scanf("%d", &p);
    vector<int> paletteK(p), paletteC(p);
    for (int i = 0; i < p; i++) scanf("%d %d", &paletteK[i], &paletteC[i]);
    ll W1, W2, W3; scanf("%lld %lld %lld", &W1, &W2, &W3);
    vector<int> cost(N);
    for (int i = 0; i < N; i++) scanf("%d", &cost[i]);
    vector<vector<int>> angles(N);
    vector<int> eu(M), ev(M), ea(M), elen(M);
    for (int i = 0; i < M; i++){
        scanf("%d %d %d %d", &eu[i], &ev[i], &ea[i], &elen[i]);
        angles[eu[i]].push_back(ea[i]);
        angles[ev[i]].push_back(ea[i]);
    }
    vector<ll> incidentLen(N, 0);
    for (int i = 0; i < M; i++){ incidentLen[eu[i]] += elen[i]; incidentLen[ev[i]] += elen[i]; }

    const int L = 24;
    static const int ADAPTER_ALLOWED[6] = {3,4,6,8,12,24}; // increasing k => increasing cost
    static const int ADAPTER_STEP[6]    = {8,6,4,3,2,1};   // matching step for each k above

    vector<ll> usage(2000, 0); // usage[k] for k up to L
    vector<string> outLine(N);

    for (int i = 0; i < N; i++){
        if (type[i] == 0){
            bool isolated = angles[i].empty();
            int bestK = paletteK[0], bestR = 0, bestBonus = -1; ll bestUsage = LLONG_MAX;
            for (int j = 0; j < p; j++){
                int k = paletteK[j], step = L / k;
                int phase, bonus;
                int r;
                if (isolated){ r = paletteC[j]; phase = r % step; bonus = 10; }
                else {
                    int rep = angles[i][0] % step; // forced residue (all edges share this cell's angle)
                    r = rep; phase = rep; bonus = (phase == paletteC[j]) ? 10 : 3;
                }
                ll u = usage[k];
                // prefer higher bonus; tie-break: least-used order so far (diversity)
                if (bonus > bestBonus || (bonus == bestBonus && u < bestUsage)){
                    bestBonus = bonus; bestUsage = u; bestK = k; bestR = r;
                }
            }
            usage[bestK]++;
            char buf[64];
            snprintf(buf, sizeof(buf), "1 %d %d\n", bestK, bestR);
            outLine[i] = buf;
        } else {
            // number-theoretic minimal adapter order
            ll G = 0;
            for (size_t j = 1; j < angles[i].size(); j++)
                G = gcdll(G, llabs((ll)angles[i][j] - (ll)angles[i][0]));
            ll Gp = (G == 0) ? (ll)L : gcdll(G, (ll)L);
            int chosenK = 24, chosenStep = 1;
            for (int t = 0; t < 6; t++){
                if (Gp % ADAPTER_STEP[t] == 0){ chosenK = ADAPTER_ALLOWED[t]; chosenStep = ADAPTER_STEP[t]; break; }
            }
            int rep = angles[i].empty() ? 0 : (angles[i][0] % chosenStep);
            ll benefit = W2 * incidentLen[i];
            ll expense = W3 * (ll)cost[i] * chosenK;
            if (!angles[i].empty() && benefit > expense){
                char buf[64];
                snprintf(buf, sizeof(buf), "1 %d %d\n", chosenK, rep);
                outLine[i] = buf;
            } else {
                outLine[i] = "0 0 0\n";
            }
        }
    }
    for (int i = 0; i < N; i++) fputs(outLine[i].c_str(), stdout);
    return 0;
}
