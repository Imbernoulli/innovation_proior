// TIER: strong
// Insight: replication turns this from partitioning into set-cover shaping.
//  1) Mine TRAFFIC-WEIGHTED affinity (sum of rows, not raw count) for the
//     base clustering -- a decoy pair that co-occurs often but carries little
//     traffic should not out-rank a pair that truly shares heavy order flow.
//  2) Quarantine components that are wide AND low-traffic onto their own
//     card before clustering, instead of trusting affinity blindly -- a wide
//     rarely-needed component dragged into a hot card taxes every order that
//     touches that card, not just its own rare orders.
//  3) Spend the replication budget on whichever components, if given a
//     second home, collapse the most order-weight from a two-card cover
//     down to a one-card cover (an exchange/marginal-gain ranking on the
//     REPLICATION decision itself, decoupled from step 1's partition).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int par[221], csize[221];
int find_(int x){ while (par[x] != x){ par[x] = par[par[x]]; x = par[x]; } return x; }

int main(){
    int F, Q, R; ll P, K;
    if (scanf("%d %d %lld %lld %d", &F, &Q, &P, &K, &R) != 5) return 0;
    vector<ll> w(F + 1);
    for (int i = 1; i <= F; i++) scanf("%lld", &w[i]);

    vector<ll> rows(Q);
    vector<vector<int>> qfields(Q);
    for (int i = 0; i < Q; i++){
        int m;
        scanf("%lld %d", &rows[i], &m);
        qfields[i].resize(m);
        for (int j = 0; j < m; j++) scanf("%d", &qfields[i][j]);
    }

    // ---- traffic-weighted heat per field & pairwise affinity ----
    vector<ll> heat(F + 1, 0);
    map<pair<int,int>, ll> aff;
    for (int i = 0; i < Q; i++){
        auto &qf = qfields[i];
        for (int c : qf) heat[c] += rows[i];
        int m = (int)qf.size();
        for (int a = 0; a < m; a++)
            for (int b = a + 1; b < m; b++){
                int u = qf[a], v = qf[b];
                if (u > v) swap(u, v);
                aff[{u, v}] += rows[i];
            }
    }

    // ---- quarantine: wide AND low-traffic components skip clustering ----
    vector<ll> wsorted(w.begin() + 1, w.end());
    vector<ll> hsorted(heat.begin() + 1, heat.end());
    sort(wsorted.begin(), wsorted.end());
    sort(hsorted.begin(), hsorted.end());
    ll medW = wsorted[wsorted.size() / 2];
    ll medH = hsorted[hsorted.size() / 2];
    vector<char> quarantined(F + 1, 0);
    for (int i = 1; i <= F; i++)
        if (w[i] > 2 * medW && heat[i] < medH) quarantined[i] = 1;

    // ---- base clustering: traffic-weighted affinity, size-capped union-find ----
    vector<pair<ll, pair<int,int>>> edges;
    edges.reserve(aff.size());
    for (auto &kv : aff){
        if (quarantined[kv.first.first] || quarantined[kv.first.second]) continue;
        edges.push_back({kv.second, kv.first});
    }
    sort(edges.begin(), edges.end(), [](const pair<ll,pair<int,int>>&a, const pair<ll,pair<int,int>>&b){
        if (a.first != b.first) return a.first > b.first;
        return a.second < b.second;
    });

    for (int i = 1; i <= F; i++){ par[i] = i; csize[i] = 1; }
    const int SIZE_CAP = 7;
    for (auto &e : edges){
        int u = find_(e.second.first), v = find_(e.second.second);
        if (u == v) continue;
        if (csize[u] + csize[v] <= SIZE_CAP){ par[u] = v; csize[v] += csize[u]; }
    }

    vector<int> home(F + 1);
    for (int i = 1; i <= F; i++) home[i] = find_(i);   // quarantined fields never unioned -> home=i

    unordered_map<int, ll> cardW;
    for (int i = 1; i <= F; i++) cardW[home[i]] += w[i];

    // ---- replication: rank components by savings from collapsing a
    //      2-card cover to 1 card, ACCUMULATED (summed) over every order
    //      where the component is the lone field on its side -- a component
    //      that plays this bridging role in many orders (like a hot narrow
    //      field pulled by both sides) should outrank one that only helps a
    //      single order once. ----
    map<pair<int,int>, ll> accum;   // (field, target card) -> summed savings
    for (int i = 0; i < Q; i++){
        auto &qf = qfields[i];
        map<int, vector<int>> byCard;
        for (int c : qf) byCard[home[c]].push_back(c);
        if (byCard.size() != 2) continue;
        auto it = byCard.begin();
        int cardP = it->first; auto &groupP = it->second; ++it;
        int cardQ = it->first; auto &groupQ = it->second;
        ll wp = cardW[cardP], wq = cardW[cardQ];
        // two separate cards, each pays its own pull fee K + its own footprint ceiling
        ll costP = K + (rows[i] * wp + P - 1) / P;
        ll costQ = K + (rows[i] * wq + P - 1) / P;
        ll oldCost = costP + costQ;
        if (groupP.size() == 1){
            int f = groupP[0];
            ll save = oldCost - costQ;              // collapses to just card Q
            if (save > 0) accum[{f, cardQ}] += save;
        }
        if (groupQ.size() == 1){
            int f = groupQ[0];
            ll save = oldCost - costP;              // collapses to just card P
            if (save > 0) accum[{f, cardP}] += save;
        }
    }

    // reduce to each field's single best (highest accumulated-savings) target
    map<int, pair<int, ll>> best;   // field -> (target card, total savings)
    for (auto &kv : accum){
        int f = kv.first.first, target = kv.first.second; ll save = kv.second;
        auto &cur = best[f];
        if (save > cur.second) cur = {target, save};
    }

    vector<pair<ll,int>> cands;   // (total savings, field) for deterministic sort
    for (auto &kv : best) cands.push_back({kv.second.second, kv.first});
    sort(cands.begin(), cands.end(), [](const pair<ll,int>&a, const pair<ll,int>&b){
        if (a.first != b.first) return a.first > b.first;
        return a.second < b.second;
    });

    vector<int> braid(F + 1, 0);
    int used = 0;
    for (auto &c : cands){
        if (used >= R) break;
        int f = c.second;
        int target = best[f].first;
        if (target == home[f]) continue;
        braid[f] = target;
        used++;
    }

    for (int i = 1; i <= F; i++) printf("%d %d\n", home[i], braid[i]);
    return 0;
}
