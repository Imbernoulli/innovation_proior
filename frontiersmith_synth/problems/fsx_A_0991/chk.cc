#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Bolt and Braid: One Shop's Cutting Cards".
//
// Input:  F Q P K R ; then F lines w_f ; then Q lines "rows_i m_i c_1..c_{m_i}".
// Output: F lines "home_f braid_f" (braid_f=0 => no second card).
//
// Feasibility: 1<=home_f<=1e6 ; braid_f=0 or 1<=braid_f<=1e6 and braid_f!=home_f;
//   #{f : braid_f != 0} <= R ; exactly F lines, no trailing tokens.
//
// Objective (MIN): for order i, let hosts(c) = {home_c} or {home_c,braid_c}.
//   The order's cover is any set of card ids S such that every needed component
//   c has hosts(c) ∩ S != empty; its cost is
//       K * |S|  +  ceil(rows_i * sum_{s in S} width(s) / P)
//   where width(s) = sum of widths of every component whose home OR braid is s,
//   and K is a flat PULL FEE charged for every distinct card in the cover (a
//   card handed across the cutting table costs something even before you
//   measure any cloth off it). The order is charged its CHEAPEST cover.
//   F_total = sum over orders (minimize). Without K, packing every component
//   onto its own card would always be free-riding optimal (no reason to ever
//   share a card); K is what makes "fewer cards, chosen well" pay off, and
//   what a wasted-width merge (or an unnecessary extra card) must overcome.
//
// Since each component contributes at most one "free choice" edge (home or
// braid) per order, and orders touch <= 6 components, the minimum cover for an
// order is found EXACTLY by: components with braid=0 force their home card in;
// components with braid!=0 contribute an edge {home,braid} needing >=1 endpoint
// selected (choosing both is never beneficial) -- brute force over the <=6 such
// edges (<=64 combinations), taking the minimum K*|cover| + total card width.
//
// Baseline B (checker-computed, do-nothing): every component on ONE card ->
//   every order's cover is exactly that one card (fee K once) with the whole
//   schema's total width. B = sum_i (K + ceil(rows_i * totalWidth / P)).
//   Score (min): sc = min(1000, 100*B/max(1,F_total)); ratio = sc/1000.
// -----------------------------------------------------------------------------

const ll CARD_MAX = 1000000;

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    int F = inf.readInt();
    int Q = inf.readInt();
    ll P = inf.readLong();
    ll K = inf.readLong();
    int R = inf.readInt();

    vector<ll> w(F + 1);
    ll totalWidth = 0;
    for (int i = 1; i <= F; i++){ w[i] = inf.readLong(); totalWidth += w[i]; }

    vector<ll> rows(Q);
    vector<vector<int>> fields(Q);
    for (int i = 0; i < Q; i++){
        rows[i] = inf.readLong();
        int m = inf.readInt();
        fields[i].resize(m);
        for (int j = 0; j < m; j++) fields[i][j] = inf.readInt();
    }

    // ---- internal baseline B: everything on one card ----
    ll B = 0;
    for (int i = 0; i < Q; i++) B += K + (rows[i] * totalWidth + P - 1) / P;
    if (B <= 0) B = 1;

    // ---- read & validate participant output ----
    vector<ll> home(F + 1), braid(F + 1);
    int braidCount = 0;
    for (int i = 1; i <= F; i++){
        home[i] = ouf.readLong(1, CARD_MAX, "home");
        braid[i] = ouf.readLong(0, CARD_MAX, "braid");
        if (braid[i] != 0){
            if (braid[i] == home[i])
                quitf(_wa, "component %d: braid card equals home card", i);
            braidCount++;
        }
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after the assignment");
    if (braidCount > R)
        quitf(_wa, "replication budget exceeded: %d braided components > R=%d", braidCount, R);

    // ---- card widths (home contributes; braid, if any, contributes too) ----
    unordered_map<ll, ll> cardWidth;
    cardWidth.reserve(F * 2 + 16);
    for (int i = 1; i <= F; i++){
        cardWidth[home[i]] += w[i];
        if (braid[i] != 0) cardWidth[braid[i]] += w[i];
    }

    // ---- score every order with the exact cheapest-cover brute force ----
    ll Ftotal = 0;
    for (int i = 0; i < Q; i++){
        vector<ll> forced;                       // fields with no braid: home is mandatory
        vector<pair<ll,ll>> freeEdges;           // fields with braid: need >=1 of {home,braid}
        for (int c : fields[i]){
            if (braid[c] == 0) forced.push_back(home[c]);
            else freeEdges.push_back({home[c], braid[c]});
        }
        sort(forced.begin(), forced.end());
        forced.erase(unique(forced.begin(), forced.end()), forced.end());
        set<ll> forcedSet(forced.begin(), forced.end());

        // drop edges already satisfied by a forced endpoint
        vector<pair<ll,ll>> kept;
        for (auto &e : freeEdges)
            if (!forcedSet.count(e.first) && !forcedSet.count(e.second)) kept.push_back(e);

        int k = (int)kept.size();
        ll bestCost = -1;
        int combos = 1 << k;
        for (int mask = 0; mask < combos; mask++){
            set<ll> cover(forced.begin(), forced.end());
            for (int b = 0; b < k; b++)
                cover.insert((mask & (1 << b)) ? kept[b].second : kept[b].first);
            ll tw = 0;
            for (ll c : cover) tw += cardWidth[c];
            ll cost = K * (ll)cover.size() + (rows[i] * tw + P - 1) / P;
            if (bestCost < 0 || cost < bestCost) bestCost = cost;
        }
        if (bestCost < 0) bestCost = 0;   // no components in this order (shouldn't happen, m>=2)
        Ftotal += bestCost;
    }

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, Ftotal));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", Ftotal, B, sc / 1000.0);
    return 0;
}
