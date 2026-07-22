// Generator for "Which Lines to Thicken Before Winter" (grid-reinforcement-load-patterns).
//
// Emits a capacitated R x C grid substation network (column 0 = generator column,
// full supply; every (i,j)-(i,j+1) and (i,j)-(i+1,j) grid edge has a base capacity
// and an upgrade cost-per-unit), an upgrade budget B, and K=10 scenarios. Each
// scenario is a (load pattern, line-outage set) pair: a demand assignment over the
// right-hand nodes (columns >= mid) plus a set of outaged edges (forced to zero
// capacity for that scenario only).
//
// PLANTED STRUCTURE: at the single column boundary mid-1 -> mid there are exactly R
// crossing edges (one per row) -- the ONLY link between the generator side and the
// demand side. Two of those R rows are permanently WEAK (low capacity, cheap to
// upgrade) and are NEVER outaged. The other R-2 rows are GENEROUS (high capacity,
// pricier) crossing edges.
//
// TRAP (testId 4,6,8,9,10): five of the ten scenarios simultaneously outage ALL
// generous crossing edges (a multi-line contingency) and set demand well above what
// the two weak edges alone can carry. In every one of those scenarios the *only*
// active crossing edges are the two weak ones -- a min-cut that RECURS identically
// across all of them. The nominal (no-outage) scenario is comfortably served by the
// generous edges alone, so those generous edges carry the largest raw flow in the
// nominal scenario even though reinforcing them helps none of the contingencies.
#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    int Rs[10] = {3, 3, 4, 4, 4, 5, 5, 5, 6, 6};
    int Cs[10] = {4, 5, 5, 6, 6, 6, 7, 7, 8, 9};
    bool trapTest[10] = {false, false, false, true, false, true, false, true, true, true};
    int R = Rs[t - 1], C = Cs[t - 1];
    bool trap = trapTest[t - 1];
    int K = 10;
    int mid = C / 2;
    int N = R * C;

    auto nid = [&](int i, int j) { return i * C + j; };

    // ---- pick the two permanently weak crossing rows ----
    int w1 = rnd.next(0, R - 1);
    int w2 = rnd.next(0, R - 1);
    while (w2 == w1) w2 = rnd.next(0, R - 1);
    vector<char> isWeakRow(R, 0);
    isWeakRow[w1] = isWeakRow[w2] = 1;

    // ---- build the edge list: horizontal rows first, then vertical columns ----
    vector<int> eu, ev; vector<ll> ecap, ecost;
    vector<int> crossEdgeOfRow(R, -1);
    for (int i = 0; i < R; i++) {
        for (int j = 0; j + 1 < C; j++) {
            ll cap, cost;
            if (j == mid - 1) {
                if (isWeakRow[i]) { cap = rnd.next(8, 14); cost = rnd.next(1, 2); }
                else              { cap = rnd.next(140, 220); cost = rnd.next(3, 7); }
            } else { cap = rnd.next(230, 300); cost = rnd.next(1, 6); } // strictly above any crossing cap: crossing is always the tightest link on its chain
            eu.push_back(nid(i, j)); ev.push_back(nid(i, j + 1));
            ecap.push_back(cap); ecost.push_back(cost);
            if (j == mid - 1) crossEdgeOfRow[i] = (int)eu.size() - 1;
        }
    }
    // Vertical (cross-row) redistribution edges are only emitted on the DEMAND side
    // (columns >= mid). Column-0 generators already have direct, unlimited access to
    // every row, so pre-crossing verticals would let flow "relay" sideways through
    // spare feeder capacity and inflate a feeder edge's raw flow reading with no
    // relation to any real bottleneck. Keeping the left side a straight per-row chain
    // makes each row's crossing edge the sole, unambiguous constraint on that row.
    for (int j = mid; j < C; j++) {
        for (int i = 0; i + 1 < R; i++) {
            ll cap = rnd.next(230, 300), cost = rnd.next(1, 6);
            eu.push_back(nid(i, j)); ev.push_back(nid(i + 1, j));
            ecap.push_back(cap); ecost.push_back(cost);
        }
    }
    int M = (int)eu.size();

    ll weakTotal = 0, generousTotal = 0;
    vector<int> generousRows;
    for (int i = 0; i < R; i++) {
        if (isWeakRow[i]) weakTotal += ecap[crossEdgeOfRow[i]];
        else { generousTotal += ecap[crossEdgeOfRow[i]]; generousRows.push_back(i); }
    }

    // ---- budget: sized off the weak-edge total so a weak-edge-focused spend can
    //      roughly double/triple the recurring cut's capacity (real leverage), while
    //      staying far short of what any single scenario's full gap needs (leaves
    //      scoring headroom, and keeps a generous-edge-only spend visibly weaker) ----
    ll B = trap ? (ll)llround(weakTotal * 1.35 + generousTotal * 0.02) + 8 + t
                : (ll)llround(weakTotal * 1.35 + generousTotal * 0.22) + 8 + t;

    // ---- eligible demand nodes: right-hand side, columns >= mid ----
    vector<int> elig;
    for (int i = 0; i < R; i++) for (int j = mid; j < C; j++) elig.push_back(nid(i, j));
    int EL = (int)elig.size();

    // ---- scenarios ----
    // trap scenario indices (1-based within 1..K-1, i.e. scenarios 2..6 in 1-indexed
    // output numbering) get the full generous-crossing outage; scenario 1 is always a
    // calm, no-outage anchor so the baseline is guaranteed positive.
    vector<int> trapScen;
    if (trap) for (int s = 1; s <= 5; s++) trapScen.push_back(s);

    printf("%d %d %d %d %lld\n", R, C, M, K, B);
    for (int e = 0; e < M; e++) printf("%d %d %lld %lld\n", eu[e], ev[e], ecap[e], ecost[e]);

    for (int s = 0; s < K; s++) {
        bool isTrap = trap && s >= 1 && s <= 5;
        ll Dtarget;
        if (s == 0) {
            Dtarget = (ll)llround(generousTotal * 1.15);
        } else if (isTrap) {
            double factor = 3.0 + rnd.next(0, 200) / 100.0; // [3.0, 5.0]
            Dtarget = max((ll)1, (ll)llround(weakTotal * factor));
        } else {
            double factor = 0.70 + rnd.next(0, 60) / 100.0; // [0.70, 1.30]
            Dtarget = max((ll)1, (ll)llround(generousTotal * factor));
        }
        vector<ll> dem(N, 0);
        for (ll u = 0; u < Dtarget; u++) dem[elig[rnd.next(0, EL - 1)]]++;
        vector<pair<int, ll>> demList;
        for (int nodeId : elig) if (dem[nodeId] > 0) demList.push_back({nodeId, dem[nodeId]});
        printf("%d\n", (int)demList.size());
        for (auto& pr : demList) printf("%d %lld\n", pr.first, pr.second);
        vector<int> out;
        if (isTrap) for (int r : generousRows) out.push_back(crossEdgeOfRow[r]);
        printf("%d\n", (int)out.size());
        if (!out.empty()) {
            for (size_t k = 0; k < out.size(); k++) printf("%d%c", out[k], k + 1 < out.size() ? ' ' : '\n');
        }
    }
    return 0;
}
