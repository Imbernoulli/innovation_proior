#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Place clinics before queues explode".
//
// Input:  n m Bud QW ; then n lines x y lambda (demand nodes) ; then m lines
//         X Y mu cost (candidate facility sites). QW converts expected queueing
//         wait into the same cost currency as travel distance (e.g. a value-of-
//         time coefficient); it is read from the input, not hardcoded, because
//         its natural scale tracks each test's coordinate range.
// Output: k ; then k distinct facility indices (1..m, the OPEN set) ; then n
//         integers, the facility each demand node is routed to (must be a
//         member of the open set).
//
// Feasibility: sum(cost over open) <= Bud; every assignment targets an open
//   facility; for every open facility f, load_f = sum of routed lambda MUST be
//   strictly < mu_f (queueing stability). Any violation -> score 0.
//
// Objective (MIN): F = travel + QW * queue, where
//   travel = sum_i lambda_i * EuclideanDist(i, assigned facility)
//   queue  = sum_{f open, load_f>0} load_f / (mu_f - load_f)
//
// Baseline B (checker-computed, budget-blind nearest-facility reference): route
//   every demand node to whichever of the ALL m candidates is geometrically
//   nearest (ignoring budget, cost and load entirely -- this is exactly the
//   "obvious" k-median-style move, just with unrestricted facility access).
//   Its queue term uses a clipped denominator max(1, mu_f-load_f) so a locally
//   over-saturated site never breaks the reference number. This is what the
//   TRIVIAL reference under-performs by roughly an order of magnitude (one
//   single facility, no distance/capacity reasoning at all).
// Score (min): sc = min(1000, 100*B/max(1,F)); ratio = sc/1000.
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();
    ll Bud = inf.readLong();
    ll QW = inf.readLong();

    vector<ll> dx(n + 1), dy(n + 1), dlam(n + 1);
    for (int i = 1; i <= n; i++) {
        dx[i] = inf.readLong();
        dy[i] = inf.readLong();
        dlam[i] = inf.readLong();
    }
    vector<ll> fx(m + 1), fy(m + 1), fmu(m + 1), fcost(m + 1);
    for (int j = 1; j <= m; j++) {
        fx[j] = inf.readLong();
        fy[j] = inf.readLong();
        fmu[j] = inf.readLong();
        fcost[j] = inf.readLong();
    }

    auto dist = [&](ll ax, ll ay, ll bx, ll by) -> double {
        double dxv = (double)(ax - bx), dyv = (double)(ay - by);
        return sqrt(dxv * dxv + dyv * dyv);
    };

    // ---- internal baseline B: budget-blind nearest-facility routing over ALL m sites ----
    vector<ll> loadB(m + 1, 0);
    double travelB = 0.0;
    for (int i = 1; i <= n; i++) {
        int best = 1; double bd = dist(dx[i], dy[i], fx[1], fy[1]);
        for (int j = 2; j <= m; j++) {
            double d = dist(dx[i], dy[i], fx[j], fy[j]);
            if (d < bd) { bd = d; best = j; }
        }
        loadB[best] += dlam[i];
        travelB += (double)dlam[i] * bd;
    }
    double queueB = 0.0;
    for (int j = 1; j <= m; j++) {
        if (loadB[j] > 0) {
            double denom = max(1.0, (double)(fmu[j] - loadB[j]));
            queueB += (double)loadB[j] / denom;
        }
    }
    double B = travelB + (double)QW * queueB;
    if (B < 1.0) B = 1.0;

    // ---- read + validate participant output ----
    int k = ouf.readInt(1, m, "k");
    vector<int> openList(k);
    vector<char> isOpen(m + 1, 0);
    ll totalCost = 0;
    for (int t = 0; t < k; t++) {
        int f = ouf.readInt(1, m, "open_facility");
        if (isOpen[f]) quitf(_wa, "facility %d listed as open more than once", f);
        isOpen[f] = 1;
        openList[t] = f;
        totalCost += fcost[f];
    }
    if (totalCost > Bud)
        quitf(_wa, "opening cost %lld exceeds budget %lld", totalCost, Bud);

    vector<ll> load(m + 1, 0);
    double travel = 0.0;
    for (int i = 1; i <= n; i++) {
        int f = ouf.readInt(1, m, "assign");
        if (!isOpen[f]) quitf(_wa, "demand node %d assigned to unopened facility %d", i, f);
        load[f] += dlam[i];
        travel += (double)dlam[i] * dist(dx[i], dy[i], fx[f], fy[f]);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    for (int t = 0; t < k; t++) {
        int f = openList[t];
        if (load[f] >= fmu[f])
            quitf(_wa, "facility %d saturated: load=%lld >= mu=%lld", f, load[f], fmu[f]);
    }

    double queue = 0.0;
    for (int t = 0; t < k; t++) {
        int f = openList[t];
        if (load[f] > 0) queue += (double)load[f] / (double)(fmu[f] - load[f]);
    }

    double F = travel + (double)QW * queue;
    if (!isfinite(F) || F < 0) quitf(_wa, "objective not finite/nonneg");
    if (F < 1.0) F = 1.0;

    double sc = min(1000.0, 100.0 * B / F);
    quitp(sc / 1000.0, "OK F=%.3f B=%.3f travel=%.3f queue=%.3f Ratio: %.6f",
          F, B, travel, queue, sc / 1000.0);
    return 0;
}
