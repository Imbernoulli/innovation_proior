#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Place clinics before queues explode"   family: facility-congestion-balancing
//
// n demand nodes (x,y,lambda) must each be routed to one OPEN facility, chosen
// from m candidate sites (X,Y,mu,cost) subject to a total opening BUDGET. Each
// open facility behaves like a single queueing server: assigned load
// lambda_f = sum of routed lambda_i must stay STRICTLY below its service rate
// mu_f, and it contributes an M/M/1-style expected-wait term
// lambda_f / (mu_f - lambda_f) to the objective, on top of the (lambda-weighted)
// Euclidean travel cost of getting demand to its facility.
//
// PLANTED STRUCTURE (never labeled in the input; a solver must discover it):
//   - TRAP cases (5,6,7,10): most demand sits in a tight geographic CLUSTER. A
//     "central" candidate site sits almost on top of the cluster (cheapest,
//     nearest to everyone in it) but its capacity mu is deliberately set BELOW
//     (or razor-thin above) the cluster's total lambda. A second "relief" site
//     sits farther away (extra travel) but has ample spare capacity. Routing
//     purely by distance (nearest-facility) floods the central site past
//     saturation -> either flatly infeasible or an exploding queueing term;
//     spreading some of the cluster's load onto the relief site costs a little
//     extra travel but avoids the 1/(mu-lambda) blowup entirely.
//   - PLANTED case (8): two well-separated clusters each have a small, cheap,
//     well-fitted "ideal" site nearby; a single "decoy" site sits between them
//     with huge capacity but costs nearly the whole budget, so affording it
//     precludes affording BOTH ideal sites (an opportunity-cost trap on the
//     facility-SELECTION side, not just assignment).
//   - NEEDLE case (9): the budget exactly covers a swarm of tiny, useless,
//     dirt-cheap "junk" sites; a single well-located, high-capacity "needle"
//     site costs more than any one piece of junk (so cost-ascending selection
//     buries it) but is the only way to reach enough total capacity to serve
//     all demand feasibly at all.
//
// Format:
//   line 1        : n m Bud
//   next n lines  : x y lambda           (demand node i, 1-indexed)
//   next m lines  : X Y mu cost          (candidate site j, 1-indexed)
// -----------------------------------------------------------------------------

static const ll CMAX = 20000;

struct Dem { ll x, y, lam; };
struct Fac { ll x, y, mu, cost; };

static ll clampll(ll v, ll lo, ll hi) { return max(lo, min(hi, v)); }

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    vector<Dem> dem;
    vector<Fac> fac;

    auto addUniformDemand = [&](int cnt, ll xlo, ll xhi, ll ylo, ll yhi, ll llo, ll lhi) {
        for (int i = 0; i < cnt; i++) {
            ll x = rnd.next(xlo, xhi), y = rnd.next(ylo, yhi);
            ll lam = rnd.next(llo, lhi);
            dem.push_back({x, y, lam});
        }
    };
    auto addClusterDemand = [&](int cnt, ll cx, ll cy, ll radius, ll llo, ll lhi) {
        for (int i = 0; i < cnt; i++) {
            ll x = clampll(cx + rnd.next(-radius, radius), 0, CMAX);
            ll y = clampll(cy + rnd.next(-radius, radius), 0, CMAX);
            ll lam = rnd.next(llo, lhi);
            dem.push_back({x, y, lam});
        }
    };

    ll TotalLambda = 0;

    if (testId == 1) {
        // ---- TINY sanity ----
        addUniformDemand(6, 0, 300, 0, 300, 5, 40);
        for (auto &d : dem) TotalLambda += d.lam;

        // hub deliberately parked in a corner (NOT centroid-based) -- a poor
        // travel choice, so funneling everyone there (the trivial reference)
        // stays a genuinely weak baseline.
        Fac hub{clampll(10 + rnd.next(0, 20), 0, 300), clampll(10 + rnd.next(0, 20), 0, 300),
                TotalLambda + rnd.next(25, 60), rnd.next(90, 150)};
        fac.push_back(hub);
        vector<ll> othCost;
        for (int j = 0; j < 3; j++) {
            ll mu = rnd.next(max((ll)3, TotalLambda / 3), max((ll)5, TotalLambda));
            ll cost = rnd.next(30, 200);
            fac.push_back({rnd.next(0, 300), rnd.next(0, 300), mu, cost});
            othCost.push_back(cost);
        }
        sort(othCost.begin(), othCost.end());
        ll Bud = hub.cost + othCost[0] + othCost[1] + rnd.next(0, 20);
        ll QW = 300;
        cout << dem.size() << " " << fac.size() << " " << Bud << " " << QW << "\n";
        for (auto &d : dem) cout << d.x << " " << d.y << " " << d.lam << "\n";
        for (auto &f : fac) cout << f.x << " " << f.y << " " << f.mu << " " << f.cost << "\n";
        return 0;
    }

    if (testId >= 2 && testId <= 4) {
        // ---- NORMAL scaling ----
        int n = (testId == 2 ? 20 : (testId == 3 ? 45 : 80));
        int m = (testId == 2 ? 5 : (testId == 3 ? 7 : 9));
        ll R = (testId == 2 ? 1000 : (testId == 3 ? 2000 : 3000));
        addUniformDemand(n, 0, R, 0, R, 1, 60);
        for (auto &d : dem) TotalLambda += d.lam;

        // hub cost is drawn from a strictly CHEAPER range than every other
        // candidate, so cost-ascending selection (the greedy tier) always buys it
        // first -> greedy stays FEASIBLE on the (non-trap) normal tests. It is
        // deliberately parked in a corner (a poor travel choice) so the
        // single-facility trivial reference stays a genuinely weak baseline.
        Fac hub{clampll(R / 20, 0, R), clampll(R / 20, 0, R),
                TotalLambda + rnd.next(30, 100), rnd.next(20, 80)};
        fac.push_back(hub);
        vector<ll> costs;
        costs.push_back(hub.cost);
        ll avgShare = max((ll)1, TotalLambda / m);
        for (int j = 0; j < m - 1; j++) {
            ll mu = rnd.next(max((ll)2, avgShare * 3 / 10), max((ll)4, avgShare * 13 / 10));
            ll cost = rnd.next(150, 900);
            fac.push_back({rnd.next(0LL, R), rnd.next(0LL, R), mu, cost});
            costs.push_back(cost);
        }
        vector<ll> sorted_costs = costs;
        sort(sorted_costs.begin(), sorted_costs.end());
        int afford = max(2, (int)sorted_costs.size() / 2);
        ll Bud = 0;
        for (int i = 0; i < afford; i++) Bud += sorted_costs[i];
        Bud += rnd.next(0, 50);
        Bud = max(Bud, hub.cost + rnd.next(5, 30));  // hub must ALWAYS be affordable (global feasibility guarantee)
        ll QW = R;
        cout << dem.size() << " " << fac.size() << " " << Bud << " " << QW << "\n";
        for (auto &d : dem) cout << d.x << " " << d.y << " " << d.lam << "\n";
        for (auto &f : fac) cout << f.x << " " << f.y << " " << f.mu << " " << f.cost << "\n";
        return 0;
    }

    if (testId == 5 || testId == 6 || testId == 7 || testId == 10) {
        // ---- TRAP: dense cluster near a cheap, capacity-tight "central" site;
        //      a farther "relief" site has ample spare capacity. ----
        int n, m;
        ll R;
        if (testId == 5) { n = 90; m = 10; R = 4000; }
        else if (testId == 6) { n = 140; m = 12; R = 5000; }
        else if (testId == 7) { n = 200; m = 14; R = 6000; }
        else { n = 2000; m = 55; R = 16000; }

        int clusterCount = (int)llround(0.65 * n);
        int scatterCount = n - clusterCount;
        ll ccx = rnd.next(R / 4, 3 * R / 4), ccy = rnd.next(R / 4, 3 * R / 4);
        ll radius = max((ll)20, R / 16);
        addClusterDemand(clusterCount, ccx, ccy, radius, 2, 80);
        addUniformDemand(scatterCount, 0, R, 0, R, 1, 60);

        ll clusterLambda = 0;
        for (int i = 0; i < clusterCount; i++) clusterLambda += dem[i].lam;
        for (auto &d : dem) TotalLambda += d.lam;

        // central: nearly on top of the cluster, cheap, capacity-tight
        double factor = (testId == 6) ? 1.0 : 0.85;   // 6 = razor-thin feasible, else infeasible-if-nearest
        ll muCentral;
        if (testId == 6) muCentral = clusterLambda + rnd.next(3, 10);
        else muCentral = max((ll)5, (ll)llround(clusterLambda * factor));
        Fac central{clampll(ccx + rnd.next(-radius / 5, radius / 5), 0, R),
                    clampll(ccy + rnd.next(-radius / 5, radius / 5), 0, R),
                    muCentral, rnd.next(40, 90)};
        fac.push_back(central);

        // relief: moderately farther, capacity sized to absorb the CLUSTER's
        // overflow only (not the whole instance) -- a genuinely good load-
        // spreading choice, not a do-everything fallback.
        double ang = rnd.next(0, 359) * acos(-1.0) / 180.0;
        ll off = (ll)llround((0.18 + 0.10 * (rnd.next(0, 100) / 100.0)) * R);
        ll rx = clampll(ccx + (ll)llround(off * cos(ang)), 0, R);
        ll ry = clampll(ccy + (ll)llround(off * sin(ang)), 0, R);
        Fac relief{rx, ry, (ll)llround(clusterLambda * 1.3) + rnd.next(20, 60), rnd.next(200, 450)};
        fac.push_back(relief);

        // farHub: the ONLY single site with enough capacity for the WHOLE
        // instance, deliberately parked in a far corner away from all demand
        // mass -- the naive/trivial "just fund one big site" fallback, but a
        // poor travel choice (this is what keeps the trivial reference weak).
        ll fhx = (ccx < R / 2) ? R - R / 20 : R / 20;
        ll fhy = (ccy < R / 2) ? R - R / 20 : R / 20;
        Fac farHub{clampll(fhx, 0, R), clampll(fhy, 0, R), TotalLambda + rnd.next(30, 120), rnd.next(150, 350)};
        fac.push_back(farHub);

        // for the largest test, plant a SECOND independent overload cluster too
        if (testId == 10) {
            ll ccx2 = rnd.next(R / 4, 3 * R / 4), ccy2 = rnd.next(R / 4, 3 * R / 4);
            ll radius2 = max((ll)20, R / 18);
            int cluster2Count = (int)llround(0.20 * n);
            size_t before = dem.size();
            addClusterDemand(cluster2Count, ccx2, ccy2, radius2, 2, 80);
            ll cluster2Lambda = 0;
            for (size_t i = before; i < dem.size(); i++) cluster2Lambda += dem[i].lam;
            TotalLambda += cluster2Lambda;
            Fac central2{clampll(ccx2 + rnd.next(-radius2 / 5, radius2 / 5), 0, R),
                        clampll(ccy2 + rnd.next(-radius2 / 5, radius2 / 5), 0, R),
                        max((ll)5, (ll)llround(cluster2Lambda * 0.85)), rnd.next(40, 90)};
            fac.push_back(central2);
            // bump farHub capacity to also cover this extra load
            fac[2].mu += cluster2Lambda + rnd.next(20, 80);
        }

        int placed = (int)fac.size();
        vector<ll> noiseCost;
        for (int j = placed; j < m; j++) {
            ll mu = rnd.next(max((ll)2, clusterLambda / 8), max((ll)5, clusterLambda / 2));
            ll cost = rnd.next(100, 700);
            fac.push_back({rnd.next(0LL, R), rnd.next(0LL, R), mu, cost});
            noiseCost.push_back(cost);
        }
        sort(noiseCost.begin(), noiseCost.end());
        ll Bud = fac[0].cost + fac[1].cost + fac[2].cost;   // central + relief + farHub
        if (testId == 10) Bud += fac[3].cost;
        int extra = min((int)noiseCost.size(), (testId == 10 ? 6 : 2));
        for (int i = 0; i < extra; i++) Bud += noiseCost[i];
        Bud += rnd.next(0, 30);
        ll QW = R;

        cout << dem.size() << " " << fac.size() << " " << Bud << " " << QW << "\n";
        for (auto &d : dem) cout << d.x << " " << d.y << " " << d.lam << "\n";
        for (auto &f : fac) cout << f.x << " " << f.y << " " << f.mu << " " << f.cost << "\n";
        return 0;
    }

    if (testId == 8) {
        // ---- PLANTED: two clusters, ideal near-each pair vs one costly decoy hub ----
        int n = 80, m = 9;
        ll R = 6000;
        ll c1x = R / 6, c1y = R / 6, c2x = 5 * R / 6, c2y = 5 * R / 6;
        int half = n / 2;
        addClusterDemand(half, c1x, c1y, R / 14, 4, 70);
        addClusterDemand(n - half, c2x, c2y, R / 14, 4, 70);
        ll c1L = 0, c2L = 0;
        for (int i = 0; i < half; i++) c1L += dem[i].lam;
        for (int i = half; i < n; i++) c2L += dem[i].lam;
        TotalLambda = c1L + c2L;

        Fac ideal1{clampll(c1x + rnd.next(-100, 100), 0, R), clampll(c1y + rnd.next(-100, 100), 0, R),
                (ll)llround(c1L * 1.15) + 5, rnd.next(90, 140)};
        Fac ideal2{clampll(c2x + rnd.next(-100, 100), 0, R), clampll(c2y + rnd.next(-100, 100), 0, R),
                (ll)llround(c2L * 1.15) + 5, rnd.next(90, 140)};
        ll idealSum = ideal1.cost + ideal2.cost;
        Fac decoy{R / 2, R / 2, (ll)llround(TotalLambda * 1.3) + 10, idealSum + rnd.next(60, 120)};
        fac.push_back(ideal1); fac.push_back(ideal2); fac.push_back(decoy);

        vector<ll> junkCost;
        for (int j = 3; j < m; j++) {
            ll mu = rnd.next(2LL, max((ll)6, TotalLambda / 10));
            ll cost = rnd.next(10, 60);
            fac.push_back({rnd.next(0LL, R), rnd.next(0LL, R), mu, cost});
            junkCost.push_back(cost);
        }
        ll Bud = decoy.cost;   // affords decoy ALONE, or ideal1+ideal2 (+maybe a little junk), not both groups
        ll QW = R;
        cout << dem.size() << " " << fac.size() << " " << Bud << " " << QW << "\n";
        for (auto &d : dem) cout << d.x << " " << d.y << " " << d.lam << "\n";
        for (auto &f : fac) cout << f.x << " " << f.y << " " << f.mu << " " << f.cost << "\n";
        return 0;
    }

    if (testId == 9) {
        // ---- NEEDLE: a swarm of cheap junk sites can exactly exhaust the budget;
        //      a well-located "needle" site costs more than any single piece of
        //      junk (so cost-ascending selection buries it) but is the only way
        //      to get GOOD travel; a separate cheap, badly-located "capSafety"
        //      site alone has enough capacity for everyone (so the instance --
        //      and the trivial/greedy references -- stay feasible even without
        //      ever finding the needle; only a value-aware pick actually uses it). ----
        int n = 110, m = 19;
        ll R = 5000;
        ll hx = R / 2, hy = R / 2;
        addClusterDemand((int)llround(0.55 * n), hx, hy, R / 6, 5, 45);
        addUniformDemand(n - (int)llround(0.55 * n), 0, R, 0, R, 5, 45);
        for (auto &d : dem) TotalLambda += d.lam;

        Fac needle{clampll(hx + rnd.next(-150, 150), 0, R), clampll(hy + rnd.next(-150, 150), 0, R),
                (ll)llround(TotalLambda * 0.55) + 10, 0};
        Fac capSafety{clampll(R - R / 20, 0, R), clampll(R / 20, 0, R),
                (ll)llround(TotalLambda * 1.2) + 15, rnd.next(15, 30)};

        int nJunk = m - 2;
        vector<ll> junkCost(nJunk);
        for (int j = 0; j < nJunk; j++) junkCost[j] = rnd.next(8, 25);
        ll junkTotal = 0; for (ll c : junkCost) junkTotal += c;
        needle.cost = clampll(junkTotal / max(1, nJunk / 2) + rnd.next(5, 15), 40, 100);   // pricier than any junk (<=25)
        // cost-ascending buys capSafety + (almost) all junk, exhausting the budget
        // strictly BEFORE it ever reaches needle's cost rank.
        ll Bud = capSafety.cost + junkTotal - rnd.next(3, 15);
        if (Bud < capSafety.cost) Bud = capSafety.cost + rnd.next(0, 10);

        fac.push_back(needle);
        fac.push_back(capSafety);
        for (int j = 0; j < nJunk; j++) {
            ll mu = rnd.next(1, 8);
            fac.push_back({rnd.next(0LL, R), rnd.next(0LL, R), mu, junkCost[j]});
        }
        ll QW = R;
        cout << dem.size() << " " << fac.size() << " " << Bud << " " << QW << "\n";
        for (auto &d : dem) cout << d.x << " " << d.y << " " << d.lam << "\n";
        for (auto &f : fac) cout << f.x << " " << f.y << " " << f.mu << " " << f.cost << "\n";
        return 0;
    }

    // fallback (should not happen for testId in 1..10)
    addUniformDemand(10, 0, 500, 0, 500, 1, 30);
    for (auto &d : dem) TotalLambda += d.lam;
    fac.push_back({250, 250, TotalLambda + 20, 100});
    cout << dem.size() << " " << fac.size() << " " << 100 << " " << 500 << "\n";
    for (auto &d : dem) cout << d.x << " " << d.y << " " << d.lam << "\n";
    for (auto &f : fac) cout << f.x << " " << f.y << " " << f.mu << " " << f.cost << "\n";
    return 0;
}
