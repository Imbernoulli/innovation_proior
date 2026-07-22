// TIER: strong
#include <bits/stdc++.h>
using namespace std;

// Three genuine insights, composed. The plant sees the FULL T-day market up
// front (this is an offline planning problem), and a human planner reading
// that whole sheet would not treat every day the same way:
//
// (1) DUAL/RELAXATION VIEW OF THE DAILY BLEND. Filling the furnace to
//     maximize flat margin subject to a ppm ceiling is NOT "take the best
//     margin item, then the next, stopping whichever one first hits the
//     ceiling" (that is exactly the myopic tier's rule, and it is provably
//     wrong: when every high-margin item is individually too dirty to add
//     to an EMPTY mix, that rule adds none of them and dumps the entire day
//     onto whatever happens to be cleanest). The correct per-day fill is a
//     2-constraint fractional knapsack (mass <= CAP, weighted ppm <=
//     ceiling), solved via a Lagrangian shadow price lambda: rank sources by
//     (margin - lambda*(ppm-ceiling)), fill greedily, bisect lambda until
//     the ppm constraint is tight.
//
// (2) SHADOW-PRICE OVER TIME, GATED BY A LOOK-AHEAD. The furnace's true
//     state is the internal return-scrap pool's ppm -- a SLOW state only
//     today's blend can move, and only LAG days from now. Diluting below
//     what today alone requires never helps today's number, so it only pays
//     off if there is something ahead worth protecting: a big cheap lot
//     later in the (already fully visible) sheet that a dirty pool would
//     crowd out. Scan forward for the best single-lot value in the next
//     window of days; only pay the deliberate-dilution price when that
//     forward scan says it is worth it. Grinding through a market that is
//     ALWAYS moderately dirty with no such lot ahead gets nothing from
//     purging (the discount lost today isn't recouped by anything later),
//     so plain full-cap harvesting is correctly used there instead.
//
// (3) NEVER LEAVE FREE CAPACITY UNPRICED. Virgin is always non-negative
//     margin (it only dilutes, never raises ppm), so any leftover room is
//     topped off with it. The pool is different -- drawing more of it
//     raises the whole day's ppm and can cut the price on everything
//     already committed, so an extra draw is only taken when a small
//     profit-gated search shows it is a genuine improvement over idle,
//     never blindly.
struct Cand { double avail, ppm, cost; int type, idx; }; // type 0=return 1=lot 2=virgin

int main() {
    int T;
    cin >> T;
    double CAP, V, RF, PPM_CAP, P0, CV, BETA;
    int LAG;
    cin >> CAP >> V >> RF >> LAG >> PPM_CAP >> P0 >> CV >> BETA;

    vector<int> K(T + 1);
    vector<vector<double>> AV(T + 1), PP(T + 1), PR(T + 1);
    for (int t = 1; t <= T; t++) {
        cin >> K[t];
        AV[t].resize(K[t]);
        PP[t].resize(K[t]);
        PR[t].resize(K[t]);
        for (int j = 0; j < K[t]; j++) cin >> AV[t][j] >> PP[t][j] >> PR[t][j];
    }

    const double SAFE_FRAC = 0.45;
    const double PURGE_TARGET_FRAC = 0.20;
    const int LOOKAHEAD_WINDOW = 55;

    // Forward scan: bestLotValue[t] = the single most valuable lot offered
    // on day t, valued at full price (avail * (P0 - price)), ignoring ppm --
    // a cheap proxy for "how good is this day's best opportunity". A window
    // max of this ahead of day t tells us whether protecting pool headroom
    // for the near future is worth anything at all.
    vector<double> bestLotValue(T + 2, 0.0);
    for (int t = 1; t <= T; t++) {
        double best = 0.0;
        for (int j = 0; j < K[t]; j++) {
            double v = AV[t][j] * max(0.0, P0 - PR[t][j]);
            best = max(best, v);
        }
        bestLotValue[t] = best;
    }
    // A day's own opportunity is never a reason to dilute FOR that same day,
    // so windowMax looks strictly ahead of t.
    vector<double> windowMax(T + 2, 0.0);
    {
        deque<int> dq; // indices with decreasing bestLotValue
        // build windowMax[t] = max(bestLotValue[t+1 .. min(T,t+LOOKAHEAD_WINDOW)])
        for (int t = T; t >= 1; t--) {
            int hi = min(T, t + LOOKAHEAD_WINDOW);
            // simple O(window) scan is fine: T<=300, window<=55
            double m = 0.0;
            for (int s = t + 1; s <= hi; s++) m = max(m, bestLotValue[s]);
            windowMax[t] = m;
        }
    }
    // Calibrate the "worth protecting for" threshold from the market itself
    // (never a hardcoded number pattern-matched to one test): a lot whose
    // value clears the market's own 75th-percentile day is a real prize.
    double OPP_THRESHOLD;
    {
        vector<double> v(bestLotValue.begin() + 1, bestLotValue.begin() + 1 + T);
        sort(v.begin(), v.end());
        int idx = (int)(0.75 * (v.size() - 1));
        OPP_THRESHOLD = max(1.0, 2.0 * v[idx]);
    }

    vector<double> histMass(T + 2, 0.0), histPpm(T + 2, 0.0);
    double remV = V;
    double poolMass = 0.0, poolPpm = 0.0;

    // Solve: maximize sum(value_i * x_i)
    //        s.t. sum(x_i) <= capRoom,  sum((ppm_i-ceiling)*x_i) <= rhs0,
    //             0 <= x_i <= avail_i.
    // via bisection on the shadow price lambda >= 0 of the ppm constraint.
    auto solveFillLP = [&](vector<Cand> &pool, double capRoom, double ceiling, double rhs0,
                            vector<double> &xOut) {
        int n = (int)pool.size();
        xOut.assign(n, 0.0);
        if (capRoom <= 1e-12 || n == 0) return;
        auto knapsack = [&](double lambda, vector<double> &x, double &mass, double &excess) {
            vector<int> order(n);
            for (int i = 0; i < n; i++) order[i] = i;
            sort(order.begin(), order.end(), [&](int a, int b) {
                double va = (P0 - pool[a].cost) - lambda * (pool[a].ppm - ceiling);
                double vb = (P0 - pool[b].cost) - lambda * (pool[b].ppm - ceiling);
                if (fabs(va - vb) > 1e-9) return va > vb;
                if (pool[a].ppm != pool[b].ppm) return pool[a].ppm < pool[b].ppm;
                return a < b;
            });
            x.assign(n, 0.0);
            mass = 0.0;
            excess = 0.0;
            for (int idx : order) {
                double remCap = capRoom - mass;
                if (remCap <= 1e-12) break;
                double adj = (P0 - pool[idx].cost) - lambda * (pool[idx].ppm - ceiling);
                if (adj < 0) break; // not worth including at this shadow price;
                                     // sorted descending so nothing later helps either
                double add = min(pool[idx].avail, remCap);
                if (add < 0) add = 0;
                x[idx] = add;
                mass += add;
                excess += add * (pool[idx].ppm - ceiling);
            }
        };
        double mass0, excess0;
        vector<double> x0;
        knapsack(0.0, x0, mass0, excess0);
        if (excess0 <= rhs0 + 1e-6) {
            xOut = x0;
            return;
        }
        double lo = 0.0, hi = 80.0;
        // Safe fallback if bisection never lands on a feasible lambda (the
        // greedy knapsack's excess(lambda) need not be perfectly monotonic
        // across the discrete item-order swaps): the all-zero allocation is
        // ALWAYS feasible here (rhs0 >= 0 whenever this function is called),
        // so never fall back to the known-infeasible lambda=0 solution.
        vector<double> best(n, 0.0);
        for (int it = 0; it < 60; it++) {
            double mid = 0.5 * (lo + hi);
            vector<double> x;
            double mass, excess;
            knapsack(mid, x, mass, excess);
            if (excess > rhs0 + 1e-6) {
                lo = mid;
            } else {
                hi = mid;
                best = x;
            }
        }
        xOut = best;
    };

    cout << setprecision(9) << fixed;
    for (int t = 1; t <= T; t++) {
        int k = K[t];

        if (t - LAG >= 1) {
            double massIn = RF * histMass[t - LAG];
            if (massIn > 1e-12) {
                double ppmIn = histPpm[t - LAG];
                double newMass = poolMass + massIn;
                poolPpm = (poolMass * poolPpm + massIn * ppmIn) / newMass;
                poolMass = newMass;
            }
        }

        bool dirty = poolPpm > SAFE_FRAC * PPM_CAP;
        bool worthProtecting = windowMax[t] >= OPP_THRESHOLD;
        bool purge = dirty && worthProtecting;
        double ceiling1 = purge ? (PURGE_TARGET_FRAC * PPM_CAP) : PPM_CAP;

        // Pacing guard (PURGE ONLY): never let a single purge day spend more
        // than a modest multiple of the SUSTAINABLE remaining-budget rate.
        double virginRoom = min(CAP, remV);
        if (purge) {
            int remainingDays = T - t + 1;
            const double PACE_FACTOR = 2.5;
            double pacedVirgin = (remV / (double)remainingDays) * PACE_FACTOR;
            virginRoom = min(virginRoom, pacedVirgin);
        }
        vector<Cand> cand;
        if (poolMass > 1e-9) cand.push_back({poolMass, poolPpm, 0.0, 0, -1});
        for (int j = 0; j < k; j++) cand.push_back({AV[t][j], PP[t][j], PR[t][j], 1, j});
        if (virginRoom > 1e-9) cand.push_back({virginRoom, 0.0, CV, 2, -1});

        vector<double> x1;
        solveFillLP(cand, CAP, ceiling1, 0.0, x1);

        double virginUsed = 0.0, returnUsed = 0.0;
        vector<double> xUsed(k, 0.0);
        double curMass = 0.0, curPpmSum = 0.0;
        for (int i = 0; i < (int)cand.size(); i++) {
            double add = x1[i];
            cand[i].avail -= add;
            curMass += add;
            curPpmSum += add * cand[i].ppm;
            if (cand[i].type == 0) returnUsed += add;
            else if (cand[i].type == 1) xUsed[cand[i].idx] += add;
            else virginUsed += add;
        }

        // Pass 2 (purge only): top up any remaining room against the real
        // hard cap using leftover lots/virgin -- the dirty pool does not
        // get a second bite here, to protect the deliberate dilution.
        if (purge && CAP - curMass > 1e-9) {
            vector<Cand> cand2;
            for (auto &c : cand)
                if (c.type != 0 && c.avail > 1e-12) cand2.push_back(c);
            double rhs0 = PPM_CAP * curMass - curPpmSum;
            vector<double> x2;
            solveFillLP(cand2, CAP - curMass, PPM_CAP, rhs0, x2);
            for (int i = 0; i < (int)cand2.size(); i++) {
                double add = x2[i];
                curMass += add;
                curPpmSum += add * cand2[i].ppm;
                if (cand2[i].type == 1) xUsed[cand2[i].idx] += add;
                else virginUsed += add;
            }
        }

        // Pass 2.5 (any mode, profit-GATED): unlike virgin, drawing MORE of
        // a dirty pool raises the ppm -- and hence cuts the sale price -- on
        // the WHOLE day's blend, not just the marginal ton, so it is NOT
        // automatically non-negative-margin the way virgin is. But leaving
        // capacity fully idle is ALSO a real cost. Grid-search how much
        // extra pool (if any) to draw against the remaining room, scoring
        // each candidate amount by actual realized profit (including its
        // effect on the price of everything already committed today), and
        // only take an amount that is a genuine improvement over idle.
        if (CAP - curMass > 1e-9 && poolMass - returnUsed > 1e-9) {
            double poolLeft = poolMass - returnUsed;
            double remCap = CAP - curMass;
            double maxAdd;
            if (poolPpm <= PPM_CAP + 1e-9) {
                maxAdd = min(poolLeft, remCap);
            } else {
                double RHS = PPM_CAP * curMass - curPpmSum;
                double denom = poolPpm - PPM_CAP;
                maxAdd = (RHS <= 0) ? 0.0 : min({poolLeft, remCap, RHS / denom});
            }
            double bestAdd = 0.0, bestProfit = -1e18;
            const int STEPS = 20;
            for (int s = 0; s <= STEPS; s++) {
                double add = maxAdd * (double)s / (double)STEPS;
                double m = curMass + add, ps = curPpmSum + add * poolPpm;
                double ppm = (m > 1e-9) ? ps / m : 0.0;
                double r = ppm / PPM_CAP;
                double quality = max(0.0, 1.0 - BETA * r * r);
                double revenue = P0 * quality * (1.0 - RF) * m; // pool itself is free
                if (revenue > bestProfit) { bestProfit = revenue; bestAdd = add; }
            }
            if (bestAdd > 1e-9) {
                curMass += bestAdd;
                curPpmSum += bestAdd * poolPpm;
                returnUsed += bestAdd;
            }
        }

        // Pass 3 (last resort, any mode): idle capacity is STRICTLY worse
        // than spending extra virgin -- it is 0 ppm (can only dilute, never
        // hurts the blend) and P0 > CV always, so any additional virgin ton
        // is non-negative-margin. The pacing guard exists to
        // protect FUTURE days from a budget run dry; it must never cause
        // THIS day to sit idle when the budget still has room, since a
        // missed day's capacity can never be recovered later.
        if (CAP - curMass > 1e-9 && remV - virginUsed > 1e-9) {
            double add = min(CAP - curMass, remV - virginUsed);
            if (add > 0) {
                curMass += add;
                // virgin is 0 ppm: curPpmSum unaffected
                virginUsed += add;
            }
        }

        remV -= virginUsed;
        poolMass -= returnUsed;
        if (poolMass < 0) poolMass = 0.0;

        double ppmToday = (curMass > 1e-9) ? (curPpmSum / curMass) : 0.0;
        histMass[t] = curMass;
        histPpm[t] = ppmToday;

        cout << virginUsed;
        for (int j = 0; j < k; j++) cout << ' ' << xUsed[j];
        cout << ' ' << returnUsed << '\n';
    }
    return 0;
}
