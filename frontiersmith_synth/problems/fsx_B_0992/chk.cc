#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// -----------------------------------------------------------------------------
// Checker / scorer for "The Flywheel Furnace: 300-Day Blend & Purge Planning"
// (family: contamination-flywheel-dilution).
//
// Input:
//   T
//   CAP V RETURN_FRAC LAG PPM_CAP P0 CV BETA
//   then for t = 1..T:  K_t
//                        K_t lines: avail ppm price
//
// Output (participant), for t = 1..T in order:
//   virgin_t  x_{t,1} .. x_{t,K_t}  return_t         (all reals >= 0)
//
// Internal state (well-mixed return-scrap pool):
//   day t's inflow (only if t > LAG) = RETURN_FRAC * mass_{t-LAG} tons, at
//   ppm = ppm_{t-LAG} (the blend achieved that day). It merges into the pool
//   as a mass-weighted average BEFORE the participant draws from the pool
//   on day t; whatever is drawn on day t comes out at that day's pool ppm.
//
// Feasibility (hard, hit any -> the whole submission scores 0):
//   0 <= virgin_t, cumulative virgin_t <= V
//   0 <= x_{t,j} <= avail_{t,j}
//   0 <= return_t <= pool mass available on day t (after inflow)
//   virgin_t + sum_j x_{t,j} + return_t <= CAP
//   blended ppm of day t's melt <= PPM_CAP
//
// Objective (MAXIMIZE), summed over all T days:
//   quality(ppm) = 1 - BETA*(ppm/PPM_CAP)^2
//   sold_mass    = (1-RETURN_FRAC) * mass_t      (the rest recycles internally)
//   revenue      = P0 * quality(ppm_t) * sold_mass
//   cost         = CV*virgin_t + sum_j price_{t,j}*x_{t,j}     (return scrap free)
//   F           += revenue - cost
//
// Baseline B (checker-computed do-nothing-clever construction): melt ONLY
// virgin metal, every day, up to min(daily CAP, remaining lifetime budget V);
// never touch scrap or the pool. This is exactly what solutions/trivial.cpp
// reproduces -> ratio 0.1.
//
// Score (max):  sc = min(1000, 100*max(F,0)/B);  ratio = sc/1000.
// -----------------------------------------------------------------------------

static double readNumOuf(double lo, double hi, const string &name) {
    double v = ouf.readDouble(lo, hi, name);
    if (!isfinite(v)) quitf(_wa, "%s is not finite", name.c_str());
    return v;
}

int main(int argc, char *argv[]) {
    registerTestlibCmd(argc, argv);

    int T = inf.readInt(1, 100000, "T");
    double CAP = inf.readDouble(1e-6, 1e9, "CAP");
    double V = inf.readDouble(0.0, 1e12, "V");
    double RETURN_FRAC = inf.readDouble(0.0, 0.999999, "RETURN_FRAC");
    int LAG = inf.readInt(1, 100000, "LAG");
    double PPM_CAP = inf.readDouble(1e-6, 1e9, "PPM_CAP");
    double P0 = inf.readDouble(0.0, 1e9, "P0");
    double CV = inf.readDouble(0.0, 1e9, "CV");
    double BETA = inf.readDouble(0.0, 1.0, "BETA");

    vector<int> K(T + 1, 0);
    vector<vector<double>> avail(T + 1), ppm(T + 1), price(T + 1);
    for (int t = 1; t <= T; t++) {
        int k = inf.readInt(0, 200, "K_t");
        K[t] = k;
        avail[t].resize(k);
        ppm[t].resize(k);
        price[t].resize(k);
        for (int j = 0; j < k; j++) {
            avail[t][j] = inf.readDouble(1e-9, 1e6, "avail");
            ppm[t][j] = inf.readDouble(0.0, 1e7, "ppm");
            price[t][j] = inf.readDouble(0.0, 1e7, "price");
        }
    }

    // ---- internal baseline B: virgin-only, every day, over the whole horizon ----
    double totalCapMass = CAP * (double)T;
    double massB = min(totalCapMass, V);
    double revB = P0 * (1.0 - RETURN_FRAC) * massB;
    double costB = CV * massB;
    double B = revB - costB;
    if (B < 1e-6) B = 1e-6;

    // ---- simulate the participant's plan ----
    double remainingV = V;
    double poolMass = 0.0, poolPpm = 0.0;
    vector<double> histMass(T + 2, 0.0), histPpm(T + 2, 0.0);
    double F = 0.0;

    for (int t = 1; t <= T; t++) {
        // pool inflow from day (t-LAG)'s output, merged BEFORE today's draw
        if (t - LAG >= 1) {
            double massIn = RETURN_FRAC * histMass[t - LAG];
            if (massIn > 1e-12) {
                double ppmIn = histPpm[t - LAG];
                double newMass = poolMass + massIn;
                poolPpm = (poolMass * poolPpm + massIn * ppmIn) / newMass;
                poolMass = newMass;
            }
        }

        double virginCapToday = min(CAP, remainingV);
        if (virginCapToday < 0) virginCapToday = 0;
        double virgin_t = readNumOuf(0.0, virginCapToday + 1e-6, format("virgin_%d", t));

        int k = K[t];
        double sumX = 0.0, sumXppm = 0.0, sumCost = 0.0;
        for (int j = 0; j < k; j++) {
            double xj = readNumOuf(0.0, avail[t][j] + 1e-6, format("x_%d_%d", t, j));
            sumX += xj;
            sumXppm += xj * ppm[t][j];
            sumCost += xj * price[t][j];
        }

        double return_t = readNumOuf(0.0, poolMass + 1e-6, format("return_%d", t));

        remainingV -= virgin_t;
        if (remainingV < 0) remainingV = 0;

        double mass_t = virgin_t + sumX + return_t;
        if (mass_t > CAP + 1e-6)
            quitf(_wa, "furnace capacity exceeded on day %d: melted %.6f > CAP=%.6f", t, mass_t, CAP);

        double ppm_t = 0.0;
        if (mass_t > 1e-9) ppm_t = (sumXppm + return_t * poolPpm) / mass_t;
        if (ppm_t > PPM_CAP + 1e-6)
            quitf(_wa, "ppm cap exceeded on day %d: blend=%.6f > PPM_CAP=%.6f", t, ppm_t, PPM_CAP);

        poolMass -= return_t;
        if (poolMass < 0) poolMass = 0.0;

        histMass[t] = mass_t;
        histPpm[t] = ppm_t;

        double soldMass = (1.0 - RETURN_FRAC) * mass_t;
        double r = ppm_t / PPM_CAP;
        double quality = 1.0 - BETA * r * r;
        if (quality < 0) quality = 0;
        double revenue = P0 * quality * soldMass;
        double cost = CV * virgin_t + sumCost;
        F += revenue - cost;
    }

    if (!ouf.seekEof()) quitf(_wa, "trailing output after day %d", T);

    double Fclip = max(F, 0.0);
    double sc = min(1000.0, 100.0 * Fclip / B);
    quitp(sc / 1000.0, "OK F=%.3f B=%.3f Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
