#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// -----------------------------------------------------------------------------
// "The Flywheel Furnace: 300-Day Blend & Purge Planning"  (generator)
// family: contamination-flywheel-dilution
//
// Fixed plant economics (same units on every test, only market STRUCTURE and
// horizon length T change with testId -- see AGENT_BRIEF: exact numbers still
// live in the input, never assumed from the statement):
//   CAP        furnace capacity per day (tons)
//   RETURN_FRAC fraction of every ton melted that becomes internal return
//               scrap LAG days later, at the ppm of the blend that made it
//   LAG        days of delay before that return scrap joins the pool
//   PPM_CAP    hard contamination cap a melt must not exceed
//   P0, BETA   sale price P0*(1-BETA*(ppm/PPM_CAP)^2): running near the cap
//               taxes revenue smoothly, not just pass/fail
//   CV         virgin metal cost per ton (virgin is 0 ppm, always safe)
//   V          LIFETIME virgin budget for the whole run (scarce, shared
//               across all T days -- this is what "virgin-dilution-budget"
//               means: it is not replenished daily)
//
// Market REGIMES (each emits K lots for one day):
//   randsmall        tiny/simple, sanity scale (test 1)
//   random           general uniform mix
//   wave1            generous, moderately contaminated, cheap -- looks great
//                    taken in isolation; feeds the pool a slug of ~0.7*cap
//                    ppm material LAG days later (recycle-loop-accumulation)
//   wave2_needle     a rare, huge, very cheap but even-more-contaminated lot;
//                    only fully exploitable if the pool was kept clean before
//                    it appears (slow-state-shadow-pricing: the payoff of an
//                    earlier dilution choice is collected here, weeks later)
//   quiet            thin, unexciting market (connective tissue between waves)
//   densenearcap     constantly tempting, always-near-cap cheap lots (traps
//                    the nonlinear quality-discount, not just the pool)
//   periodicbase / periodicbonus  a bonus lot appears on a fixed cadence;
//                    only a plant with banked headroom benefits every time
//   hard             many lots, tight margins, larger K (stress test)
//
// TestId ladder: 1,2,7 sanity/general (no engineered pool contamination);
// 3,4,6 planted wave1->quiet->wave2_needle flywheel traps; 5 constant
// near-cap temptation feeding into a wave1/wave2 cycle; 8,9,10 run the full
// 300-day theme scale with several repeated flywheel cycles (8 dense-cap
// driven, 9 wave1 driven, 10 wave1 driven through a noisy "hard" market),
// filling the constraint envelope on the largest tests.
// -----------------------------------------------------------------------------

struct Lot { double avail, ppm, price; };

static double clampd(double v, double lo, double hi) { return max(lo, min(hi, v)); }

vector<Lot> regimeRandSmall() {
    int K = 2;
    vector<Lot> v;
    for (int i = 0; i < K; i++) {
        double ppm = rnd.next(100.0, 600.0);
        double avail = rnd.next(5.0, 20.0);
        double price = clampd(950.0 - 0.7 * ppm + rnd.next(-100.0, 100.0), 80.0, 1500.0);
        v.push_back({avail, ppm, price});
    }
    return v;
}

vector<Lot> regimeRandom() {
    int K = rnd.next(2, 4);
    vector<Lot> v;
    for (int i = 0; i < K; i++) {
        double ppm = rnd.next(80.0, 1350.0);
        double avail = rnd.next(8.0, 35.0);
        double price = clampd(1550.0 - 0.9 * ppm + rnd.next(-150.0, 150.0), 80.0, 1600.0);
        v.push_back({avail, ppm, price});
    }
    return v;
}

vector<Lot> regimeWave1() {
    int K = 3;
    vector<Lot> v;
    for (int i = 0; i < K; i++) {
        double ppm = rnd.next(650.0, 750.0);
        double avail = rnd.next(20.0, 35.0);
        double price = rnd.next(480.0, 620.0);
        v.push_back({avail, ppm, price});
    }
    return v;
}

vector<Lot> regimeWave2Needle() {
    int K = 2;
    vector<Lot> v;
    for (int i = 0; i < K; i++) {
        double ppm = rnd.next(920.0, 980.0);
        double avail = rnd.next(48.0, 62.0);
        double price = rnd.next(180.0, 280.0);
        v.push_back({avail, ppm, price});
    }
    return v;
}

vector<Lot> regimeQuiet() {
    int K = 2;
    vector<Lot> v;
    for (int i = 0; i < K; i++) {
        double ppm = rnd.next(200.0, 500.0);
        double avail = rnd.next(14.0, 26.0);
        double price = rnd.next(700.0, 950.0);
        v.push_back({avail, ppm, price});
    }
    return v;
}

vector<Lot> regimeDenseNearCap() {
    int K = 3;
    vector<Lot> v;
    for (int i = 0; i < K; i++) {
        double ppm = rnd.next(600.0, 950.0);
        double avail = rnd.next(15.0, 30.0);
        double price = rnd.next(400.0, 600.0);
        v.push_back({avail, ppm, price});
    }
    return v;
}

vector<Lot> regimePeriodicBase() {
    int K = rnd.next(1, 2);
    vector<Lot> v;
    for (int i = 0; i < K; i++) {
        double ppm = rnd.next(150.0, 450.0);
        double avail = rnd.next(5.0, 12.0);
        double price = rnd.next(750.0, 1000.0);
        v.push_back({avail, ppm, price});
    }
    return v;
}

vector<Lot> regimePeriodicBonus() {
    vector<Lot> v;
    double ppm = rnd.next(500.0, 650.0);
    double avail = rnd.next(30.0, 40.0);
    double price = rnd.next(250.0, 380.0);
    v.push_back({avail, ppm, price});
    return v;
}

vector<Lot> regimeHard() {
    int K = rnd.next(5, 8);
    vector<Lot> v;
    for (int i = 0; i < K; i++) {
        double ppm = rnd.next(100.0, 1300.0);
        double avail = rnd.next(5.0, 20.0);
        double price = clampd(1500.0 - 0.8 * ppm + rnd.next(-200.0, 200.0), 80.0, 1600.0);
        v.push_back({avail, ppm, price});
    }
    return v;
}

// day index is 1-based
vector<Lot> lotsForDay(int testId, int t) {
    switch (testId) {
        case 1:
            return regimeRandSmall();
        case 2:
            return regimeRandom();
        case 3:
            // Continuous temptation right up to the needle: nothing ever
            // forces a myopic scan to touch virgin (dirty cheap lots are
            // always the better flat-value pick), so the pool never gets a
            // chance to self-dilute -- only a deliberate purge lowers it.
            if (t <= 20) return regimeWave1();
            if (t <= 27) return regimeWave2Needle();
            return regimeQuiet();
        case 4:
            // Two full flywheel cycles: each is continuous temptation right
            // up to its own needle, with only a short breather between
            // cycles.
            if (t <= 15) return regimeWave1();
            if (t <= 22) return regimeWave2Needle();
            if (t <= 33) return regimeQuiet();
            if (t <= 48) return regimeWave1();
            if (t <= 55) return regimeWave2Needle();
            return regimeQuiet();
        case 5:
            // Constant near-cap temptation first (loads the pool AND taxes
            // the quality discount every day it's used), then a still-dirty
            // "wave1" stretch that keeps tempting a myopic scan while the
            // pool should be recovering, then the payoff: a big cheap
            // needle only a clean pool can fully absorb.
            if (t <= 18) return regimeDenseNearCap();
            if (t <= 30) return regimeWave1();
            if (t <= 37) return regimeWave2Needle();
            return regimeQuiet();
        case 6:
            // A dirtying phase now precedes the needle (previously this
            // test never dirtied the pool at all, so there was nothing to
            // protect and both tiers looked identical).
            if (t <= 15) return regimeWave1();
            if (t <= 25) return regimeQuiet();
            if (t <= 32) return regimeWave2Needle();
            return regimeQuiet();
        case 7:
            // General/textured market (not an engineered trap): a thin
            // periodic base with a recurring bonus lot on a fixed cadence.
            return (t % 10 == 0) ? regimePeriodicBonus() : regimePeriodicBase();
        case 8: {
            // Three flywheel cycles driven by constant near-cap temptation
            // (rather than wave1's moderate lots) -- a different
            // contamination source than test 9, same reward structure.
            int off = (t - 1) % 100;
            if (off < 20) return regimeDenseNearCap();
            if (off < 50) return regimeQuiet();
            if (off < 57) return regimeWave2Needle();
            return regimeQuiet();
        }
        case 9: {
            int off = (t - 1) % 100; // three 100-day cycles across T=300
            if (off < 15) return regimeWave1();
            if (off < 45) return regimeQuiet();
            if (off < 51) return regimeWave2Needle();
            return regimeQuiet();
        }
        case 10: {
            // Same reward structure again, plus a bounded noisy/tight-margin
            // "hard" stress interlude each cycle (more lots per day, tighter
            // margins) -- largest constraint envelope.
            int off = (t - 1) % 100;
            if (t % 15 == 0) return regimePeriodicBonus();
            if (off < 15) return regimeWave1();
            if (off < 45) return regimeQuiet();
            if (off < 51) return regimeWave2Needle();
            if (off < 80) return regimeQuiet();
            return regimeHard();
        }
        default:
            return regimeRandom();
    }
}

int Tfor(int testId) {
    static const int tbl[10] = {12, 25, 40, 90, 60, 70, 150, 300, 300, 300};
    return tbl[testId - 1];
}

double Vfracfor(int testId) {
    static const double tbl[10] = {0.35, 0.32, 0.26, 0.31, 0.28, 0.31, 0.31, 0.34, 0.38, 0.34};
    return tbl[testId - 1];
}

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    if (testId < 1) testId = 1;
    if (testId > 10) testId = 10;

    const double CAP = 100.0;
    const double RETURN_FRAC = 0.45;
    const int LAG = 6;
    const double PPM_CAP = 800.0;
    const double P0 = 2000.0;
    const double CV = 900.0;
    const double BETA = 0.6;

    int T = Tfor(testId);
    double V = Vfracfor(testId) * CAP * (double)T;

    printf("%d\n", T);
    printf("%.3f %.3f %.6f %d %.3f %.3f %.3f %.3f\n",
           CAP, V, RETURN_FRAC, LAG, PPM_CAP, P0, CV, BETA);

    for (int t = 1; t <= T; t++) {
        vector<Lot> lots = lotsForDay(testId, t);
        printf("%d\n", (int)lots.size());
        for (auto &l : lots) {
            double avail = clampd(l.avail, 1.0, 1000.0);
            double ppm = clampd(l.ppm, 0.0, 100000.0);
            double price = clampd(l.price, 0.0, 100000.0);
            printf("%.3f %.3f %.3f\n", avail, ppm, price);
        }
    }
    return 0;
}
