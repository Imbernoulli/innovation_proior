#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Mining the Cannibal River"  (generator)  family: cannibal-cascade-heat-mining
//
// Emits: T K Tsink
//        then T lines: q_i theta_i eta_i cap_i
//
// A single stream flows through segments 1..T; at each segment a tributary
// (q_i, theta_i) mixes in (flow-weighted average). eta_i is the local unit
// quality (used only if a converter is installed there); cap_i is the local
// unit's hardware extraction capacity (prevents a single downstream unit,
// which naturally accumulates the largest flow, from trivially hoovering up
// the whole river -- forces genuine multi-unit trade-offs).
//
// TestId 1-3:  sanity / small random / cleanly PLANTED well-separated good spots.
// TestId 4:    DILUTION TRAP -- several co-equal hot spikes; fully draining the
//              first (as a naive "hottest first, drain hard" greedy does) dumps
//              a large mass of Tsink-cold water into the mix, suppressing every
//              later spike's actual mixed temperature far below its natural one.
// TestId 5:    ETA-DECOY TRAP -- the single hottest natural spot has terrible
//              eta (bad hardware); several cooler spots have excellent eta and
//              are the true prize. A temperature-only greedy wastes a slot there.
// TestId 6-7:  combined dilution+decoy trap, and a NEEDLE (one huge tributary
//              in an otherwise flat lukewarm river, with a handful of smaller
//              follow-on spots that only pay off if the needle is NOT fully
//              drained).
// TestId 8-10: same trap families at LARGE scale (T up to 4000), filling the
//              stated size envelope.
// -----------------------------------------------------------------------------

static int T, K, Tsink;
static vector<ll> q, theta, eta, cap;

static void resize(int n) {
    T = n;
    q.assign(T + 1, 0); theta.assign(T + 1, 0); eta.assign(T + 1, 0); cap.assign(T + 1, 0);
}

// fill every segment with random "filler" (low value, low flow) tributaries
static void fillFiller(ll qlo, ll qhi, ll tlo, ll thi, ll elo, ll ehi, ll fillCapMult) {
    for (int i = 1; i <= T; i++) {
        q[i]   = rnd.next(qlo, qhi);
        theta[i] = Tsink + rnd.next(tlo, thi);
        eta[i] = rnd.next(elo, ehi);
        cap[i] = q[i] * fillCapMult;
    }
}

static void plant(int p, ll qq, ll tt, ll ee, ll capMult) {
    q[p] = qq;
    theta[p] = Tsink + tt;
    eta[p] = ee;
    cap[p] = qq * capMult;
}

static void emit() {
    printf("%d %d %d\n", T, K, Tsink);
    for (int i = 1; i <= T; i++) {
        printf("%lld %lld %lld %lld\n", q[i], theta[i], eta[i], cap[i]);
    }
}

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    Tsink = 300;
    const ll CAPMULT = 8;

    if (testId == 1) {
        // tiny sanity case
        resize(6); K = 2;
        fillFiller(50, 150, 50, 400, 600, 1000, CAPMULT);
        for (int i = 1; i <= T; i++) cap[i] = q[i] * CAPMULT;
    } else if (testId == 2) {
        // small, purely random uniform -- no special traps
        resize(25); K = 4;
        fillFiller(30, 300, 20, 600, 400, 1000, CAPMULT);
        for (int i = 1; i <= T; i++) cap[i] = q[i] * CAPMULT;
    } else if (testId == 3) {
        // PLANTED: 5 well-separated clean good spots amid cool low-flow filler
        resize(60); K = 5;
        fillFiller(5, 15, 10, 60, 500, 700, 2);
        int spots[5] = {6, 18, 30, 42, 54};
        for (int p : spots) plant(p, rnd.next(200, 300), rnd.next(600, 850), rnd.next(850, 1000), CAPMULT);
    } else if (testId == 4) {
        // DILUTION TRAP: 4 co-equal hot spikes; draining #1 hard dilutes #2..#4
        resize(90); K = 4;
        fillFiller(3, 10, 10, 40, 500, 650, 2);
        int spots[4] = {1, 29, 58, 87};
        for (int p : spots) plant(p, rnd.next(280, 320), rnd.next(650, 750), rnd.next(850, 950), CAPMULT);
    } else if (testId == 5) {
        // ETA-DECOY TRAP: global hottest spot has terrible eta; real winners are cooler
        resize(90); K = 5;
        fillFiller(3, 10, 10, 40, 500, 650, 2);
        plant(1, 350, 890, 120, CAPMULT);
        int spots[5] = {18, 36, 54, 72, 89};
        for (int p : spots) plant(p, rnd.next(150, 220), rnd.next(420, 520), rnd.next(920, 1000), CAPMULT);
    } else if (testId == 6) {
        // COMBINED dilution + eta-decoy, larger
        resize(170); K = 6;
        fillFiller(3, 10, 10, 40, 500, 650, 2);
        plant(1, 350, 890, 150, CAPMULT);
        int spots[6] = {28, 56, 84, 112, 140, 168};
        for (int p : spots) plant(p, rnd.next(180, 260), rnd.next(500, 650), rnd.next(880, 1000), CAPMULT);
    } else if (testId == 7) {
        // NEEDLE: flat lukewarm river + one huge tributary near start + follow-ons
        resize(170); K = 5;
        fillFiller(5, 15, 5, 25, 500, 700, 2);
        plant(3, 450, 880, 980, CAPMULT);
        int spots[4] = {45, 90, 135, 165};
        for (int p : spots) plant(p, rnd.next(150, 220), rnd.next(400, 550), rnd.next(850, 970), CAPMULT);
    } else if (testId == 8) {
        // LARGE dilution-trap scale-up
        resize(1200); K = 7;
        fillFiller(3, 10, 10, 40, 500, 650, 2);
        int spots[7];
        for (int k = 0; k < 7; k++) spots[k] = max(1, min(T, (int)((ll)T * (k + 1) / 8)));
        spots[0] = 1;
        for (int p : spots) plant(p, rnd.next(280, 340), rnd.next(650, 780), rnd.next(850, 970), CAPMULT);
    } else if (testId == 9) {
        // LARGE dense: many comparable good tributaries -> real multi-unit
        // marginal-value equalization at scale, minimal position-selection trickery
        resize(2000); K = 8;
        fillFiller(60, 120, 150, 400, 700, 950, CAPMULT);
        for (int i = 1; i <= T; i++) cap[i] = q[i] * CAPMULT;
    } else {
        // LARGEST: combined decoy + spike ladder, fills the size envelope
        resize(4000); K = 8;
        fillFiller(3, 10, 10, 40, 500, 650, 2);
        plant(1, 480, 895, 100, CAPMULT);
        double fracs[7] = {0.06, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95};
        for (double f : fracs) {
            int p = max(2, min(T, (int)(T * f)));
            plant(p, rnd.next(280, 340), rnd.next(600, 750), rnd.next(880, 990), CAPMULT);
        }
    }

    emit();
    return 0;
}
