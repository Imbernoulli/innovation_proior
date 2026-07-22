#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// chk.cc -- scorer for Cross-Dock Door Draw and Pallet Relay. Minimization.
//
// Validates: door draw is a permutation of 1..D; for every listed (i,j,f) pair the
// printed trip sizes lie in [1,K] and sum exactly to f.
//
// Objective for a door draw + batch plan:
//   for every trip crossing doors a,b (a!=b) it crosses every segment s with
//   min(a,b) <= s < max(a,b); L_s = total trips (over all pairs) crossing segment s;
//   F = sum_s L_s * base_t * (1 + (L_s/cap)^2)
// computed in O(trips + D) via a difference array (each trip does one +1/-1 update
// regardless of how many segments it spans; L_s recovered by a prefix sum).
//
// Baseline B: door draw = input order (inbound trucks get slots 1..T_in in listed
// order, outbound truck j gets slot T_in+j), every pallet moved as its own
// single-pallet trip. F evaluated on that construction (always positive, independent
// of the participant's output).
//
//   ratio = min(1, B / (10 * max(1,F)))

static double segCost(long long L, long long cap) {
    double r = (double)L / (double)cap;
    return 1.0 + r * r;
}

// Given a list of (a,b) door-pairs each carrying `cnt` trips, accumulate a difference
// array over segments 1..D-1 and return total F.
static double totalF(int D, const vector<array<long long,3>>& segTrips /* a,b,cnt */,
                      long long cap) {
    vector<long long> diff(D + 2, 0);
    for (auto& t : segTrips) {
        long long a = t[0], b = t[1], cnt = t[2];
        long long lo = min(a, b), hi = max(a, b); // segments lo..hi-1
        diff[lo] += cnt;
        diff[hi] -= cnt;
    }
    double F = 0.0;
    long long run = 0;
    for (int s = 1; s <= D - 1; s++) {
        run += diff[s];
        if (run > 0) F += (double)run * segCost(run, cap);
    }
    return F;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int Tin = inf.readInt();
    int Tout = inf.readInt();
    int M = inf.readInt();
    int K = inf.readInt();
    int cap = inf.readInt();
    int D = Tin + Tout;

    vector<int> pi_(M), pj_(M), pf_(M);
    for (int e = 0; e < M; e++) {
        pi_[e] = inf.readInt();
        pj_[e] = inf.readInt();
        pf_[e] = inf.readInt();
    }

    // ---- read participant door draw ----
    vector<int> door(D + 1, 0); // door[truckIdx] = slot, truckIdx 1..D
    vector<bool> usedSlot(D + 1, false);
    for (int k = 1; k <= D; k++) {
        int s = ouf.readInt(1, D, "door slot");
        if (usedSlot[s]) quitf(_wa, "slot %d assigned to more than one truck", s);
        usedSlot[s] = true;
        door[k] = s;
    }

    // ---- read participant batch plan, accumulate segment-load contributions ----
    vector<array<long long,3>> segTripsP; // (doorA, doorB, tripCount) per pair, cnt trips each crossing the same door pair
    segTripsP.reserve(M);
    for (int e = 0; e < M; e++) {
        int f = pf_[e];
        int m = ouf.readInt(1, f, "trip count");
        long long sum = 0;
        for (int t = 0; t < m; t++) {
            int sz = ouf.readInt(1, K, "trip size");
            sum += sz;
        }
        if (sum != f)
            quitf(_wa, "pair %d: trip sizes sum to %lld, expected %d", e + 1, sum, f);
        int truckI = pi_[e];               // inbound truck index 1..Tin
        int truckJ = Tin + pj_[e];         // outbound truck index Tin+1..D
        int a = door[truckI], b = door[truckJ];
        if (a == b) quitf(_wa, "pair %d: inbound and outbound truck share a door slot", e + 1);
        segTripsP.push_back({(long long)a, (long long)b, (long long)m});
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after the batch plan");

    double F = totalF(D, segTripsP, cap);

    // ---- checker's own baseline B: input-order block draw, unbatched (single-pallet) trips ----
    vector<array<long long,3>> segTripsB;
    segTripsB.reserve(M);
    for (int e = 0; e < M; e++) {
        int truckI = pi_[e];        // slot = truckI under identity draw (inbound k -> slot k)
        int truckJ = Tin + pj_[e];  // slot = Tin+j under identity draw
        int a = truckI, b = truckJ;
        segTripsB.push_back({(long long)a, (long long)b, (long long)pf_[e]}); // f single-pallet trips
    }
    double B = totalF(D, segTripsB, cap);
    if (B <= 0) B = 1.0;

    // canonical convention: sc = min(1000, 100*B/F) => ratio = sc/1000 = min(1, B/(10F))
    // (matching baseline exactly => ratio 0.1; reaching 1/10th of baseline cost or
    // better saturates the 1.0 cap, leaving headroom above it).
    double sc = min(1000.0, 100.0 * B / max(1.0, F));
    double ratio = sc / 1000.0;
    quitp(ratio, "OK F=%.3f B=%.3f Ratio: %.6f", F, B, ratio);
    return 0;
}
