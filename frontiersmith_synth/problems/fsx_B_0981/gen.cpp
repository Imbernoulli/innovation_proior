#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Generator for "Bessel Sideband Inversion". testId is a difficulty/trap ladder.
// Every trap mode ENUMERATES every reachable (slot,ratio,order) combination for
// this test's windows/H and picks uniformly among them -- never a blind draw
// that can miss -- so the intended trap always genuinely engages.
//   1      tiny sanity (planted target, small K/H)
//   2      generic random-noise target
//   3,8    NEEDLE traps: one harmonic needs an amplitude only reachable near a
//          Bessel local maximum of a HIGHER sideband order (2 or 3) -> a linear
//          index-vs-amplitude guess calibrated on order-1 behaviour misses it
//   4,9    CROSS-CONSTRUCT traps: a target harmonic is reachable by TWO DIFFERENT
//          operators (different ratios, different orders) whose contributions add
//          past what either operator alone can reach -- greedy assigns only ONE
//          operator per target harmonic and can never route a second one in
//   5      CROSS-CANCEL trap: an operator's own order-1 sideband (loud target)
//          and order-2 sideband (near-silent target) sit exactly r apart, so
//          greedy's ratio=h-rc rule for the loud target inevitably spills into
//          the quiet one from that SAME operator
//   6      bigger planted target (K=4)
//   7      high-cost-pressure trap (large lambda: raw amplitude-chasing is punished)
//   10     stress case combining several traps at the largest scale

static const int NMAX = 10;

static void besselRow(double x, vector<double> &out) {
    out.assign(2 * NMAX + 1, 0.0);
    for (int n = 0; n <= NMAX; n++) {
        double v = std::cyl_bessel_j((double)n, x);
        out[NMAX + n] = v;
        out[NMAX - n] = (n % 2 == 0) ? v : -v;
    }
}

static void enumerate(int pos, int K, long long hsum, double prod,
                       const vector<vector<double>> &J, const vector<int> &ratio,
                       int H, vector<double> &bin) {
    if (pos == K) {
        long long ah = llabs(hsum);
        if (ah <= H) bin[(int)ah] += prod;
        return;
    }
    for (int n = -NMAX; n <= NMAX; n++) {
        double c = J[pos][NMAX + n];
        if (c == 0.0) continue;
        enumerate(pos + 1, K, hsum + (long long)n * ratio[pos], prod * c, J, ratio, H, bin);
    }
}

static vector<double> synthesize(int K, long long rc, int H, const vector<int> &ratio,
                                  const vector<double> &idx) {
    vector<vector<double>> J(K);
    for (int i = 0; i < K; i++) besselRow(idx[i], J[i]);
    vector<double> bin(H + 1, 0.0);
    enumerate(0, K, rc, 1.0, J, ratio, H, bin);
    vector<double> A(H + 1, 0.0);
    for (int h = 1; h <= H; h++) A[h] = fabs(bin[h]);
    return A;
}

int main(int argc, char *argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int K, H, rc;
    double lambda;
    vector<int> Rlo, Rhi;
    vector<double> Cmax;
    string mode;

    auto mkWindow = [&](int lo, int hi, double cap) {
        Rlo.push_back(lo); Rhi.push_back(hi); Cmax.push_back(cap);
    };

    if (testId == 1) {
        K = 2; H = 6; rc = 1; lambda = 0.05; mode = "planted";
        mkWindow(1, 3, 1.8); mkWindow(2, 5, 1.8);
    } else if (testId == 2) {
        K = 2; H = 8; rc = 2; lambda = 0.05; mode = "noise";
        mkWindow(1, 4, 2.5); mkWindow(3, 7, 2.0);
    } else if (testId == 3) {
        K = 2; H = 10; rc = 1; lambda = 0.05; mode = "needle";
        mkWindow(1, 5, 3.0); mkWindow(1, 5, 3.0);
    } else if (testId == 4) {
        K = 3; H = 12; rc = 2; lambda = 0.08; mode = "cross_construct";
        mkWindow(1, 6, 3.5); mkWindow(1, 6, 3.5); mkWindow(2, 8, 3.0);
    } else if (testId == 5) {
        K = 3; H = 14; rc = 1; lambda = 0.08; mode = "cross_cancel";
        mkWindow(1, 6, 3.5); mkWindow(1, 6, 3.5); mkWindow(2, 8, 3.0);
    } else if (testId == 6) {
        K = 4; H = 16; rc = 2; lambda = 0.05; mode = "planted";
        mkWindow(1, 5, 3.0); mkWindow(2, 6, 3.0); mkWindow(1, 7, 3.0); mkWindow(3, 9, 3.0);
    } else if (testId == 7) {
        K = 4; H = 18; rc = 3; lambda = 0.35; mode = "highcost";
        mkWindow(1, 6, 4.0); mkWindow(2, 7, 4.0); mkWindow(1, 8, 4.0); mkWindow(3, 9, 4.0);
    } else if (testId == 8) {
        K = 4; H = 20; rc = 2; lambda = 0.10; mode = "needle";
        mkWindow(1, 7, 3.5); mkWindow(2, 8, 3.5); mkWindow(1, 9, 3.5); mkWindow(3, 10, 3.5);
    } else if (testId == 9) {
        K = 5; H = 24; rc = 3; lambda = 0.10; mode = "cross_construct";
        mkWindow(1, 8, 4.0); mkWindow(1, 8, 4.0); mkWindow(2, 9, 4.0);
        mkWindow(2, 10, 4.0); mkWindow(3, 11, 3.5);
    } else {
        K = 5; H = 28; rc = 4; lambda = 0.15; mode = "stress";
        mkWindow(1, 10, 6.0); mkWindow(2, 12, 6.0); mkWindow(1, 14, 5.5);
        mkWindow(3, 16, 5.0); mkWindow(4, 20, 4.5);
    }

    vector<double> T(H + 1, 0.0);

    if (mode == "planted" || mode == "stress") {
        // hidden config: random ratios/indices inside each slot's window
        vector<int> hr(K); vector<double> hi(K);
        for (int i = 0; i < K; i++) {
            hr[i] = rnd.next(Rlo[i], Rhi[i]);
            hi[i] = rnd.next(0.2, Cmax[i]);
        }
        vector<double> A = synthesize(K, rc, H, hr, hi);
        for (int h = 1; h <= H; h++) T[h] = min(3.0, A[h]);
        if (mode == "stress") {
            for (int h = 1; h <= H; h++)
                T[h] = max(0.0, min(3.0, T[h] + rnd.next(-0.05, 0.05)));
        }
    } else if (mode == "noise") {
        for (int h = 1; h <= H; h++)
            T[h] = (rnd.next(0, 99) < 55) ? rnd.next(0.05, 0.9) : 0.0;
    } else if (mode == "needle") {
        for (int h = 1; h <= H; h++) T[h] = rnd.next(0.0, 0.06);
        // Place the needle on a HIGHER sideband order (2 or 3): |J_2|,|J_3| peak
        // at larger I (~3.05, ~4.20) than |J_1| (~1.84), so a "index ~ amplitude"
        // linear guess calibrated on small-amplitude/low-order behaviour badly
        // undershoots the true index this needle needs. Enumerate every reachable
        // (slot,order,ratio) exhaustively (not a blind random draw) so a needle is
        // ALWAYS placed, and fold the harmonic index with abs() exactly like the
        // checker does (rc-order*r can be negative; |rc-order*r| still lands a
        // real, checker-scored sideband).
        vector<array<long long,4>> cand; // slot, order, r, h
        for (int slot = 0; slot < K; slot++)
            for (int order = 2; order <= 3; order++)
                for (int r = Rlo[slot]; r <= Rhi[slot]; r++) {
                    long long h = rc + (long long)order * r;
                    if (h >= 1 && h <= H) cand.push_back({slot, order, r, h});
                }
        if (!cand.empty()) {
            auto c = cand[rnd.next(0, (int)cand.size() - 1)];
            int slot = (int)c[0], order = (int)c[1], r = (int)c[2];
            long long h = c[3];
            // true peak of |J_order| WITHIN this slot's own cap (numeric scan, so
            // the needle is always genuinely reachable by the intended order).
            double bestVal = 0.0;
            for (int s = 0; s <= 4000; s++) {
                double I = Cmax[slot] * (double)s / 4000.0;
                double v = fabs(std::cyl_bessel_j((double)order, I));
                if (v > bestVal) bestVal = v;
            }
            T[h] = min(3.0, bestVal * rnd.next(0.85, 0.98));
            long long h2 = llabs(rc - (long long)order * r);   // folded mirror sideband
            if (h2 >= 1 && h2 <= H && h2 != h)
                T[h2] = min(3.0, bestVal * rnd.next(0.85, 0.98));
        }
    } else if (mode == "cross_cancel") {
        for (int h = 1; h <= H; h++) T[h] = rnd.next(0.0, 0.05);
        // A loud target at hLo = rc + r (this operator's OWN order-1 sideband)
        // forces a healthy index onto ratio r; that SAME operator's order-2
        // sideband then inevitably lands at hMid = rc + 2r (adjacent sideband
        // orders of one operator are always exactly r apart), which the target
        // wants near-silent. Greedy's per-harmonic rule always sets ratio=h-rc
        // (order 1) to chase a loud target, so it walks straight into this
        // coupling; an insightful solver can route around it (e.g. reach hLo via
        // a DIFFERENT operator/order instead of ratio r's own order 1).
        // Enumerate every reachable (slot,r) exhaustively so the trap always
        // engages, regardless of the random draw.
        vector<pair<int,int>> cand; // slot, r
        for (int slot = 0; slot < K; slot++)
            for (int r = Rlo[slot]; r <= Rhi[slot]; r++) {
                long long hLo = rc + (long long)r, hMid = rc + 2LL * r;
                if (hLo >= 1 && hLo <= H && hMid >= 1 && hMid <= H && hLo != hMid)
                    cand.push_back({slot, r});
            }
        if (!cand.empty()) {
            auto pr = cand[rnd.next(0, (int)cand.size() - 1)];
            int r = pr.second;
            long long hLo = rc + (long long)r, hMid = rc + 2LL * r;
            T[hLo] = rnd.next(0.30, 0.50);
            T[hMid] = rnd.next(0.0, 0.02);
        }
    } else if (mode == "cross_construct") {
        for (int h = 1; h <= H; h++) T[h] = rnd.next(0.0, 0.06);
        // Two DIFFERENT operators (different ratios r_i != r_j) whose DIFFERENT
        // sideband orders (n_i,n_j) both land on the SAME harmonic h: since the
        // two operators run at different frequencies, their contributions are
        // genuinely separate terms in the multi-operator Jacobi-Anger expansion
        // (no Bessel-addition-theorem collapse the way same-ratio operators
        // would have), so their amplitudes at h can add constructively past what
        // either operator alone could reach. Greedy's rule assigns ONE operator
        // (order 1) per target harmonic and can never route a second operator
        // onto the SAME harmonic via a different order.
        vector<array<long long,6>> cand; // slot_i, r_i, n_i, slot_j, r_j, n_j (h implicit)
        for (int i = 0; i < K; i++)
            for (int j = i + 1; j < K; j++)
                for (int ni = 1; ni <= 3; ni++)
                    for (int ri = Rlo[i]; ri <= Rhi[i]; ri++)
                        for (int nj = 1; nj <= 3; nj++) {
                            long long off = (long long)ni * ri;
                            if (off % nj != 0) continue;
                            long long rj = off / nj;
                            if (rj < Rlo[j] || rj > Rhi[j] || rj == ri) continue;
                            long long h = rc + off;
                            if (h < 1 || h > H) continue;
                            cand.push_back({i, ri, ni, j, rj, nj});
                        }
        if (!cand.empty()) {
            auto c = cand[rnd.next(0, (int)cand.size() - 1)];
            int i = (int)c[0], ri = (int)c[1], ni = (int)c[2];
            int j = (int)c[3], nj = (int)c[5];
            long long h = rc + (long long)ni * ri;
            double peakI = 0.0, peakJ = 0.0;
            for (int s = 0; s <= 2000; s++) {
                double Iv = Cmax[i] * s / 2000.0;
                peakI = max(peakI, fabs(std::cyl_bessel_j((double)ni, Iv)));
            }
            for (int s = 0; s <= 2000; s++) {
                double Iv = Cmax[j] * s / 2000.0;
                peakJ = max(peakJ, fabs(std::cyl_bessel_j((double)nj, Iv)));
            }
            // request comfortably above either single peak, at most both peaks summed
            double want = max(peakI, peakJ) * rnd.next(1.08, 1.28);
            want = min(want, (peakI + peakJ) * 0.92);
            T[h] = min(3.0, want);
        }
    } else { // highcost
        for (int h = 1; h <= H; h++)
            T[h] = (rnd.next(0, 99) < 65) ? rnd.next(0.15, 0.7) : 0.0;
    }

    println(K, H, rc, lambda);
    for (int i = 0; i < K; i++) println(Rlo[i], Rhi[i], Cmax[i]);
    for (int h = 1; h <= H; h++) {
        printf("%.4f%c", T[h], h == H ? '\n' : ' ');
    }
    return 0;
}
