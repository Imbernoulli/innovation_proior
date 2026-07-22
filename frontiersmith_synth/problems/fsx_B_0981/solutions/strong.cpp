// TIER: strong
#include <bits/stdc++.h>
using namespace std;

// Genuine insight over the greedy trap:
//  (1) BESSEL INVERSION: to hit a desired sideband amplitude we scan I over its
//      cap and evaluate the true J_n(I) (bounded, non-monotonic -- it peaks near
//      I~1.84 for n=1 then falls), instead of assuming amplitude scales linearly
//      with I.
//  (2) We build the spectrum with a windowed accumulator (running per-harmonic
//      real magnitude, folded slot by slot; the window is generous but, like any
//      finite window, can drop an extreme-order combination reached only through
//      very large ratios -- an internal search-quality tradeoff, not a scoring
//      rule) so a candidate is scored by its actual effect on the current
//      partial spectrum -- this naturally lets
//      two operators reinforce the SAME target harmonic (constructive interference
//      the greedy's one-harmonic-per-slot rule can never reach) and naturally
//      penalizes a choice that would spill/interfere destructively where the
//      target wants near-silence.
//  (3) Selection always compares actual total objective F = spectral error +
//      lambda*cost, so it trades off amplitude accuracy against modulation cost
//      instead of chasing raw amplitude.
//  (4) A short coordinate-descent refinement re-optimizes every slot against the
//      OTHERS' current choices, correcting the order-dependence of the initial pass.

static const int NMAX = 10;

static double besselJ(int n, double x) {
    int an = abs(n);
    double v = std::cyl_bessel_j((double)an, x);
    if (n < 0 && (an % 2) != 0) v = -v;   // J_{-n}(x) = (-1)^n J_n(x)
    return v;
}

static void besselRow(double x, vector<double> &out) {
    out.assign(2 * NMAX + 1, 0.0);
    for (int n = 0; n <= NMAX; n++) {
        double v = std::cyl_bessel_j((double)n, x);
        out[NMAX + n] = v;
        out[NMAX - n] = (n % 2 == 0) ? v : -v;
    }
}

int K, H, rc;
double lambda;
vector<int> Rlo, Rhi;
vector<double> Cmax;
vector<double> T;
int W, CENTER;   // windowed accumulator: index = CENTER + offset, offset in [-W,W]

// Fold one operator (ratio r, Bessel row J) into accumulator `in`, returning new acc.
static vector<double> foldAcc(const vector<double> &in, int r, const vector<double> &J) {
    vector<double> out(in.size(), 0.0);
    int sz = (int)in.size();
    for (int c = 0; c < sz; c++) {
        double base = in[c];
        if (base == 0.0) continue;
        for (int n = -NMAX; n <= NMAX; n++) {
            double coeff = J[NMAX + n];
            if (coeff == 0.0) continue;
            int nc = c + n * r;
            if (nc < 0 || nc >= sz) continue;
            out[nc] += base * coeff;
        }
    }
    return out;
}

// A signed sideband combination lands at raw signed harmonic index
// hsum = rc + sum(n_i*r_i), and the checker folds |hsum| -> the SAME magnitude
// bin. Our accumulator is indexed by offset-from-rc (hsum - rc), so the two
// branches that fold onto harmonic h are hsum=+h (offset h-rc) and hsum=-h
// (offset -h-rc); both must be added (with sign) before taking the magnitude.
static double signedAmpAt(const vector<double> &acc, int h) {
    double v = 0.0;
    int off1 = h - rc;
    if (off1 >= -W && off1 <= W) v += acc[CENTER + off1];
    int off2 = -h - rc;
    if (off2 >= -W && off2 <= W) v += acc[CENTER + off2];
    return v;
}

static double totalF(const vector<double> &acc, double costSoFar) {
    double F = 0.0;
    for (int h = 1; h <= H; h++) {
        double amp = fabs(signedAmpAt(acc, h));
        double d = amp - T[h];
        F += d * d;
    }
    F += lambda * costSoFar;
    return F;
}

int main() {
    cin >> K >> H >> rc >> lambda;
    Rlo.assign(K, 0); Rhi.assign(K, 0); Cmax.assign(K, 0.0);
    for (int i = 0; i < K; i++) cin >> Rlo[i] >> Rhi[i] >> Cmax[i];
    T.assign(H + 1, 0.0);
    for (int h = 1; h <= H; h++) cin >> T[h];

    // Generous enough to hold the drift from several folds at the largest
    // allowed ratio (<=20) and sideband order (<=NMAX=10) without dropping terms
    // that matter for a decent internal search; still cheap (see foldAcc cost).
    W = H + rc + NMAX * 20 * 3 + 50;
    CENTER = W;
    int sz = 2 * W + 1;

    vector<int> ratio(K);
    vector<double> idx(K, 0.0);
    for (int i = 0; i < K; i++) ratio[i] = Rlo[i];

    vector<double> baseAcc(sz, 0.0);
    baseAcc[CENTER] = 1.0;   // bare carrier, offset 0

    // background(excludeIdx): fold every slot except `excludeIdx` (using its
    // current committed ratio/idx) starting from the bare-carrier accumulator.
    auto background = [&](int excludeIdx) {
        vector<double> acc = baseAcc;
        double cost = 0.0;
        for (int i = 0; i < K; i++) {
            if (i == excludeIdx) continue;
            vector<double> J; besselRow(idx[i], J);
            acc = foldAcc(acc, ratio[i], J);
            cost += idx[i];
        }
        return make_pair(acc, cost);
    };

    // optimizeSlot(i, acc, costOthers): given the OTHER slots' fixed contribution,
    // pick the best (ratio,index) for slot i -- returns (ratio, index, achieved F).
    // Two complementary candidate sources: (a) BESSEL INVERSION -- for each
    // reachable sideband order, scan I to make |J_n(I)| land the true amplitude
    // needed at that harmonic (accounts for non-monotonicity), and (b) a direct
    // dense I scan evaluated by actual full objective (catches small supportive
    // indices that don't target one harmonic exactly, e.g. shaving a neighbour).
    auto optimizeSlot = [&](int i, const vector<double> &acc, double costOthers) {
        int bestR = Rlo[i]; double bestI = 0.0; double bestF = 1e18;
        for (int r = Rlo[i]; r <= Rhi[i]; r++) {
            // candidate 1: no modulation from this slot at all
            {
                vector<double> J; besselRow(0.0, J);
                vector<double> trial = foldAcc(acc, r, J);
                double F = totalF(trial, costOthers + 0.0);
                if (F < bestF) { bestF = F; bestR = r; bestI = 0.0; }
            }
            // candidate orders: invert each reachable order's Bessel function
            // against the harmonic it would land on, given the current background.
            for (int n = -4; n <= 4; n++) {
                if (n == 0) continue;
                long long h = (long long)rc + (long long)n * r;
                if (h < 1 || h > H) continue;
                double backgroundVal = signedAmpAt(acc, (int)h);
                double target = T[(int)h];
                int steps = 200;
                double bestErr = 1e18, bestIforN = 0.0;
                for (int s = 0; s <= steps; s++) {
                    double I = Cmax[i] * (double)s / steps;
                    double val = besselJ(n, I);
                    double err = fabs(fabs(backgroundVal + val) - target);
                    if (err < bestErr) { bestErr = err; bestIforN = I; }
                }
                vector<double> J; besselRow(bestIforN, J);
                vector<double> trial = foldAcc(acc, r, J);
                double F = totalF(trial, costOthers + bestIforN);
                if (F < bestF) { bestF = F; bestR = r; bestI = bestIforN; }
            }
            // candidate set (b): dense direct index scan, judged by ACTUAL total F.
            int dsteps = 40;
            for (int s = 0; s <= dsteps; s++) {
                double I = Cmax[i] * (double)s / dsteps;
                vector<double> J; besselRow(I, J);
                vector<double> trial = foldAcc(acc, r, J);
                double F = totalF(trial, costOthers + I);
                if (F < bestF) { bestF = F; bestR = r; bestI = I; }
            }
        }
        return make_tuple(bestR, bestI, bestF);
    };

    // ---- construction: matching-pursuit -- place whichever remaining slot gives
    // the biggest global improvement first (order-independent, unlike a fixed
    // slot-by-slot pass), against the growing background of already-placed slots.
    vector<double> acc = baseAcc;
    double costSoFar = 0.0;
    vector<bool> placed(K, false);
    for (int step = 0; step < K; step++) {
        int bestSlot = -1, bestR = 0; double bestI = 0.0, bestF = 1e18;
        for (int i = 0; i < K; i++) {
            if (placed[i]) continue;
            auto pr = optimizeSlot(i, acc, costSoFar);
            if (get<2>(pr) < bestF) { bestF = get<2>(pr); bestSlot = i; bestR = get<0>(pr); bestI = get<1>(pr); }
        }
        placed[bestSlot] = true;
        ratio[bestSlot] = bestR; idx[bestSlot] = bestI;
        vector<double> J; besselRow(bestI, J);
        acc = foldAcc(acc, bestR, J);
        costSoFar += bestI;
    }

    // ---- coordinate-descent refinement: re-optimize each slot against the others ----
    for (int pass = 0; pass < 8; pass++) {
        for (int i = 0; i < K; i++) {
            auto bg = background(i);
            auto pr = optimizeSlot(i, bg.first, bg.second);
            ratio[i] = get<0>(pr); idx[i] = get<1>(pr);
        }
    }

    for (int i = 0; i < K; i++) printf("%d %.6f\n", ratio[i], idx[i]);
    return 0;
}
