// TIER: greedy
#include <bits/stdc++.h>
using namespace std;

// The "obvious first pass": assign each modulator, in slot order, to whichever
// still-uncovered target harmonic it can reach directly at sideband order n=1
// (ratio r = h - rc) with the largest remaining target amplitude, then set the
// index by a LINEAR guess index = amplitude * SCALE. This treats "sideband
// amplitude" as if it scaled linearly with modulation index and treats each
// harmonic as owned by a single slot -- it never accounts for J_n(I) being a
// bounded, non-monotonic function of I (it peaks near I~1.84 for n=1 and then
// falls, and the peak sits at even LARGER I -- and a lower ceiling -- for
// higher sideband orders), never revisits a choice, never lets two slots
// reinforce the same harmonic, and never weighs the lambda*index cost term.
int main() {
    int K, H, rc;
    double lambda;
    cin >> K >> H >> rc >> lambda;
    vector<int> Rlo(K), Rhi(K);
    vector<double> Cmax(K);
    for (int i = 0; i < K; i++) cin >> Rlo[i] >> Rhi[i] >> Cmax[i];
    vector<double> T(H + 1);
    for (int h = 1; h <= H; h++) cin >> T[h];

    const double SCALE = 2.5;   // naive "index ~ amplitude" calibration
    vector<bool> used(H + 1, false);

    for (int i = 0; i < K; i++) {
        int bestH = -1;
        double bestVal = -1.0;
        for (int h = 1; h <= H; h++) {
            if (used[h]) continue;
            int rCand = h - rc;
            if (rCand < Rlo[i] || rCand > Rhi[i]) continue;
            if (T[h] > bestVal) { bestVal = T[h]; bestH = h; }
        }
        int r; double idx;
        if (bestH == -1) {
            r = Rlo[i]; idx = 0.0;
        } else {
            r = bestH - rc;
            used[bestH] = true;
            idx = min(Cmax[i], T[bestH] * SCALE);
        }
        printf("%d %.6f\n", r, idx);
    }
    return 0;
}
