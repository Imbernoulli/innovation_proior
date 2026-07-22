// TIER: strong
// Insight: the balance constraint only requires each window count to stay
// within a *tolerance* of the ideal proportional value -- it never demands
// the ratio itself be exactly the (secretly low-denominator, rational)
// input fraction. Any EXACT-ratio balanced binary word is forced (by the
// classical balance/Morse-Hedlund theorem) into the SAME small period as
// that fraction's reduced denominator -- that is why trivial's period-<=12
// construction cannot be improved by re-deriving it differently, only by
// abandoning the exact ratio. So nudge the target ratio slightly off the
// input fraction instead. For a binary alphabet the nudge is, after fixing
// its magnitude, only a +/- sign choice (which letter gets nudged up) --
// the direction does not matter by symmetry, so which one PHI happens to
// pick is cosmetic. What is NOT cosmetic, and is the actual content of the
// insight, is the magnitude's SCALING LAW in L: the fair-share ("deficit")
// rule keeps every prefix count within O(1) of i*target[c], so after L
// symbols the whole-word count drifts from freq[c] by about L*|nudge|. For
// that drift to stay inside tol_g at EVERY tested L (not just the ones you
// happened to try), the nudge magnitude must shrink like Theta(1/L) --
// exactly what `budget` below does. Get that rate right (this file) and the
// ratio is irrational-for-all-practical-purposes at any tested K: the
// fair-share rule never re-synchronizes to a short period, so factor
// complexity AND palindromic (richness) complexity keep growing with K
// instead of saturating after a handful of letters. Get the rate wrong
// (see greedy.cpp, which uses the plausible-looking but incorrect 1/sqrt(L))
// and the SAME idea silently stops being feasible once L grows.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static string deficitIrrational(ll L, int a, const vector<double>& target) {
    vector<ll> cnt(a, 0);
    string w(L, '0');
    for (ll i = 0; i < L; i++) {
        int best = 0;
        double bestScore = target[0] * (double)(i + 1) - (double)cnt[0];
        for (int c = 1; c < a; c++) {
            double sc = target[c] * (double)(i + 1) - (double)cnt[c];
            if (sc > bestScore + 1e-12) { bestScore = sc; best = c; }
        }
        cnt[best]++;
        w[i] = (char)('0' + best);
    }
    return w;
}

int main() {
    int a, K, tol_w, tol_g, w_pal;
    ll L;
    cin >> a >> L >> K >> tol_w >> tol_g >> w_pal;
    vector<ll> freq(a);
    for (int c = 0; c < a; c++) cin >> freq[c];

    const double PHI = 1.6180339887498948482;
    vector<double> pert(a);
    double mean = 0.0;
    for (int c = 0; c < a; c++) {
        double x = (c + 1) * PHI;
        x -= floor(x);            // fractional part, irrational for c+1 >= 1
        pert[c] = x - 0.5;        // range (-0.5, 0.5)
        mean += pert[c];
    }
    mean /= a;
    double maxAbs = 1e-9;
    for (int c = 0; c < a; c++) { pert[c] -= mean; maxAbs = max(maxAbs, fabs(pert[c])); }

    // keep the accumulated drift over L steps comfortably inside tol_g
    double budget = 0.4 * (double)max(1, tol_g) / (double)L;
    double scale = budget / maxAbs;

    vector<double> target(a);
    double tsum = 0.0;
    for (int c = 0; c < a; c++) {
        target[c] = (double)freq[c] / (double)L + pert[c] * scale;
        if (target[c] < 0) target[c] = 0;
        tsum += target[c];
    }
    for (int c = 0; c < a; c++) target[c] /= tsum;

    string w = deficitIrrational(L, a, target);
    cout << w << "\n";
    return 0;
}
