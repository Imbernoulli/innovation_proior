// TIER: greedy
// The obvious first idea, once you notice the frequency check has *some*
// slack (tol_g): nudge the fair-share target ratio slightly off the exact
// input fraction so the fair-share ("deficit") rule stops re-synchronizing
// to a short period, unlocking more distinct factors. A plausible way to
// pick "how big a nudge is still safe" is to shrink it roughly like
// 1/sqrt(L) (a "the noise averages out over more samples" argument) --
// works nicely on the instances you'd typically test by hand. It is the
// WRONG rate, though: the running letter-count deviation this nudge causes
// after L symbols grows like L * (C/sqrt(L)) = C*sqrt(L), which INCREASES
// without bound as L grows, instead of staying inside tol_g. So on the
// small/medium tests it looks exactly as good as the real insight -- and
// on the larger tests it silently drifts the whole-word letter counts
// outside the allowed tolerance and is rejected outright. (For this binary
// alphabet the nudge DIRECTION is just a sign choice -- the golden-ratio
// term below only decides which letter gets the + -- so the only thing
// that actually separates this from solutions/strong.cpp is that scaling
// law, not the source of the perturbation.)
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static const double PHI = 1.6180339887498948482;
static const double C = 0.07; // "looked safe" during small-scale testing

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

    vector<double> pert(a);
    double mean = 0.0;
    for (int c = 0; c < a; c++) {
        double x = (c + 1) * PHI;
        x -= floor(x);
        pert[c] = x - 0.5;
        mean += pert[c];
    }
    mean /= a;
    double maxAbs = 1e-9;
    for (int c = 0; c < a; c++) { pert[c] -= mean; maxAbs = max(maxAbs, fabs(pert[c])); }

    // WRONG rate: shrinks like 1/sqrt(L) instead of 1/L
    double budget = C / sqrt((double)L);
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
