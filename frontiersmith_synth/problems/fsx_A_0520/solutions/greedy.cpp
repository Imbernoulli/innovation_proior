// TIER: greedy
// The obvious recipe: pooled weighted k-means. Sum demand over seasons, run weighted
// Lloyd (L2 means) to place P depots on the pooled demand. This chases DENSITY, so it
// packs depots onto the compact/dense zones and starves the spread zones -- exactly the
// average-vs-worst-case trap. Ignores the max-over-seasons objective entirely.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, P, K;
    if (scanf("%d %d %d", &N, &P, &K) != 3) return 0;
    vector<double> X(N), Y(N);
    vector<double> s(N, 0.0);              // pooled demand
    for (int i = 0; i < N; i++) {
        long long x, y, w;
        scanf("%lld %lld", &x, &y);
        X[i] = (double)x; Y[i] = (double)y;
        for (int k = 0; k < K; k++) { scanf("%lld", &w); s[i] += (double)w; }
    }
    const double C = 100000.0;
    mt19937 rng(12345u);

    // Seed depots by sampling addresses PROPORTIONAL TO POOLED DEMAND -- the obvious "put
    // depots where the demand is" recipe. With every zone carrying equal demand this hands
    // each zone roughly the same number of depots, blind to how spread out a zone is.
    vector<double> cx(P), cy(P);
    {
        vector<double> pref(N + 1, 0.0);
        for (int i = 0; i < N; i++) pref[i + 1] = pref[i] + s[i] + 1e-9;
        double tot = pref[N];
        for (int c = 0; c < P; c++) {
            double r = uniform_real_distribution<double>(0, tot)(rng);
            int lo = 0, hi = N;                      // first index with pref[idx+1] > r
            while (lo < hi) { int mid = (lo + hi) / 2; if (pref[mid + 1] > r) hi = mid; else lo = mid + 1; }
            cx[c] = X[lo]; cy[c] = Y[lo];
        }
    }

    // weighted Lloyd (L2 means)
    vector<int> asg(N, 0);
    for (int it = 0; it < 40; it++) {
        for (int i = 0; i < N; i++) {
            double best = 1e30; int bj = 0;
            for (int p = 0; p < P; p++) {
                double dd = (X[i]-cx[p])*(X[i]-cx[p]) + (Y[i]-cy[p])*(Y[i]-cy[p]);
                if (dd < best) { best = dd; bj = p; }
            }
            asg[i] = bj;
        }
        vector<double> sx(P,0), sy(P,0), sw(P,0);
        for (int i = 0; i < N; i++) { double ww = s[i] + 1e-9; sx[asg[i]] += ww*X[i]; sy[asg[i]] += ww*Y[i]; sw[asg[i]] += ww; }
        for (int p = 0; p < P; p++) if (sw[p] > 0) { cx[p] = sx[p]/sw[p]; cy[p] = sy[p]/sw[p]; }
    }

    for (int p = 0; p < P; p++) {
        long long ox = (long long)llround(min(C, max(0.0, cx[p])));
        long long oy = (long long)llround(min(C, max(0.0, cy[p])));
        printf("%lld %lld\n", ox, oy);
    }
    return 0;
}
