// TIER: strong
// The insight: the checker turns a bin's declared weight into a cell BUDGET
// (largest-remainder apportionment of the weights), and a budget of area A
// grown as a compact wedge reaches out to radius ~ sqrt(2A/dtheta) -- so
// matching the target silhouette needs w(bin) ~ AREA SWEPT BY THE TARGET's
// true boundary across that bin's arc, i.e. the INTEGRAL of r(theta)^2 over
// the bin, not a value read off at a single angle. A single point sample
// (the greedy tier) is exactly what fails on a bin whose arc straddles a
// sharp vertex: it grabs whatever radius happens to sit at the bin's center
// and misses the area actually concentrated near the vertex. Here we
// numerically integrate r(theta)^2 across each 5-degree bin (Simpson's rule)
// to recover that area honestly -- this is the Wulff-shape consequence: no
// isotropic (direction-blind) budget can ever do this, regardless of tuning.
#include <bits/stdc++.h>
using namespace std;

static double boundaryRadius(int K, const vector<double>& r, double theta) {
    double sector = 2.0 * M_PI / K;
    int i = (int)floor(theta / sector);
    if (i < 0) i = 0;
    if (i >= K) i = K - 1;
    int j = (i + 1) % K;
    double th_i = i * sector, th_j = (i + 1) * sector;
    double ax = r[i] * cos(th_i), ay = r[i] * sin(th_i);
    double bx = r[j] * cos(th_j), by = r[j] * sin(th_j);
    double dx = cos(theta), dy = sin(theta);
    double abx = bx - ax, aby = by - ay;
    double denom = dx * aby - dy * abx;
    if (fabs(denom) < 1e-12) denom = (denom < 0 ? -1e-12 : 1e-12);
    double t = (ax * by - ay * bx) / denom;
    if (t < 1e-6) t = 1e-6;
    return t;
}

// Simpson's rule for the integral of r(theta)^2 over [lo, hi].
static double integratedArea(int K, const vector<double>& r, double lo, double hi, int nsub) {
    double step = (hi - lo) / nsub;
    double s = 0.0;
    for (int i = 0; i <= nsub; i++) {
        double th = lo + i * step;
        double rt = boundaryRadius(K, r, th);
        double val = rt * rt;
        double wgt = (i == 0 || i == nsub) ? 1.0 : (i % 2 == 1 ? 4.0 : 2.0);
        s += wgt * val;
    }
    return s * step / 3.0;
}

int main() {
    long long seed, M; int K;
    if (scanf("%lld %d %lld", &seed, &K, &M) != 3) return 0;
    vector<double> r(K);
    for (int i = 0; i < K; i++) { long long ri; scanf("%lld", &ri); r[i] = (double)ri; }

    const int NBINS = 72;
    for (int b = 0; b < NBINS; b++) {
        double lo = b * (2.0 * M_PI / NBINS);
        double hi = (b + 1) * (2.0 * M_PI / NBINS);
        double w = integratedArea(K, r, lo, hi, 20);
        printf("%s%.6f", b ? " " : "", w);
    }
    printf("\n");
    return 0;
}
