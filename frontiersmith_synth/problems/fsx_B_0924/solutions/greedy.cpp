// TIER: greedy
// The obvious first attempt: "aim growth toward where the target reaches
// farther." Sample the target's boundary radius r(theta) at the CENTER of each
// of the 72 bins and use it directly as the growth budget. This correctly
// points budget toward tips and away from facets, but a single point sample at
// the bin's center badly misjudges any bin that straddles a sharp vertex
// (where the true boundary radius changes fast across the bin's 5-degree
// arc) -- exactly the needle spikes that make these tests hard.
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

int main() {
    long long seed, M; int K;
    if (scanf("%lld %d %lld", &seed, &K, &M) != 3) return 0;
    vector<double> r(K);
    for (int i = 0; i < K; i++) { long long ri; scanf("%lld", &ri); r[i] = (double)ri; }

    for (int b = 0; b < 72; b++) {
        double theta = (b + 0.5) * (2.0 * M_PI / 72.0);
        double rt = boundaryRadius(K, r, theta);
        printf("%s%.6f", b ? " " : "", rt);
    }
    printf("\n");
    return 0;
}
