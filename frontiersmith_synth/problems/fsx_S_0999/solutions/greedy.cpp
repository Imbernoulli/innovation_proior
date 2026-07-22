// TIER: greedy
// The "obvious" approach an average strong coder writes first: build a single deterministic
// one-pass schedule that tries to make BOTH shifts individually decent, balanced, well-connected
// graphs (never looking at how the two shifts interact as a non-commuting PRODUCT -- a static,
// average-Laplacian recipe), then spend a SMALL, fixed budget of blind random single-edge swaps
// polishing it (a light, unguided touch-up -- not a search strategy). It never moves a node's
// edges as a block and never uses more than a token amount of feedback, so it cannot discover or
// preserve the edge-disjoint substructures that make the two shifts truly complementary.
#include <bits/stdc++.h>
using namespace std;

struct E { int u, v; double w; };
static int N;
static const int KTERMS = 8;
static const double SUBSTEP_BOUND = 1.0;
static const int POWER_ITERS = 12;

static void applyL(const vector<E>& edges, const vector<double>& x, vector<double>& y) {
    fill(y.begin(), y.end(), 0.0);
    for (const auto& e : edges) {
        double diff = (x[e.u] - x[e.v]) * e.w;
        y[e.u] += diff; y[e.v] -= diff;
    }
}
static vector<double> applyExpL(const vector<E>& edges, double tau, double lambdaBound, vector<double> x) {
    if (tau <= 0.0 || edges.empty()) return x;
    int s = 0; double delta = tau;
    while (delta * lambdaBound > SUBSTEP_BOUND && s < 40) { delta *= 0.5; s++; }
    long long reps = 1LL << s;
    vector<double> cur = x, term(N), Lterm(N), y(N);
    for (long long r = 0; r < reps; r++) {
        term = cur; y = cur;
        for (int j = 1; j <= KTERMS; j++) {
            applyL(edges, term, Lterm);
            double coef = -delta / j;
            for (int i = 0; i < N; i++) term[i] = coef * Lterm[i];
            for (int i = 0; i < N; i++) y[i] += term[i];
        }
        cur = y;
    }
    return cur;
}
static double floquetRadiusFrom(vector<double> v, const vector<E>& dayE, const vector<E>& nightE,
                                 double tau, double lambdaBound) {
    double mean = 0; for (int i = 0; i < N; i++) mean += v[i]; mean /= N;
    for (int i = 0; i < N; i++) v[i] -= mean;
    double nrm0 = 0; for (int i = 0; i < N; i++) nrm0 += v[i] * v[i]; nrm0 = sqrt(nrm0);
    if (nrm0 < 1e-12) return 0.0;
    for (int i = 0; i < N; i++) v[i] /= nrm0;
    double rho = 0.0;
    for (int it = 0; it < POWER_ITERS; it++) {
        v = applyExpL(dayE, tau, lambdaBound, v);
        v = applyExpL(nightE, tau, lambdaBound, v);
        double m2 = 0; for (int i = 0; i < N; i++) m2 += v[i]; m2 /= N;
        for (int i = 0; i < N; i++) v[i] -= m2;
        double nrm = 0; for (int i = 0; i < N; i++) nrm += v[i] * v[i]; nrm = sqrt(nrm);
        if (nrm < 1e-300) { rho = 0.0; break; }
        rho = nrm;
        for (int i = 0; i < N; i++) v[i] /= nrm;
    }
    return rho;
}
static double floquetRadius(const vector<E>& dayE, const vector<E>& nightE, double tau, double lambdaBound) {
    double best = 0.0;
    {
        vector<double> v(N);
        for (int i = 0; i < N; i++) v[i] = (double)i - (N - 1) / 2.0;
        best = max(best, floquetRadiusFrom(v, dayE, nightE, tau, lambdaBound));
    }
    {
        vector<double> v(N);
        for (int i = 0; i < N; i++) v[i] = (i % 2 == 0) ? 1.0 : -1.0;
        best = max(best, floquetRadiusFrom(v, dayE, nightE, tau, lambdaBound));
    }
    {
        vector<double> v(N);
        unsigned long long x = 88172645463325252ULL;
        for (int i = 0; i < N; i++) {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            v[i] = (double)((x % 2000001ULL)) / 1000000.0 - 1.0;
        }
        best = max(best, floquetRadiusFrom(v, dayE, nightE, tau, lambdaBound));
    }
    return best;
}

struct DSU {
    vector<int> p;
    DSU(int n) : p(n) { iota(p.begin(), p.end(), 0); }
    int find(int x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }
    bool unite(int a, int b) { a = find(a); b = find(b); if (a == b) return false; p[a] = b; return true; }
};

int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    double tau;
    scanf("%lf", &tau);
    long long capDay, capNight;
    scanf("%lld %lld", &capDay, &capNight);
    vector<int> eu(m), ev_(m), ew(m);
    double maxDeg = 0;
    vector<double> deg(n, 0.0);
    for (int i = 0; i < m; i++) {
        int u, v, w;
        scanf("%d %d %d", &u, &v, &w);
        u--; v--;
        eu[i] = u; ev_[i] = v; ew[i] = w;
        deg[u] += w; deg[v] += w;
    }
    N = n;
    for (int i = 0; i < n; i++) maxDeg = max(maxDeg, deg[i]);
    double lambdaBound = max(2.0 * maxDeg, 1e-9);

    // --- one-pass "balanced half graph" recipe: no objective evaluation at all ---
    DSU dsuDay(n), dsuNight(n);
    long long cntDay = 0, cntNight = 0;
    vector<int> shift(m);
    for (int i = 0; i < m; i++) {
        bool canDay = cntDay < capDay, canNight = cntNight < capNight;
        bool helpsDay = canDay && dsuDay.find(eu[i]) != dsuDay.find(ev_[i]);
        bool helpsNight = canNight && dsuNight.find(eu[i]) != dsuNight.find(ev_[i]);
        int pick;
        if (helpsDay && helpsNight) pick = (cntDay <= cntNight) ? 0 : 1;
        else if (helpsDay) pick = 0;
        else if (helpsNight) pick = 1;
        else if (canDay && canNight) pick = (cntDay <= cntNight) ? 0 : 1;
        else if (canDay) pick = 0;
        else pick = 1;
        shift[i] = pick;
        if (pick == 0) { dsuDay.unite(eu[i], ev_[i]); cntDay++; }
        else { dsuNight.unite(eu[i], ev_[i]); cntNight++; }
    }

    // --- a token amount of blind, unguided polish: a handful of random single-edge swaps ---
    auto buildLists = [&](vector<E>& dayE, vector<E>& nightE) {
        dayE.clear(); nightE.clear();
        for (int i = 0; i < m; i++) {
            E e{eu[i], ev_[i], (double)ew[i]};
            if (shift[i] == 0) dayE.push_back(e); else nightE.push_back(e);
        }
    };
    vector<E> dayE, nightE;
    buildLists(dayE, nightE);
    double cur = floquetRadius(dayE, nightE, tau, lambdaBound);

    mt19937 rng(12345u);
    int budget = min(14, m / 6 + 3);
    for (int iter = 0; iter < budget; iter++) {
        int ia = rng() % m, ib = rng() % m;
        if (shift[ia] == shift[ib]) continue;
        swap(shift[ia], shift[ib]);
        buildLists(dayE, nightE);
        double cand = floquetRadius(dayE, nightE, tau, lambdaBound);
        if (cand <= cur + 1e-12) cur = cand;
        else swap(shift[ia], shift[ib]);
    }

    string out;
    out.reserve(2 * (size_t)m);
    for (int i = 0; i < m; i++) { out += (shift[i] == 0 ? '0' : '1'); out += '\n'; }
    fwrite(out.data(), 1, out.size(), stdout);
    return 0;
}
