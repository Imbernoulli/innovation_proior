// TIER: strong
// The switching insight: a shift does not need to be a good diffuser on its own -- it only has
// to kill whatever slow subspace the OTHER shift leaves behind. Single-edge swaps (the greedy
// move) can't discover this because they never move a whole node's incident edges together, so
// they can't consolidate a hub's spokes onto one shift while leaving the complementary long-range
// links to the other. This solver adds a second move type -- "pick a node, try moving ALL of its
// currently-scattered incident edges onto whichever shift already holds more of them" -- which is
// exactly a decomposition/consolidation move: it discovers edge-disjoint substructures (a hub's
// star, a matching) and keeps each one coherent on one shift instead of averaging it across both.
// Both move types are scored by the SAME internal Floquet-radius estimate as greedy; only the
// search neighborhood (and the iteration budget) differs -- this is insight, not just more of the
// same move.
#include <bits/stdc++.h>
using namespace std;

struct E { int u, v; double w; };
static int N;
static const int KTERMS = 6;
static const double SUBSTEP_BOUND = 1.3;
static const int POWER_ITERS = 10;

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
// races 3 structurally different start vectors, returns the max (robust to a slow mode hidden
// from any single one of them -- see chk.cc for the rationale).
// returns the mean-deflated dominant-direction vector after POWER_ITERS steps from the ramp
// start (used to RANK nodes by how much they still resist damping -- the "which slow subspace is
// still alive" signal that guides the block-consolidation moves below).
static vector<double> floquetVector(const vector<E>& dayE, const vector<E>& nightE, double tau, double lambdaBound) {
    vector<double> v(N);
    for (int i = 0; i < N; i++) v[i] = (double)i - (N - 1) / 2.0;
    double mean = 0; for (int i = 0; i < N; i++) mean += v[i]; mean /= N;
    for (int i = 0; i < N; i++) v[i] -= mean;
    double nrm0 = 0; for (int i = 0; i < N; i++) nrm0 += v[i] * v[i]; nrm0 = sqrt(nrm0);
    if (nrm0 > 1e-12) for (int i = 0; i < N; i++) v[i] /= nrm0;
    for (int it = 0; it < POWER_ITERS; it++) {
        v = applyExpL(dayE, tau, lambdaBound, v);
        v = applyExpL(nightE, tau, lambdaBound, v);
        double m2 = 0; for (int i = 0; i < N; i++) m2 += v[i]; m2 /= N;
        for (int i = 0; i < N; i++) v[i] -= m2;
        double nrm = 0; for (int i = 0; i < N; i++) nrm += v[i] * v[i]; nrm = sqrt(nrm);
        if (nrm < 1e-300) break;
        for (int i = 0; i < N; i++) v[i] /= nrm;
    }
    return v;
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
    vector<vector<int>> incident(n);
    for (int i = 0; i < m; i++) {
        int u, v, w;
        scanf("%d %d %d", &u, &v, &w);
        u--; v--;
        eu[i] = u; ev_[i] = v; ew[i] = w;
        deg[u] += w; deg[v] += w;
        incident[u].push_back(i);
        incident[v].push_back(i);
    }
    N = n;
    for (int i = 0; i < n; i++) maxDeg = max(maxDeg, deg[i]);
    double lambdaBound = max(2.0 * maxDeg, 1e-9);

    // initial guess: the same one-pass "balanced half graph" recipe as the greedy tier (connect
    // new components when possible, else balance by count) -- strong then refines it further.
    vector<int> shift(m);
    long long cntDay = 0, cntNight = 0;
    {
        vector<int> pDay(n), pNight(n);
        iota(pDay.begin(), pDay.end(), 0);
        iota(pNight.begin(), pNight.end(), 0);
        function<int(vector<int>&, int)> find = [&](vector<int>& p, int x) {
            while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; }
            return x;
        };
        for (int i = 0; i < m; i++) {
            bool canDay = cntDay < capDay, canNight = cntNight < capNight;
            bool helpsDay = canDay && find(pDay, eu[i]) != find(pDay, ev_[i]);
            bool helpsNight = canNight && find(pNight, eu[i]) != find(pNight, ev_[i]);
            int pick;
            if (helpsDay && helpsNight) pick = (cntDay <= cntNight) ? 0 : 1;
            else if (helpsDay) pick = 0;
            else if (helpsNight) pick = 1;
            else if (canDay && canNight) pick = (cntDay <= cntNight) ? 0 : 1;
            else if (canDay) pick = 0;
            else pick = 1;
            shift[i] = pick;
            if (pick == 0) { pDay[find(pDay, eu[i])] = find(pDay, ev_[i]); cntDay++; }
            else { pNight[find(pNight, eu[i])] = find(pNight, ev_[i]); cntNight++; }
        }
    }

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
    vector<int> best = shift;
    double bestScore = cur;

    mt19937 rng(987654321u);

    // try a single block-consolidation move for node u; evaluate/accept via the shared proxy.
    auto tryBlockMove = [&](int u) {
        if ((int)incident[u].size() < 2) return;
        int inDay = 0, inNight = 0;
        for (int eid : incident[u]) (shift[eid] == 0 ? inDay : inNight)++;
        int target = (inDay >= inNight) ? 0 : 1;
        bool anyChange = false;
        for (int eid : incident[u]) if (shift[eid] != target) { anyChange = true; break; }
        if (!anyChange) return;
        long long willDay = cntDay, willNight = cntNight;
        for (int eid : incident[u]) {
            if (shift[eid] != target) {
                if (target == 0) { willDay++; willNight--; } else { willNight++; willDay--; }
            }
        }
        if (willDay > capDay || willNight > capNight) return;
        vector<int> savedShift = shift;
        for (int eid : incident[u]) {
            if (shift[eid] != target) {
                if (target == 0) { cntDay++; cntNight--; } else { cntNight++; cntDay--; }
                shift[eid] = target;
            }
        }
        buildLists(dayE, nightE);
        double cand = floquetRadius(dayE, nightE, tau, lambdaBound);
        if (cand <= cur + 1e-12) {
            cur = cand;
            if (cur < bestScore) { bestScore = cur; best = shift; }
        } else {
            shift = savedShift;
            cntDay = 0; cntNight = 0;
            for (int i = 0; i < m; i++) (shift[i] == 0 ? cntDay : cntNight)++;
        }
    };

    // --- interleaved cycles: eigenvector-guided epoch, then a chunk of blind polish, repeated.
    // The eigenvector-guided move is the genuine insight -- compute the CURRENT residual
    // slow-direction vector v, rank hubs by how much of that residual direction still lives on
    // their incident edges (w*(v_u - v_x)^2), and consolidate the biggest offenders FIRST, a
    // targeted decomposition move plain random search (greedy) never gets to try. Interleaving
    // with blind swaps lets each guided pass "unstick" a plateau the swaps found, and vice versa.
    int CYCLES = (n <= 40) ? 8 : 6;
    int totalBudget = min(520, 11 * m + 20);
    int perCycle = max(8, totalBudget / CYCLES);
    for (int cyc = 0; cyc < CYCLES; cyc++) {
        buildLists(dayE, nightE);
        vector<double> v = floquetVector(dayE, nightE, tau, lambdaBound);
        vector<pair<double,int>> impact(n);
        for (int u = 0; u < n; u++) {
            double s = 0;
            for (int eid : incident[u]) {
                int other = (eu[eid] == u) ? ev_[eid] : eu[eid];
                double d = v[u] - v[other];
                s += ew[eid] * d * d;
            }
            impact[u] = {-s, u}; // sort descending impact -> ascending on negated value
        }
        sort(impact.begin(), impact.end());
        for (auto& pr : impact) tryBlockMove(pr.second);

        for (int iter = 0; iter < perCycle; iter++) {
            int ia = rng() % m, ib = rng() % m;
            if (shift[ia] == shift[ib]) continue;
            swap(shift[ia], shift[ib]);
            buildLists(dayE, nightE);
            double cand = floquetRadius(dayE, nightE, tau, lambdaBound);
            if (cand <= cur + 1e-12) {
                cur = cand;
                if (cur < bestScore) { bestScore = cur; best = shift; }
            } else {
                swap(shift[ia], shift[ib]);
            }
        }
    }

    string out;
    out.reserve(2 * (size_t)m);
    for (int i = 0; i < m; i++) { out += (best[i] == 0 ? '0' : '1'); out += '\n'; }
    fwrite(out.data(), 1, out.size(), stdout);
    return 0;
}
