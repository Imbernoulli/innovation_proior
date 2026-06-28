// ale-30: Tower Placement for Signal Coverage.
//
// Minimise the number of chosen tower sites so that every demand node i
// receives accumulated signal >= req[i], where a site j delivers power
// P[i][j] to demand i.  This is the covering ILP
//        min sum_j x_j   s.t.   sum_j P[i][j] x_j >= req[i],  x_j in {0,1}.
//
// Pipeline (the innovation):
//   1. Solve the COVERING LP RELAXATION (0<=x_j<=1) approximately with a
//      multiplicative-weights / fractional primal-dual solver -> fractional
//      site values xf[j] in [0,1].
//   2. GREEDY ROUNDING guided by "coverage-per-cost": process sites in order of
//      a score combining the LP fractional weight and current marginal
//      deficit-reduction, opening sites until all demands are met.  We do this
//      from several randomised perturbations of the LP-guided order and keep
//      the smallest feasible set (multi-start LP rounding).
//   3. LAZY-GREEDY REPAIR: if a rounding leaves a demand short, repeatedly add
//      the site with the largest remaining deficit-reduction gain, using a lazy
//      upper-bound priority queue (CELF) so few marginal gains are recomputed.
//   4. LOCAL SEARCH: redundancy pruning, plus remove-2-add-1 / remove-1-add-0
//      reductions to escape the pure-greedy local optimum and shrink the count.
//
// Output: k  then the k chosen 0-based site indices.  An all-on safety net
// guarantees we ALWAYS emit a feasible solution within the time budget.

#include <bits/stdc++.h>
using namespace std;

static int N, M;
static vector<double> req;          // req[i]
static vector<vector<double>> Pji;  // column-major: Pji[j][i] = power site j -> demand i

static const double EPS = 1e-7;

// timing
static chrono::steady_clock::time_point T0;
static double TIME_LIMIT = 3.5;     // seconds, comfortably under any 5s/10s judge
static double elapsed() {
    return chrono::duration<double>(chrono::steady_clock::now() - T0).count();
}

// rng
static uint64_t rng_state = 0x9e3779b97f4a7c15ULL;
static inline uint64_t xrand() {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
    return rng_state;
}
static inline double urand() { return (xrand() >> 11) * (1.0 / 9007199254740992.0); }

// ---- read instance ---------------------------------------------------------
static void readInput() {
    if (scanf("%d %d", &N, &M) != 2) { N = 0; M = 0; return; }
    req.assign(N, 0.0);
    for (int i = 0; i < N; i++) { if (scanf("%lf", &req[i]) != 1) req[i] = 0; }
    Pji.assign(M, vector<double>(N, 0.0));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++) {
            double v; if (scanf("%lf", &v) != 1) v = 0; Pji[j][i] = v;
        }
}

// total raw coverage of a site (sum over demands)
static vector<double> siteCov;
static void precompute() {
    siteCov.assign(M, 0.0);
    for (int j = 0; j < M; j++) {
        double s = 0; for (int i = 0; i < N; i++) s += Pji[j][i];
        siteCov[j] = s;
    }
}

// ---- covering-LP relaxation via multiplicative weights ---------------------
// Approximately solve  min 1^T x  s.t.  P x >= req, 0<=x<=1, returning xf.
// Dual weights y[i] emphasise still-unmet demands; each round opens a small
// fraction of the site that best serves the weighted unmet mass.
static vector<double> solveLP_MW() {
    vector<double> xf(M, 0.0);
    vector<double> recv(N, 0.0);
    const int ROUNDS = 240;
    for (int r = 0; r < ROUNDS; r++) {
        double wsum = 0.0;
        static vector<double> y; y.assign(N, 0.0);
        for (int i = 0; i < N; i++) {
            double deficit = req[i] - recv[i];
            if (deficit < 0) deficit = 0;
            double rel = (req[i] > EPS) ? deficit / req[i] : 0.0;
            y[i] = rel * rel + 1e-9;
            wsum += y[i];
        }
        if (wsum < 1e-9) break;
        int bestJ = -1; double bestScore = -1.0;
        for (int j = 0; j < M; j++) {
            if (xf[j] >= 1.0 - 1e-9) continue;
            const vector<double>& col = Pji[j];
            double sc = 0.0;
            for (int i = 0; i < N; i++) {
                if (y[i] <= 0) continue;
                double need = req[i] - recv[i];
                if (need <= 0) continue;
                sc += y[i] * min(need, col[i]);
            }
            if (sc > bestScore) { bestScore = sc; bestJ = j; }
        }
        if (bestJ < 0 || bestScore <= EPS) break;
        double step = 0.5, room = 1.0 - xf[bestJ];
        if (step > room) step = room;
        xf[bestJ] += step;
        const vector<double>& col = Pji[bestJ];
        for (int i = 0; i < N; i++) recv[i] += step * col[i];
    }
    return xf;
}

// ---- feasibility helper ----------------------------------------------------
static inline bool allMet(const vector<double>& recv) {
    for (int i = 0; i < N; i++) if (recv[i] < req[i] - EPS) return false;
    return true;
}

// marginal deficit-reduction gain of site j given current recv
static inline double marginalGain(int j, const vector<double>& recv) {
    const vector<double>& col = Pji[j];
    double g = 0.0;
    for (int i = 0; i < N; i++) {
        double need = req[i] - recv[i];
        if (need > 0) g += (need < col[i] ? need : col[i]);
    }
    return g;
}

// ---- one rounding+repair run from a (possibly perturbed) LP-guided order ----
// Returns an `open` mask and fills recv; always feasible (all-on fallback).
static vector<char> buildOnce(const vector<double>& xf, double noise) {
    vector<char> open(M, 0);
    vector<double> recv(N, 0.0);

    // priority key = (LP value + small const) * coverage, with optional noise.
    vector<pair<double,int>> order(M);
    for (int j = 0; j < M; j++) {
        double key = (xf[j] + 1e-3) * (siteCov[j] + 1.0);
        if (noise > 0) key *= (1.0 + noise * (urand() - 0.5));
        order[j] = { key, j };
    }
    sort(order.begin(), order.end(), greater<pair<double,int>>());

    for (auto& pr : order) {
        if (allMet(recv)) break;
        int j = pr.second;
        if (marginalGain(j, recv) <= EPS) continue;   // would not help any unmet demand
        open[j] = 1;
        const vector<double>& col = Pji[j];
        for (int i = 0; i < N; i++) recv[i] += col[i];
    }

    // lazy-greedy CELF repair for any residual shortfall
    if (!allMet(recv)) {
        priority_queue<tuple<double,int,int>> pq;
        vector<int> ver(M, 0);
        for (int j = 0; j < M; j++) {
            if (open[j]) continue;
            double g = marginalGain(j, recv);
            if (g > EPS) pq.push(make_tuple(g, j, 0));
        }
        while (!allMet(recv) && !pq.empty()) {
            auto top = pq.top(); pq.pop();
            int j = get<1>(top), v = get<2>(top);
            if (open[j] || v != ver[j]) continue;
            double fresh = marginalGain(j, recv);
            if (fresh <= EPS) continue;
            if (!pq.empty() && fresh < get<0>(pq.top()) - 1e-12) {
                ver[j]++; pq.push(make_tuple(fresh, j, ver[j])); continue;
            }
            open[j] = 1;
            const vector<double>& col = Pji[j];
            for (int i = 0; i < N; i++) recv[i] += col[i];
        }
    }

    // all-on safety net
    if (!allMet(recv)) {
        for (int j = 0; j < M; j++) if (!open[j]) {
            open[j] = 1;
            const vector<double>& col = Pji[j];
            for (int i = 0; i < N; i++) recv[i] += col[i];
        }
    }
    return open;
}

// recompute recv for a mask
static vector<double> recvOf(const vector<char>& open) {
    vector<double> recv(N, 0.0);
    for (int j = 0; j < M; j++) if (open[j]) {
        const vector<double>& col = Pji[j];
        for (int i = 0; i < N; i++) recv[i] += col[i];
    }
    return recv;
}

// redundancy pruning: drop opened sites (least coverage first) while feasible
static void prune(vector<char>& open, vector<double>& recv) {
    vector<pair<double,int>> ord;
    for (int j = 0; j < M; j++) if (open[j]) ord.push_back({ siteCov[j], j });
    sort(ord.begin(), ord.end());
    for (auto& pr : ord) {
        int j = pr.second;
        const vector<double>& col = Pji[j];
        bool ok = true;
        for (int i = 0; i < N; i++) if (recv[i] - col[i] < req[i] - EPS) { ok = false; break; }
        if (ok) { open[j] = 0; for (int i = 0; i < N; i++) recv[i] -= col[i]; }
    }
}

// remove-2-add-1 local search: try to replace two opened sites by one closed
// site and stay feasible (a net reduction of one tower).  Bounded effort.
static bool reduce2for1(vector<char>& open, vector<double>& recv) {
    vector<int> opened, closed;
    for (int j = 0; j < M; j++) (open[j] ? opened : closed).push_back(j);
    // order opened by least coverage (most droppable), closed by most coverage
    sort(opened.begin(), opened.end(), [](int a, int b){ return siteCov[a] < siteCov[b]; });
    sort(closed.begin(), closed.end(), [](int a, int b){ return siteCov[a] > siteCov[b]; });

    int OP = opened.size();
    for (int x = 0; x < OP && elapsed() < TIME_LIMIT; x++) {
        int a = opened[x];
        for (int y = x + 1; y < OP; y++) {
            int b = opened[y];
            // recv after removing a and b
            const vector<double>& ca = Pji[a];
            const vector<double>& cb = Pji[b];
            for (int c : closed) {
                if (c == a || c == b) continue;
                const vector<double>& cc = Pji[c];
                bool ok = true;
                for (int i = 0; i < N; i++) {
                    double r = recv[i] - ca[i] - cb[i] + cc[i];
                    if (r < req[i] - EPS) { ok = false; break; }
                }
                if (ok) {
                    open[a] = 0; open[b] = 0; open[c] = 1;
                    for (int i = 0; i < N; i++) recv[i] += cc[i] - ca[i] - cb[i];
                    return true;   // one improvement; caller re-runs
                }
            }
        }
    }
    return false;
}

int main() {
    T0 = chrono::steady_clock::now();
    readInput();
    if (N == 0) { printf("0\n"); return 0; }
    precompute();

    vector<double> xf = solveLP_MW();

    // multi-start LP rounding: deterministic order first, then perturbed.
    vector<char> best = buildOnce(xf, 0.0);
    {
        vector<double> rv = recvOf(best);
        prune(best, rv);
    }
    int bestCount = (int)count(best.begin(), best.end(), (char)1);

    int restart = 0;
    while (elapsed() < TIME_LIMIT * 0.55 && restart < 40) {
        restart++;
        double noise = 0.15 + 0.35 * urand();
        vector<char> cand = buildOnce(xf, noise);
        vector<double> rv = recvOf(cand);
        prune(cand, rv);
        int c = (int)count(cand.begin(), cand.end(), (char)1);
        if (c < bestCount) { best = cand; bestCount = c; }
    }

    // local search on the best: alternate 2-for-1 reductions and re-pruning.
    {
        vector<double> recv = recvOf(best);
        while (elapsed() < TIME_LIMIT) {
            bool improved = reduce2for1(best, recv);
            if (!improved) break;
            prune(best, recv);
        }
        // final prune guarantee
        prune(best, recv);
        bestCount = (int)count(best.begin(), best.end(), (char)1);
    }

    // emit
    vector<int> chosen;
    for (int j = 0; j < M; j++) if (best[j]) chosen.push_back(j);
    printf("%d", (int)chosen.size());
    for (int j : chosen) printf(" %d", j);
    printf("\n");
    return 0;
}
