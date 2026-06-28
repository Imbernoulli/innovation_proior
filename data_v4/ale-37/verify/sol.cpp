// Quadratic Assignment Placement (QAP) -- place n facilities on n locations
// (a permutation p, facility i -> location p[i]) minimizing the quadratic cost
//   cost(p) = sum_i sum_j f[i][j] * d[p[i]][p[j]].
// Read the instance (n, then the n x n flow matrix f, then the n x n distance
// matrix d) from stdin; write a permutation (n locations, p[0..n-1]) to stdout.
//
// Method (the innovation): ROBUST TABU SEARCH on the 2-swap neighbourhood with
// Taillard's incremental O(1) swap-delta recurrence.
//
//   1. FEASIBLE BASELINE. Start from the identity permutation p[i] = i. It is a
//      valid permutation, so we always hold a feasible answer; we also keep the
//      best permutation seen and print THAT, so any time cutoff is still valid.
//
//   2. 2-SWAP NEIGHBOURHOOD. The move is "swap the locations of two facilities
//      r and s". This keeps p a permutation for free -- swapping two entries of
//      a permutation is always a permutation -- so feasibility is invariant.
//
//   3. CLOSED-FORM SWAP DELTA. The change in cost from swapping facilities r,s
//      is a single O(n) sum over the other facilities (the standard QAP delta
//      for symmetric matrices). Evaluating one move is O(n), not O(n^2).
//
//   4. TAILLARD'S INCREMENTAL DELTA MATRIX (the lever). We store delta[r][s] for
//      every pair. After performing a swap of (u,v), every delta[r][s] with
//      {r,s} disjoint from {u,v} updates in O(1) via a closed-form recurrence;
//      only the deltas that involve u or v are recomputed from scratch in O(n).
//      So a full neighbourhood scan after a move costs O(n^2) with a tiny
//      constant instead of O(n^3). This is what makes thousands of sweeps / sec
//      feasible and lets tabu search escape the deep local minima QAP is
//      infamous for.
//
//   5. ROBUST TABU + ASPIRATION. A move (facility,location) becomes tabu for a
//      randomized tenure (Taillard's "robust" range); a tabu move is allowed
//      only if it would beat the best cost ever seen (aspiration). We also force
//      a long-unused move occasionally to diversify. The best permutation found
//      is what we print.
#include <bits/stdc++.h>
using namespace std;

static inline double now_sec() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(
               steady_clock::now().time_since_epoch())
        .count();
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 0x9e3779b97f4a7c15ULL) {}
    inline uint64_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
    inline uint32_t u32() { return (uint32_t)(next() >> 32); }
    inline int below(int n) { return (int)(u32() % (uint32_t)n); }
    inline double unit() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

int n;
vector<vector<long long>> f;   // flow matrix, n x n
vector<vector<long long>> d;   // distance matrix, n x n

// full cost of permutation p (p[i] = location of facility i)
static long long full_cost(const vector<int> &p) {
    long long c = 0;
    for (int i = 0; i < n; i++) {
        const auto &fi = f[i];
        const auto &dpi = d[p[i]];
        for (int j = 0; j < n; j++) {
            long long fij = fi[j];
            if (fij) c += fij * dpi[p[j]];
        }
    }
    return c;
}

// delta of swapping facilities r and s in permutation p (general, O(n)).
// Works for arbitrary (not necessarily symmetric) f and d.
static long long swap_delta(const vector<int> &p, int r, int s) {
    int pr = p[r], ps = p[s];
    long long dlt = 0;
    // terms that change are those touching facility r or s.
    // delta = sum over k of contributions where one index is r or s.
    // Use the standard four-index expansion (handles asymmetry too).
    dlt += (f[r][r]) * (d[ps][ps] - d[pr][pr]);
    dlt += (f[s][s]) * (d[pr][pr] - d[ps][ps]);
    dlt += (f[r][s]) * (d[ps][pr] - d[pr][ps]);
    dlt += (f[s][r]) * (d[pr][ps] - d[ps][pr]);
    for (int k = 0; k < n; k++) {
        if (k == r || k == s) continue;
        int pk = p[k];
        dlt += f[r][k] * (d[ps][pk] - d[pr][pk]);
        dlt += f[k][r] * (d[pk][ps] - d[pk][pr]);
        dlt += f[s][k] * (d[pr][pk] - d[ps][pk]);
        dlt += f[k][s] * (d[pk][pr] - d[pk][ps]);
    }
    return dlt;
}

// Taillard's O(1) update of delta[r][s] AFTER a swap of facilities (u,v) was
// performed on p, given the value BEFORE that swap. Valid when {r,s} disjoint
// from {u,v}. p here is the permutation AFTER the (u,v) swap.
static inline long long update_delta(const vector<int> &p, long long prev,
                                     int r, int s, int u, int v) {
    int pr = p[r], ps = p[s], pu = p[u], pv = p[v];
    return prev
        + (f[r][u] - f[r][v] + f[s][v] - f[s][u]) *
              (d[ps][pu] - d[ps][pv] + d[pr][pv] - d[pr][pu])
        + (f[u][r] - f[v][r] + f[v][s] - f[u][s]) *
              (d[pu][ps] - d[pv][ps] + d[pv][pr] - d[pu][pr]);
}

int main() {
    double t0 = now_sec();
    const double TIME_LIMIT = 1.7;  // wall-clock budget (s); see context.md

    if (scanf("%d", &n) != 1) return 0;
    if (n <= 0) { printf("\n"); return 0; }
    f.assign(n, vector<long long>(n));
    d.assign(n, vector<long long>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) scanf("%lld", &f[i][j]);
    for (int k = 0; k < n; k++)
        for (int l = 0; l < n; l++) scanf("%lld", &d[k][l]);

    // n == 1: the only permutation is [0].
    if (n == 1) { printf("0\n"); return 0; }

    Rng rng(0x9E3779B97F4A7C15ULL ^ ((uint64_t)n << 32));

    // ----- feasible baseline: identity permutation -----
    vector<int> p(n);
    for (int i = 0; i < n; i++) p[i] = i;
    long long cur = full_cost(p);

    vector<int> best = p;
    long long best_cost = cur;

    // ----- delta matrix: delta[i][j] for i<j -----
    vector<vector<long long>> delta(n, vector<long long>(n, 0));
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            delta[i][j] = swap_delta(p, i, j);

    // ----- tabu structure: tabu[facility][location] = iteration until which the
    // (facility,location) pair is forbidden -----
    vector<vector<long long>> tabu(n, vector<long long>(n, 0));

    // robust tabu tenure range (Taillard): around n, randomized each move.
    long long t_min = (long long)(0.9 * n) + 1;
    long long t_max = (long long)(1.1 * n) + 1;
    if (t_min < 2) t_min = 2;
    if (t_max <= t_min) t_max = t_min + 1;

    // aspiration horizon: a move is forced if it has not been used for a very
    // long time (diversification). Taillard uses ~ a few * n^2 iterations.
    long long forced_horizon = (long long)9 * n * n + 10;

    long long iter = 0;
    int check_mask = 1023;  // check the clock every 1024 iterations

    while (true) {
        if ((iter & check_mask) == 0) {
            if (now_sec() - t0 > TIME_LIMIT) break;
        }
        iter++;

        // ----- scan the neighbourhood for the best admissible swap -----
        long long best_move_delta = LLONG_MAX;
        int br = -1, bs = -1;
        bool any = false;

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                long long dl = delta[i][j];
                // a move swaps facility i to location p[j] and facility j to
                // location p[i]; it is tabu if BOTH (i, p[j]) and (j, p[i])
                // are currently forbidden.
                bool is_tabu =
                    (tabu[i][p[j]] >= iter) && (tabu[j][p[i]] >= iter);
                bool forced =
                    (tabu[i][p[j]] < iter - forced_horizon) ||
                    (tabu[j][p[i]] < iter - forced_horizon);
                bool aspire = (cur + dl < best_cost);

                if ((!is_tabu) || aspire || forced) {
                    // forced moves are always considered; break ties by delta
                    long long eff = dl;
                    if (forced && !aspire) {
                        // bias forced moves to be chosen even if not improving,
                        // but only when nothing better exists: emulate by
                        // treating their delta as is, and remembering we saw a
                        // legal move. We still pick the min-delta admissible.
                    }
                    if (!any || eff < best_move_delta) {
                        best_move_delta = eff;
                        br = i;
                        bs = j;
                        any = true;
                    }
                }
            }
        }

        if (!any) {
            // every move tabu and none aspires/forced: pick the globally min
            // delta ignoring tabu (guarantees progress / a legal move).
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                    if (br < 0 || delta[i][j] < best_move_delta) {
                        best_move_delta = delta[i][j];
                        br = i;
                        bs = j;
                    }
        }

        int u = br, v = bs;          // facilities being swapped
        int pu = p[u], pv = p[v];    // their locations before the swap

        // ----- perform the swap -----
        swap(p[u], p[v]);
        cur += best_move_delta;

        // ----- set tabu: forbid returning facility u to location pu and
        // facility v to location pv for a randomized tenure -----
        long long tenure = t_min + rng.below((int)(t_max - t_min));
        tabu[u][pu] = iter + tenure;
        tabu[v][pv] = iter + tenure;

        // ----- incremental delta-matrix update (Taillard's recurrence) -----
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (i == u || i == v || j == u || j == v) {
                    // pairs touching u or v: recompute from scratch, O(n)
                    delta[i][j] = swap_delta(p, i, j);
                } else {
                    // disjoint pairs: O(1) update
                    delta[i][j] = update_delta(p, delta[i][j], i, j, u, v);
                }
            }
        }

        // ----- track best -----
        if (cur < best_cost) {
            best_cost = cur;
            best = p;
        }
    }

    // ----- emit the best permutation found (always a valid permutation) -----
    string out;
    out.reserve(n * 4);
    for (int i = 0; i < n; i++) {
        out += to_string(best[i]);
        out += (i + 1 == n) ? '\n' : ' ';
    }
    fputs(out.c_str(), stdout);
    return 0;
}
