// Facility Layout Assignment (Quadratic Assignment Problem) -- heuristic solver.
//
// Objective: given an n x n flow matrix F and an n x n distance matrix D, find a
// permutation p (facility i placed on location p[i]) minimizing
//     C(p) = sum_{i,j} F[i][j] * D[p[i]][p[j]].
// Read the instance from stdin, write the permutation p[0..n-1] (one index per
// line) to stdout. Lower cost is better.
//
// Method (the innovation): Robust Tabu Search (Taillard) with an O(n)
// incremental swap-delta and an O(1) delta-table update between consecutive
// swaps -- the established strong heuristic for QAP.
//   1. Maintain a delta table delta[r][s] = exact cost change of swapping the
//      facilities currently at positions r and s. The first build is O(n^3),
//      then it is kept up to date incrementally.
//   2. Each iteration scans all O(n^2) pairs, picks the best admissible swap
//      (not tabu, or tabu but better than the incumbent -- aspiration), applies
//      it, and updates the delta table: a swap of (u,v) touches only rows/cols u
//      and v, so a pair (r,s) disjoint from {u,v} updates in O(1); pairs meeting
//      {u,v} are recomputed in O(n). One move therefore costs O(n^2).
//   3. Tabu tenure is randomized in a band around n (robust tabu); whenever the
//      incumbent has not improved for a long stretch the search diversifies.
// The current permutation is always valid, so any early stop (including hitting
// the wall-clock budget mid-scan) still prints a feasible solution.
#include <bits/stdc++.h>
using namespace std;

static inline double now_sec() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    uint64_t next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return s;
    }
    uint32_t nextu(uint32_t m) { return (uint32_t)(next() % m); }  // [0, m)
    double nextd() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

int N;
vector<long long> F, D;   // row-major n x n matrices
static inline long long Fm(int i, int j) { return F[(size_t)i * N + j]; }
static inline long long Dm(int i, int j) { return D[(size_t)i * N + j]; }

int main() {
    const double T0 = now_sec();
    const double TIME_LIMIT = 1.85;  // wall-clock budget (seconds)

    if (scanf("%d", &N) != 1) return 0;
    if (N <= 0) return 0;
    F.assign((size_t)N * N, 0);
    D.assign((size_t)N * N, 0);
    for (size_t k = 0; k < (size_t)N * N; k++) {
        long long v;
        if (scanf("%lld", &v) != 1) v = 0;
        F[k] = v;
    }
    for (size_t k = 0; k < (size_t)N * N; k++) {
        long long v;
        if (scanf("%lld", &v) != 1) v = 0;
        D[k] = v;
    }
    if (N == 1) { printf("0\n"); return 0; }

    Rng rng(0xC0FFEEULL ^ (uint64_t)N * 1000003ULL);

    // p[i] = location of facility i  (the permutation we output)
    vector<int> p(N);
    for (int i = 0; i < N; i++) p[i] = i;

    auto cost = [&](const vector<int> &perm) -> long long {
        long long c = 0;
        for (int i = 0; i < N; i++) {
            int pi = perm[i];
            for (int j = 0; j < N; j++) {
                long long f = Fm(i, j);
                if (f) c += f * Dm(pi, perm[j]);
            }
        }
        return c;
    };

    // ---- exact delta of swapping the facilities at positions r and s ----
    // We work in the "position space": swapping positions r and s exchanges the
    // locations p[r] and p[s]. The standard Taillard O(n) formula for the cost
    // change C(p') - C(p), valid for arbitrary (not necessarily symmetric) F,D:
    auto deltaFull = [&](const vector<int> &perm, int r, int s) -> long long {
        int pr = perm[r], ps = perm[s];
        long long d = (Fm(r, r) - Fm(s, s)) * (Dm(ps, ps) - Dm(pr, pr))
                    + (Fm(r, s) - Fm(s, r)) * (Dm(ps, pr) - Dm(pr, ps));
        for (int k = 0; k < N; k++) {
            if (k == r || k == s) continue;
            int pk = perm[k];
            d += (Fm(k, r) - Fm(k, s)) * (Dm(pk, ps) - Dm(pk, pr))
               + (Fm(r, k) - Fm(s, k)) * (Dm(ps, pk) - Dm(pr, pk));
        }
        return d;
    };

    // ---- O(1) update of delta after a swap, when the new pair (r,s) is
    // disjoint from the just-applied pair (u,v) (Taillard's fast update). It
    // needs the value of delta BEFORE the swap was applied. ----
    auto deltaFast = [&](const vector<int> &perm, int r, int s, int u, int v,
                         long long prev) -> long long {
        // perm here is AFTER the (u,v) swap has been applied.
        int pr = perm[r], ps = perm[s];
        int pu = perm[u], pv = perm[v];
        return prev
             + (Fm(r, u) - Fm(r, v) + Fm(s, v) - Fm(s, u))
                 * (Dm(ps, pu) - Dm(ps, pv) + Dm(pr, pv) - Dm(pr, pu))
             + (Fm(u, r) - Fm(v, r) + Fm(v, s) - Fm(u, s))
                 * (Dm(pu, ps) - Dm(pv, ps) + Dm(pv, pr) - Dm(pu, pr));
    };

    // delta[r][s] for r<s ; stored full n x n for simple indexing (r<s used)
    vector<long long> delta((size_t)N * N, 0);
    auto Dl = [&](int r, int s) -> long long& { return delta[(size_t)r * N + s]; };

    for (int r = 0; r < N; r++)
        for (int s = r + 1; s < N; s++)
            Dl(r, s) = deltaFull(p, r, s);

    long long curCost = cost(p);

    // best solution found so far (always a valid permutation)
    vector<int> best = p;
    long long bestCost = curCost;

    // tabu[r][s] = iteration until which swapping positions r,s is forbidden
    vector<long long> tabu((size_t)N * N, 0);
    auto Tb = [&](int r, int s) -> long long& { return tabu[(size_t)r * N + s]; };

    long long iter = 0;
    // robust-tabu tenure band: a randomized tenure around the problem size
    int tenureLo = max(2, N / 2);
    int tenureHi = max(tenureLo + 1, (3 * N) / 2);

    long long lastImprove = 0;
    long long stagnationLimit = (long long)N * N + 100;  // diversify if stuck

    long long clk = 0;
    auto timeUp = [&]() {
        if ((++clk & 7) == 0) return now_sec() - T0 > TIME_LIMIT;
        return false;
    };

    int lastU = -1, lastV = -1;  // last applied swap, for the fast update path

    while (!timeUp()) {
        iter++;
        // ---- scan all pairs, pick the best admissible swap ----
        long long bestDelta = LLONG_MAX;
        int br = -1, bs = -1;
        bool anyAdmissible = false;
        for (int r = 0; r < N; r++) {
            for (int s = r + 1; s < N; s++) {
                long long dl = Dl(r, s);
                bool isTabu = Tb(r, s) > iter;
                // aspiration: a tabu move is allowed if it would beat the incumbent
                bool aspire = (curCost + dl < bestCost);
                if (isTabu && !aspire) continue;
                anyAdmissible = true;
                if (dl < bestDelta) { bestDelta = dl; br = r; bs = s; }
            }
        }
        if (!anyAdmissible) {
            // everything tabu: pick the globally smallest delta regardless
            for (int r = 0; r < N; r++)
                for (int s = r + 1; s < N; s++) {
                    long long dl = Dl(r, s);
                    if (dl < bestDelta) { bestDelta = dl; br = r; bs = s; }
                }
        }
        if (br < 0) break;  // degenerate; nothing to do

        // ---- apply the swap of positions br,bs ----
        int u = br, v = bs;
        // update delta table BEFORE mutating p where the fast path needs old vals:
        // strategy -- first apply swap to p, then refresh deltas.
        // Save pre-swap deltas for the fast-update formula.
        // We refresh in two phases:
        //   (1) for pairs disjoint from {u,v}: O(1) fast update using prev delta;
        //   (2) for pairs meeting {u,v}: O(n) full recompute.
        // The fast update needs delta value BEFORE this swap, which is exactly
        // the current Dl(r,s) (it has not been touched yet this iteration).

        // First, perform the swap on p.
        swap(p[u], p[v]);
        curCost += bestDelta;

        // (1) O(1) fast updates for pairs entirely outside {u,v}.
        for (int r = 0; r < N; r++) {
            if (r == u || r == v) continue;
            for (int s = r + 1; s < N; s++) {
                if (s == u || s == v) continue;
                long long prev = Dl(r, s);
                Dl(r, s) = deltaFast(p, r, s, u, v, prev);
            }
        }
        // (2) O(n) full recompute for every pair that touches u or v.
        for (int s = 0; s < N; s++) {
            if (s == u) continue;
            int a = min(u, s), b = max(u, s);
            Dl(a, b) = deltaFull(p, a, b);
        }
        for (int s = 0; s < N; s++) {
            if (s == v) continue;
            int a = min(v, s), b = max(v, s);
            Dl(a, b) = deltaFull(p, a, b);
        }

        // ---- set tabu tenure for the reverse move ----
        int tenure = tenureLo + (int)rng.nextu((uint32_t)(tenureHi - tenureLo));
        Tb(u, v) = iter + tenure;

        // ---- track incumbent ----
        if (curCost < bestCost) {
            bestCost = curCost;
            best = p;
            lastImprove = iter;
        }
        lastU = u; lastV = v;
        (void)lastU; (void)lastV;

        // ---- diversification when stuck: a few random forced swaps ----
        if (iter - lastImprove > stagnationLimit) {
            int kicks = 2 + (int)rng.nextu((uint32_t)max(1, N / 10));
            for (int t = 0; t < kicks; t++) {
                int a = (int)rng.nextu((uint32_t)N);
                int b = (int)rng.nextu((uint32_t)N);
                if (a == b) continue;
                if (a > b) swap(a, b);
                long long dl = Dl(a, b);
                swap(p[a], p[b]);
                curCost += dl;
                // refresh deltas after this forced swap (same two-phase scheme)
                for (int r = 0; r < N; r++) {
                    if (r == a || r == b) continue;
                    for (int s = r + 1; s < N; s++) {
                        if (s == a || s == b) continue;
                        long long prev = Dl(r, s);
                        Dl(r, s) = deltaFast(p, r, s, a, b, prev);
                    }
                }
                for (int s = 0; s < N; s++) {
                    if (s == a) continue;
                    int x = min(a, s), y = max(a, s);
                    Dl(x, y) = deltaFull(p, x, y);
                }
                for (int s = 0; s < N; s++) {
                    if (s == b) continue;
                    int x = min(b, s), y = max(b, s);
                    Dl(x, y) = deltaFull(p, x, y);
                }
            }
            if (curCost < bestCost) { bestCost = curCost; best = p; }
            lastImprove = iter;
        }
    }

    // ---- output the best permutation found (always valid) ----
    {
        vector<char> seen(N, 0);
        bool ok = true;
        for (int i = 0; i < N; i++) {
            int loc = best[i];
            if (loc < 0 || loc >= N || seen[loc]) { ok = false; break; }
            seen[loc] = 1;
        }
        if (!ok) for (int i = 0; i < N; i++) best[i] = i;
    }
    string out; out.reserve((size_t)N * 7);
    char buf[16];
    for (int i = 0; i < N; i++) {
        int len = snprintf(buf, sizeof(buf), "%d\n", best[i]);
        out.append(buf, len);
    }
    fputs(out.c_str(), stdout);
    return 0;
}
