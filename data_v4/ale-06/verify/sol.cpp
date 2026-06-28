// ale-06 "Production Line Scheduling" (permutation flow shop, makespan).
//
// We are given n jobs and m machines in a fixed line. Every job passes through
// all machines in order 0..m-1; we choose ONE job permutation (used on every
// machine -- the permutation flow shop restriction) and minimise the makespan
// Cmax = completion time of the last job on the last machine. This is the
// classic Fm|prmu|Cmax problem: strongly NP-hard for m>=3, judged by a
// continuous score = 1e9 / Cmax (so smaller Cmax => higher score).
//
// Heuristic: the best-known PFSP-makespan family.
//   1) NEH construction: sort jobs by descending total processing time, then
//      insert them one at a time into the best position of the partial order.
//   2) Taillard's acceleration: the makespan of inserting a job in EVERY one of
//      the k+1 positions of a length-k sequence is computed in O(k*m) total via
//      forward "head" (e) and backward "tail" (q) completion-time arrays plus a
//      relative-completion (f) array -- not O(k) separate O(k*m) evaluations.
//      This is exactly the lever the candidate names: the insertion move's cost
//      is evaluated incrementally instead of re-simulating the whole schedule.
//   3) Iterated Greedy (Ruiz & Stuetzle 2007): repeatedly DESTRUCT (remove d
//      random jobs) and RECONSTRUCT (reinsert each greedily with the Taillard
//      accelerator), accept by a constant-temperature Metropolis rule, and keep
//      the best permutation ever seen. An insertion-neighbourhood local search
//      (move each job to its best position, again via the accelerator) polishes
//      after each reconstruction. IG-on-NEH is the established state of the art.
// We always hold a feasible permutation, so whenever we stop we print a legal
// one: any permutation of 0..n-1 is feasible, only the makespan differs.

#include <bits/stdc++.h>
using namespace std;

static inline double now_sec() {
    return (double)chrono::duration_cast<chrono::microseconds>(
               chrono::steady_clock::now().time_since_epoch())
               .count() *
           1e-6;
}

int n, m;
vector<vector<int>> P;  // P[job][machine]

// Full makespan of an explicit permutation (used for the best-snapshot and as
// the ground truth; O(n*m)). Kept separate from the accelerated inserter so the
// two cross-check during debugging.
static long long makespan(const vector<int>& perm) {
    static vector<long long> C;
    C.assign(m, 0);
    for (int i = 0; i < (int)perm.size(); i++) {
        int job = perm[i];
        long long prev = 0;
        for (int k = 0; k < m; k++) {
            long long start = C[k] > prev ? C[k] : prev;
            C[k] = start + P[job][k];
            prev = C[k];
        }
    }
    return C[m - 1];
}

// Taillard's accelerated insertion.
// Given a current partial permutation `seq` (length L) and a job `job` to
// insert, returns the BEST insertion position [0..L] and the resulting makespan.
// Builds:
//   e[i][k] = completion time of seq's i-th job on machine k (head, forward)
//   q[i][k] = "tail": time from the start of seq's i-th job (on machine k) to
//             the end of the schedule, computed backward
//   f[i][k] = completion time of `job` on machine k if inserted at position i
// Cmax if inserted at position i = max_k ( f[i][k] + q[i][k] ).
// Total work O(L*m); returns position and makespan via out-params.
struct Inserter {
    // scratch arrays reused across calls (sized to n*m once).
    vector<long long> e, q, f;  // flattened [pos*m + k]
    void ensure_size() {
        size_t need = (size_t)(n + 1) * m;
        if (e.size() < need) {
            e.assign(need, 0);
            q.assign(need, 0);
            f.assign(need, 0);
        }
    }
    // returns best makespan; sets bestPos to the chosen insertion index.
    long long best_insert(const vector<int>& seq, int job, int& bestPos) {
        int L = (int)seq.size();
        ensure_size();
        auto E = [&](int i, int k) -> long long& { return e[(size_t)i * m + k]; };
        auto Q = [&](int i, int k) -> long long& { return q[(size_t)i * m + k]; };
        auto F = [&](int i, int k) -> long long& { return f[(size_t)i * m + k]; };

        // head e[i][k]: completion of seq[i] on machine k, i in [0, L-1].
        for (int i = 0; i < L; i++) {
            int jb = seq[i];
            for (int k = 0; k < m; k++) {
                long long up = (i > 0) ? E(i - 1, k) : 0;      // same machine, prev job
                long long left = (k > 0) ? E(i, k - 1) : 0;    // same job, prev machine
                E(i, k) = max(up, left) + P[jb][k];
            }
        }
        // tail q[i][k]: time from START of seq[i] on machine k to the end of the
        // tail-subsequence seq[i..L-1]. Computed backward over jobs and
        // machines. q[L][*] = 0 (empty tail). We store q indexed so that
        // Q(i,k) is the tail starting at position i.
        for (int k = 0; k < m; k++) Q(L, k) = 0;
        for (int i = L - 1; i >= 0; i--) {
            int jb = seq[i];
            for (int k = m - 1; k >= 0; k--) {
                long long down = (i + 1 <= L) ? Q(i + 1, k) : 0;   // next job, same machine
                long long right = (k + 1 < m) ? Q(i, k + 1) : 0;   // same job, next machine
                Q(i, k) = max(down, right) + P[jb][k];
            }
        }
        // f[i][k]: completion of `job` on machine k when inserted at position i
        // (i jobs of seq before it). f[i][0] = e[i-1][0] + P[job][0];
        // f[i][k] = max(f[i][k-1], e[i-1][k]) + P[job][k], with e[-1][*] = 0.
        long long best = LLONG_MAX;
        bestPos = 0;
        for (int i = 0; i <= L; i++) {
            for (int k = 0; k < m; k++) {
                long long up = (i > 0) ? E(i - 1, k) : 0;
                long long left = (k > 0) ? F(i, k - 1) : 0;
                F(i, k) = max(up, left) + P[job][k];
            }
            // makespan if inserted here = max over k of f[i][k] + q[i][k]
            long long cm = 0;
            for (int k = 0; k < m; k++) {
                long long v = F(i, k) + Q(i, k);
                if (v > cm) cm = v;
            }
            if (cm < best) {
                best = cm;
                bestPos = i;
            }
        }
        return best;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    const double T_BUDGET = 1.9;  // seconds
    const double t_start = now_sec();

    if (!(cin >> n >> m)) return 0;
    if (n <= 0) {
        // Nothing to schedule; an empty permutation is the only feasible answer.
        return 0;
    }
    if (m <= 0) m = 0;  // defensive; makespan of 0 machines is 0
    P.assign(n, vector<int>(max(m, 0)));
    for (int j = 0; j < n; j++)
        for (int k = 0; k < m; k++) cin >> P[j][k];

    // Trivial degenerate guards: always emit a feasible permutation.
    if (n == 1) {
        cout << 0 << "\n";
        return 0;
    }
    if (m == 0) {
        // Makespan is 0 for any order; print identity.
        for (int j = 0; j < n; j++) cout << j << (j + 1 < n ? ' ' : '\n');
        return 0;
    }

    // Deterministic RNG seeded from the instance (reproducible).
    uint64_t rng_state = 0x9e3779b97f4a7c15ULL ^ ((uint64_t)n << 32) ^ (uint64_t)m;
    for (int j = 0; j < n; j++)
        for (int k = 0; k < m; k++) {
            rng_state ^= (uint64_t)(P[j][k] + 1) * 0x9e3779b97f4a7c15ULL;
            rng_state ^= rng_state >> 29;
            rng_state *= 0xbf58476d1ce4e5b9ULL;
            rng_state ^= rng_state >> 32;
        }
    auto nextu = [&]() -> uint64_t {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        return rng_state;
    };
    auto nextd = [&]() -> double {
        return (double)(nextu() >> 11) / 9007199254740992.0;  // [0,1)
    };
    auto randint = [&](int lo, int hi) -> int {  // inclusive
        return lo + (int)(nextu() % (uint64_t)(hi - lo + 1));
    };

    Inserter ins;

    // ---- NEH construction -------------------------------------------------
    // Order jobs by descending total processing time across machines, then
    // insert one by one at the best position via the accelerator.
    vector<long long> tot(n, 0);
    for (int j = 0; j < n; j++)
        for (int k = 0; k < m; k++) tot[j] += P[j][k];
    vector<int> order(n);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(),
         [&](int a, int b) { return tot[a] > tot[b]; });

    vector<int> perm;
    perm.reserve(n);
    perm.push_back(order[0]);
    for (int idx = 1; idx < n; idx++) {
        int job = order[idx];
        int pos;
        ins.best_insert(perm, job, pos);
        perm.insert(perm.begin() + pos, job);
    }

    long long curCmax = makespan(perm);
    vector<int> best = perm;
    long long bestCmax = curCmax;

    // ---- insertion-neighbourhood local search -----------------------------
    // For each job, remove it and reinsert it at its best position (accelerated).
    // Repeat full sweeps until a sweep makes no improvement. Each job's best
    // reinsertion is O(n*m); a sweep is O(n^2*m).
    auto local_search = [&](vector<int>& seq, long long& cm) {
        bool improved = true;
        while (improved) {
            improved = false;
            if (now_sec() - t_start > T_BUDGET) return;
            for (int t = 0; t < n; t++) {
                if ((t & 15) == 0 && now_sec() - t_start > T_BUDGET) return;
                int job = seq[t];
                // remove job at position t
                seq.erase(seq.begin() + t);
                int pos;
                long long newcm = ins.best_insert(seq, job, pos);
                seq.insert(seq.begin() + pos, job);
                if (newcm + 0 < cm) {
                    cm = newcm;
                    improved = true;
                }
            }
        }
    };
    local_search(perm, curCmax);
    if (curCmax < bestCmax) {
        bestCmax = curCmax;
        best = perm;
    }

    // ---- Iterated Greedy (Ruiz & Stuetzle) --------------------------------
    // Constant-temperature acceptance. Temperature scaled by the average
    // processing time, the standard IG setting:
    //   T = lambda * (sum of all p) / (n * m * 10).
    long long sumP = 0;
    for (int j = 0; j < n; j++)
        for (int k = 0; k < m; k++) sumP += P[j][k];
    const double lambda = 0.4;
    double Temp = lambda * (double)sumP / (double)(n * m * 10);
    if (Temp < 1e-9) Temp = 1.0;

    int d = max(2, min(4, n - 1));  // number of jobs to destruct each iteration

    vector<int> cur = perm;       // working permutation (accepted)
    long long workCmax = curCmax;

    vector<int> removed;
    removed.reserve(d);

    long long iters = 0;
    while (now_sec() - t_start <= T_BUDGET) {
        iters++;
        // ---- destruction: remove d distinct random jobs ----
        vector<int> cand = cur;
        removed.clear();
        for (int r = 0; r < d && (int)cand.size() > 1; r++) {
            int idx = randint(0, (int)cand.size() - 1);
            removed.push_back(cand[idx]);
            cand.erase(cand.begin() + idx);
        }
        // ---- construction: greedily reinsert each removed job ----
        long long candCmax = 0;
        for (int job : removed) {
            int pos;
            candCmax = ins.best_insert(cand, job, pos);
            cand.insert(cand.begin() + pos, job);
        }
        // ---- local search polish ----
        local_search(cand, candCmax);

        // ---- acceptance (Metropolis with constant temperature) ----
        if (candCmax <= workCmax) {
            cur = cand;
            workCmax = candCmax;
            if (candCmax < bestCmax) {
                bestCmax = candCmax;
                best = cand;
            }
        } else {
            double delta = (double)(candCmax - workCmax);
            if (nextd() < exp(-delta / Temp)) {
                cur = cand;
                workCmax = candCmax;
            }
        }
    }

    // ---- emit feasible solution -------------------------------------------
    // `best` is a permutation of 0..n-1; print it space-separated.
    // (The scorer also accepts a leading header equal to n; we omit it.)
    for (int i = 0; i < n; i++) cout << best[i] << (i + 1 < n ? ' ' : '\n');
    return 0;
}
