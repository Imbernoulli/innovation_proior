// Machine Assignment with Sequence-Dependent Setups -- heuristic solver.
//
// Objective. n jobs, M identical parallel machines, T job types. Job j has a
// duration d[j] and a type c[j]. Each machine runs the jobs assigned to it in
// some order; before its first job (type a) it pays init[a], and between two
// consecutive jobs of types a then b it pays the setup s[a][b]. The score is
// normalized against a round-robin baseline; we MINIMIZE the TOTAL over all
// machines of (sum of durations + all setups). We read the instance from stdin
// and print, per machine, an ordered job list:
//     M lines, line m = "k j_1 j_2 ... j_k".
//
// Method (the innovation).
//   The sum of durations is a CONSTANT (every job runs exactly once, on some
//   machine), so minimizing total cost == minimizing total SETUP cost. Setup
//   only depends on TYPES and adjacency, so each machine's contribution is a
//   path through its jobs' types starting from a virtual "ground" type whose
//   out-edge to type a costs init[a]: i.e. an OPEN ASYMMETRIC TSP-on-types per
//   machine, embedded inside an assignment of jobs to machines.
//
//   We run simulated annealing with two coupled, O(1)-delta neighborhoods on
//   the explicit per-machine sequences:
//     * RELOCATE (inter- or intra-machine): pull job out of its current slot
//       (delete edges prev->job, job->next, add edge prev->next) and insert it
//       before some position on a (possibly different) machine (delete edge
//       p->q there, add p->job, job->q). The delta touches only the 2 donor
//       neighbors and the 2 receiver neighbors -> O(1) from cached neighbor
//       types.
//     * SWAP two jobs (each may be on a different machine): re-evaluate only the
//       <=6 incident setup edges -> O(1).
//   Because durations are constant we anneal purely on the cheap setup delta,
//   so a single iteration is O(1) regardless of n, and we do millions of them.
//   We always hold a valid assignment (a partition of all jobs), so any early
//   stop still prints a feasible schedule.
#include <bits/stdc++.h>
using namespace std;

static inline double now_sec() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    uint32_t nextu(uint32_t m) { return (uint32_t)(next() % m); }
    double nextd() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

int N, M, T;
vector<int> D, C;                 // duration, type per job
vector<int> INIT;                 // init[type]
vector<int> S;                    // flattened T x T setup matrix
inline int setup(int a, int b) { return S[a * T + b]; }

// Explicit schedule as per-machine sequences plus reverse lookup.
vector<vector<int>> seqv;         // seqv[m] = ordered job ids on machine m
vector<int> mac;                  // mac[j]  = machine of job j
vector<int> pos;                  // pos[j]  = index of j within seqv[mac[j]]

// Cost of the setup edge entering job at position p on machine m (the edge
// from its predecessor, or the init edge if it is first). Used for deltas.
inline int enterCost(int m, int p) {
    int j = seqv[m][p];
    if (p == 0) return INIT[C[j]];
    return setup(C[seqv[m][p - 1]], C[j]);
}

// Total setup cost of a machine (init + all changeovers). O(len). For init/check.
long long machineSetup(int m) {
    const auto& q = seqv[m];
    if (q.empty()) return 0;
    long long t = INIT[C[q[0]]];
    for (size_t i = 1; i < q.size(); i++) t += setup(C[q[i - 1]], C[q[i]]);
    return t;
}

int main() {
    const double T0 = now_sec();
    const double TIME_LIMIT = 1.9;

    if (scanf("%d %d %d", &N, &M, &T) != 3) return 0;
    if (N <= 0) {                 // degenerate: print M empty machines
        for (int m = 0; m < M; m++) printf("0\n");
        return 0;
    }
    if (M <= 0) M = 1;
    D.resize(N); C.resize(N);
    for (int j = 0; j < N; j++) if (scanf("%d", &D[j]) != 1) D[j] = 0;
    for (int j = 0; j < N; j++) {
        if (scanf("%d", &C[j]) != 1) C[j] = 0;
        if (C[j] < 0 || C[j] >= T) C[j] = 0;   // clamp for safety
    }
    INIT.assign(T, 0);
    for (int a = 0; a < T; a++) if (scanf("%d", &INIT[a]) != 1) INIT[a] = 0;
    S.assign((size_t)T * T, 0);
    for (int a = 0; a < T; a++)
        for (int b = 0; b < T; b++)
            if (scanf("%d", &S[a * T + b]) != 1) S[a * T + b] = 0;

    Rng rng(0x9E3779B1ull ^ ((uint64_t)N << 32) ^ ((uint64_t)M << 16) ^ (uint64_t)T);

    // ----- Constructive start: greedy type-aware assignment. -----------------
    // Group jobs by type, then hand whole type-runs to the machine that is
    // currently cheapest to extend. This already clusters same-type jobs (cheap
    // changeovers) and beats round-robin handily; SA refines from here.
    seqv.assign(M, {});
    mac.assign(N, 0);
    pos.assign(N, 0);

    // jobs bucketed by type
    vector<vector<int>> byType(T);
    for (int j = 0; j < N; j++) byType[C[j]].push_back(j);
    // process types in descending multiplicity so big runs are placed first
    vector<int> typeOrder(T);
    iota(typeOrder.begin(), typeOrder.end(), 0);
    sort(typeOrder.begin(), typeOrder.end(),
         [&](int a, int b) { return byType[a].size() > byType[b].size(); });

    // last type on each machine (for changeover estimate); -1 == empty (init)
    vector<int> lastType(M, -1);
    for (int ti : typeOrder) {
        for (int j : byType[ti]) {
            // pick the machine cheapest to append this job to right now:
            // entering cost (init or changeover from lastType) is the lever;
            // tie-break toward the machine with fewer jobs to keep balance.
            int bestM = 0; long long bestKey = LLONG_MAX;
            for (int m = 0; m < M; m++) {
                long long enter = (lastType[m] < 0) ? INIT[ti]
                                                    : setup(lastType[m], ti);
                long long key = enter * 1000000LL + (long long)seqv[m].size();
                if (key < bestKey) { bestKey = key; bestM = m; }
            }
            seqv[bestM].push_back(j);
            lastType[bestM] = ti;
        }
    }
    // refresh reverse lookup
    auto reindex = [&]() {
        for (int m = 0; m < M; m++)
            for (int p = 0; p < (int)seqv[m].size(); p++) {
                mac[seqv[m][p]] = m; pos[seqv[m][p]] = p;
            }
    };
    reindex();

    // current total SETUP cost (durations are constant; we anneal on setups)
    long long curSetup = 0;
    for (int m = 0; m < M; m++) curSetup += machineSetup(m);
    long long bestSetup = curSetup;
    vector<vector<int>> bestSeq = seqv;

    // ----- Simulated annealing on the two O(1) neighborhoods. ---------------
    // Cost scale: a single changeover is ~tens-to-hundreds, so start the
    // temperature near a typical large changeover and cool geometrically.
    double Tstart = 120.0, Tend = 0.05;
    long long iters = 0;

    // helper: setup contribution of the edge ENTERING position p on machine m
    // and the edge LEAVING position p (to p+1), used to localize deltas.
    auto leaveCost = [&](int m, int p) -> int {
        const auto& q = seqv[m];
        if (p + 1 >= (int)q.size()) return 0;   // last job: no leaving edge
        return setup(C[q[p]], C[q[p + 1]]);
    };

    while (true) {
        if ((iters & 1023) == 0) {
            if (now_sec() - T0 > TIME_LIMIT) break;
        }
        iters++;
        double frac = std::min(1.0, (now_sec() - T0) / TIME_LIMIT);
        double Temp = Tstart * pow(Tend / Tstart, frac);

        int mv = rng.nextu(2);

        if (mv == 0) {
            // ---------- RELOCATE: move one job to another slot --------------
            int j = rng.nextu(N);
            int fm = mac[j], fp = pos[j];
            int Lf = (int)seqv[fm].size();
            // removal delta on the donor: we drop edges enter(fp) and leave(fp)
            // and, if j had both a predecessor and successor, add prev->next.
            int prevT = (fp > 0) ? C[seqv[fm][fp - 1]] : -1;
            int nextT = (fp + 1 < Lf) ? C[seqv[fm][fp + 1]] : -1;
            int cj = C[j];
            long long removeGain;
            {
                int enter = (prevT < 0) ? INIT[cj] : setup(prevT, cj);
                int leave = (nextT < 0) ? 0 : setup(cj, nextT);
                int bridge = 0;
                if (prevT < 0 && nextT >= 0) bridge = INIT[nextT];     // next becomes first
                else if (prevT >= 0 && nextT >= 0) bridge = setup(prevT, nextT);
                removeGain = (long long)enter + leave - bridge;        // saved by removing j
            }

            // choose a target machine and an insertion position in [0, len]
            int tm = rng.nextu(M);
            int Lt = (int)seqv[tm].size();
            // if inserting back into the same machine, length excludes j's slot
            // effectively; handle via a temporary "logical" view: we compute the
            // delta as if j were already removed.
            // Build the target sequence reference excluding j when tm == fm.
            // To keep O(1), we sample an insertion index over the post-removal
            // length and read neighbor types directly, skipping j's old slot.
            int targetLen = Lt - (tm == fm ? 1 : 0);
            int ip = (targetLen <= 0) ? 0 : (int)rng.nextu(targetLen + 1);

            // Resolve predecessor/successor types at insertion point ip in the
            // post-removal target sequence.
            auto jobAt = [&](int m, int idx) -> int {
                // index into machine m's sequence AFTER logically removing j if m==fm.
                if (m == fm) {
                    if (idx < fp) return seqv[m][idx];
                    return seqv[m][idx + 1];   // skip j's slot
                }
                return seqv[m][idx];
            };
            int pT = (ip > 0) ? C[jobAt(tm, ip - 1)] : -1;
            int sT = (ip < targetLen) ? C[jobAt(tm, ip)] : -1;
            long long insertCost;
            {
                int enter = (pT < 0) ? INIT[cj] : setup(pT, cj);
                int leave = (sT < 0) ? 0 : setup(cj, sT);
                int oldEdge = 0;
                if (pT < 0 && sT >= 0) oldEdge = INIT[sT];         // sT was first
                else if (pT >= 0 && sT >= 0) oldEdge = setup(pT, sT);
                insertCost = (long long)enter + leave - oldEdge;   // added by inserting j
            }

            long long delta = insertCost - removeGain;
            // Metropolis acceptance on the localized setup delta.
            if (delta <= 0 || rng.nextd() < exp(-(double)delta / Temp)) {
                // apply: erase j from the donor first. After this erase BOTH the
                // donor and (if same machine) the target are in the post-removal
                // frame -- exactly the frame `ip` and jobAt() were computed in --
                // so we insert at `ip` directly in every case.
                seqv[fm].erase(seqv[fm].begin() + fp);
                seqv[tm].insert(seqv[tm].begin() + ip, j);
                // reindex the (one or two) affected machines only.
                for (int p = 0; p < (int)seqv[fm].size(); p++) { mac[seqv[fm][p]] = fm; pos[seqv[fm][p]] = p; }
                if (tm != fm)
                    for (int p = 0; p < (int)seqv[tm].size(); p++) { mac[seqv[tm][p]] = tm; pos[seqv[tm][p]] = p; }
                curSetup += delta;
                if (curSetup < bestSetup) { bestSetup = curSetup; bestSeq = seqv; }
            }
        } else {
            // ---------- SWAP two jobs (positions exchanged) -----------------
            if (N < 2) continue;
            int a = rng.nextu(N), b = rng.nextu(N);
            if (a == b) continue;
            int ma = mac[a], pa = pos[a];
            int mb = mac[b], pb = pos[b];
            if (ma == mb && (pa == pb)) continue;

            // Compute old incident-edge cost, then new, locally. We sum the
            // enter+leave edges around a and around b BEFORE, swap types in the
            // sequence view, and recompute. Adjacent-on-same-machine pairs share
            // an edge -- handled by recomputing the small window directly.
            auto windowCost = [&](int m, int p) -> long long {
                // enter(p) + leave(p)
                return (long long)enterCost(m, p) + leaveCost(m, p);
            };

            long long before;
            if (ma == mb) {
                int lo = min(pa, pb), hi = max(pa, pb);
                if (hi == lo + 1) {
                    // adjacent: edges = enter(lo) + leave(lo)(=lo->hi) + leave(hi)
                    before = (long long)enterCost(ma, lo) + leaveCost(ma, lo) + leaveCost(ma, hi);
                } else {
                    before = windowCost(ma, pa) + windowCost(mb, pb);
                }
            } else {
                before = windowCost(ma, pa) + windowCost(mb, pb);
            }

            // perform the swap on the sequence, recompute the same window.
            std::swap(seqv[ma][pa], seqv[mb][pb]);
            // pos/mac of a and b temporarily inconsistent; recompute window using
            // the new contents directly (enterCost/leaveCost read seqv only).
            long long after;
            if (ma == mb) {
                int lo = min(pa, pb), hi = max(pa, pb);
                if (hi == lo + 1) {
                    after = (long long)enterCost(ma, lo) + leaveCost(ma, lo) + leaveCost(ma, hi);
                } else {
                    after = windowCost(ma, pa) + windowCost(mb, pb);
                }
            } else {
                after = windowCost(ma, pa) + windowCost(mb, pb);
            }
            long long delta = after - before;

            if (delta <= 0 || rng.nextd() < exp(-(double)delta / Temp)) {
                // accept: fix reverse lookup for the two touched slots.
                mac[seqv[ma][pa]] = ma; pos[seqv[ma][pa]] = pa;
                mac[seqv[mb][pb]] = mb; pos[seqv[mb][pb]] = pb;
                curSetup += delta;
                if (curSetup < bestSetup) { bestSetup = curSetup; bestSeq = seqv; }
            } else {
                // revert
                std::swap(seqv[ma][pa], seqv[mb][pb]);
            }
        }
    }

    // ----- Emit the best schedule found. ------------------------------------
    seqv = bestSeq;
    // sanity: ensure every job appears exactly once (defensive; construction
    // guarantees this, but never emit an infeasible schedule).
    {
        vector<char> seen(N, 0);
        long long cnt = 0;
        for (int m = 0; m < M; m++)
            for (int j : seqv[m]) { if (j >= 0 && j < N && !seen[j]) { seen[j] = 1; cnt++; } }
        if (cnt != N) {
            // rebuild a trivial feasible schedule (round-robin) as a last resort
            seqv.assign(M, {});
            for (int j = 0; j < N; j++) seqv[j % M].push_back(j);
        }
    }

    string buf;
    buf.reserve((size_t)N * 5 + M * 4);
    for (int m = 0; m < M; m++) {
        buf += to_string((int)seqv[m].size());
        for (int j : seqv[m]) { buf += ' '; buf += to_string(j); }
        buf += '\n';
    }
    fputs(buf.c_str(), stdout);
    return 0;
}
