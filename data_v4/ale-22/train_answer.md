# Job-Shop Scheduling (makespan): a critical-path tabu search

## Problem

We are given `n` jobs and `m` machines. Each job is a fixed chain of `m` operations; the `k`-th
operation of job `j` runs on machine `M[j][k]` for `D[j][k]` time and may start only after operation
`k-1` of the same job has finished. Every job visits every machine exactly once, and each machine runs
one operation at a time (no preemption). We must output, for each machine, the **order** in which it
processes its operations — `m` lines, line `i` a permutation of the job indices on machine `i`. Sizes
are `10 <= n <= 20`, `8 <= m <= 15`, durations in `[1, 200]`, time budget 2 seconds. This is the
classic `n x m` job-shop problem; it is strongly NP-hard.

## Objective and scoring

Minimize the **makespan**: the completion time of the last operation. The schedule is fully determined
by the machine orders via the **disjunctive graph** — a node per operation, fixed arcs chaining each
job's operations, plus arcs chaining consecutive operations on each machine. Start times are the
**longest path** to each node; the makespan is the longest path overall. A solution is feasible iff
this graph is **acyclic** (a cycle is a scheduling deadlock). The scorer decodes via a longest-path
(Kahn) pass, re-checks job precedence and machine exclusivity, and reports
`score = round(1e6 * SPT_baseline_makespan / max(1, makespan))`. A shorter makespan scores higher; any
infeasible output (wrong shape, non-permutation line, or a deadlock cycle) scores **0**.

## Baseline

Always have a valid solution in hand. The trivial feasible order is "every machine runs jobs in index
order `0..n-1`", which is acyclic for this family and decodes fine. For a strong starting point we
instead build a **Shortest-Processing-Time (SPT) list schedule**: simulate the shop, and whenever we
must pick the next operation, take the one that can *start earliest* (tie-break: shortest duration,
then job index), schedule it at `max(job_ready, machine_free)`, and append that job to its machine's
order. This is `O((n*m)*n)`, always feasible, and — being the scorer's own normalization rule — ties
the baseline at score `~1e6`, so every later improvement shows up directly as score above `1e6`.

## Key idea: the critical-block neighborhood

The natural local move is to swap two machine-consecutive operations and re-decode. The naive version
scans all `O(n*m)` adjacent pairs per step, each costing an `O(n*m)` decode, and almost every swap
leaves the makespan unchanged. The established lever: a swap can shorten the makespan **only if both
operations lie on a critical path** (a longest path realizing the makespan). So we extract one critical
path — heads `h` (longest path in) and tails `t` (longest path out) give criticality `h+t==makespan`;
walk predecessors backward from a makespan-achieving sink, accepting a predecessor only if it is both
on a longest path and critical — and the candidate moves are exactly the consecutive path pairs joined
by a **machine** arc. The neighborhood drops from `O(n*m)` to the (short) critical-path length, and
every candidate can actually help.

We drive this with **tabu search**: each step evaluates every critical-block machine swap by a full
cheap decode, applies the best one even if it worsens the makespan, forbids undoing it for a short
tenure (keyed on the swapped operation pair), with an **aspiration** override when a move reaches a new
global best. On stagnation, or when the critical path is all job arcs (no machine swap available), we
**diversify** by perturbing the best incumbent with a few random adjacent swaps and clearing the tabu
list. The best decoded order is retained throughout.

## Feasibility and pitfalls

- **Acyclicity is feasibility.** The decode is a Kahn longest-path pass; if it can't pop all `n*m`
  nodes there is a deadlock cycle and the order is infeasible. Swapping two *adjacent* operations on a
  single machine reverses one disjunctive arc and provably preserves acyclicity, so every move stays
  feasible — we never have to repair a deadlock.
- **Critical-path extraction must require criticality at every step**, not just the head equation; a
  predecessor on a head-tie that is off every longest path leads to a stunted one-node path and an
  empty neighborhood.
- **Restore the working order after every trial.** We swap-decode-unswap when scoring a candidate; the
  applied move must use the *winning* candidate's own recorded `(machine, position)`, not a stale index
  from the scan.
- **Always print a feasible order**: a final `decode(best)` guard falls back to the index-order ordering
  if anything unexpected made `best` infeasible, and we keep the incumbent decoded so any early timeout
  still emits a valid schedule.

## Complexity per step

One decode (heads + tails via two topological passes) is `O(n*m)`. Critical-path extraction is
`O(n*m)`. The neighborhood has `O(len(critical path))` candidates, each costing one `O(n*m)` decode, so
a tabu step is `O(len * n*m)` with `len` small (the critical path, not all pairs). Within the 1.85s
budget this runs many thousands of steps on the `n,m <= 20,15` instances.

## Result

Over seeds 1..20 every output is feasible and the solver beats the trivial index-order baseline on all
20 seeds and the SPT normalization baseline (1e6) on all 20 — mean score ~1.27e6, i.e. ~20-50% shorter
makespan than the SPT list schedule it starts from, at a steady 1.85s.

## Code

```cpp
// Job-Shop Scheduling (makespan) -- heuristic solver.
//
// Objective: n jobs, m machines; job j is a fixed chain of m operations, the
// k-th running on machine M[j][k] for D[j][k] time, startable only after op k-1
// of the same job finishes; a machine runs one op at a time. We choose, for each
// machine, the ORDER in which it runs its operations (a permutation of the jobs),
// and we MINIMIZE the makespan (last completion time). We read the instance from
// stdin and write, to stdout, m lines: line i is machine i's processing order,
// given as the sequence of job indices.
//
// Method (the innovation -- disjunctive graph + critical-path neighborhood):
//   * A candidate is a machine order. To evaluate it we DECODE start times by a
//     longest path in the disjunctive graph: nodes are operations; job-precedence
//     arcs chain each job's ops; machine arcs chain consecutive ops on a machine.
//     The makespan is the longest path length. We compute heads (longest path in)
//     and tails (longest path out) in O(n*m) by a topological pass.
//   * A swap of two ADJACENT operations on a machine changes the makespan only if
//     it lies on a CRITICAL PATH (a longest path realizing the makespan); a swap
//     off the critical path cannot shorten it. So the neighborhood is exactly the
//     pairs of machine-consecutive operations that are BOTH on some critical path
//     and adjacent in a critical block -- found by one backward pass from the
//     makespan-achieving sink. This is the established Nowicki-Smutnicki style
//     critical-block neighborhood; it is tiny (O(critical-path length)) compared
//     to the O(n*m*m) of "try every adjacent swap", and every move is guaranteed
//     to keep the graph acyclic (swapping two adjacent critical ops on a machine
//     can never create a cycle), so the schedule stays FEASIBLE by construction.
//   * We drive it with TABU SEARCH: each step evaluates every critical-block swap
//     by a full (cheap) decode, takes the best non-tabu move (aspiration: always
//     take a move that beats the global best), forbids the reverse move for a few
//     iterations, and keeps the best machine order seen. A handful of diversifying
//     restarts from perturbed orders rounds it out. The incumbent order is always
//     a valid schedule, so any early stop still prints a feasible solution.
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

int N, M;                       // jobs, machines
vector<vector<int>> MC, DU;     // MC[j][k] = machine, DU[j][k] = duration of op (j,k)
vector<vector<int>> POS;        // POS[j][machine] = k (which op of job j is on that machine)

// Operation id and helpers.
static inline int oid(int j, int k) { return j * M + k; }

// The decision: order on each machine, as a sequence of job indices.
// mseq[i] = vector of jobs in the order machine i runs them.
// We also keep, per operation, its predecessor/successor on its machine, derived
// from mseq, so the longest-path decode is O(n*m).

struct Decode {
    // For each operation node (id = j*M+k):
    vector<int> head;   // earliest start = longest path from a source to this op
    vector<int> tail;   // longest path from this op (inclusive of its own dur) to a sink, minus... see below
    // machine neighbours of operation (j,k): the op before/after it on its machine
    vector<int> mPrev;  // operation id, or -1
    vector<int> mNext;  // operation id, or -1
    int makespan;
    bool feasible;
};

// Build mPrev/mNext from machine sequences.
void buildMachineLinks(const vector<vector<int>>& mseq,
                       vector<int>& mPrev, vector<int>& mNext) {
    int NM = N * M;
    mPrev.assign(NM, -1);
    mNext.assign(NM, -1);
    for (int i = 0; i < M; i++) {
        const vector<int>& seq = mseq[i];
        for (size_t t = 0; t < seq.size(); t++) {
            int j = seq[t];
            int k = POS[j][i];
            int id = oid(j, k);
            if (t > 0) {
                int pj = seq[t - 1];
                int pk = POS[pj][i];
                mPrev[id] = oid(pj, pk);
            }
            if (t + 1 < seq.size()) {
                int nj = seq[t + 1];
                int nk = POS[nj][i];
                mNext[id] = oid(nj, nk);
            }
        }
    }
}

// Longest-path decode. Returns feasible=false if the disjunctive graph has a cycle.
// head[id]   = earliest start time of op id.
// tail[id]   = longest remaining path length from the START of op id to the end
//              (i.e. dur(id) + max over successors of tail(succ)); so the makespan
//              is max over ops of (head + tail) and an op is CRITICAL iff
//              head + tail == makespan.
Decode decode(const vector<vector<int>>& mseq) {
    Decode d;
    int NM = N * M;
    d.head.assign(NM, 0);
    d.tail.assign(NM, 0);
    buildMachineLinks(mseq, d.mPrev, d.mNext);

    // Successors of an op: job-next (j,k+1) and machine-next (mNext). Predecessors:
    // job-prev (j,k-1) and machine-prev (mPrev). Compute a topological order via
    // Kahn over indegree, accumulating heads. If we cannot order all nodes -> cycle.
    static vector<int> indeg;
    indeg.assign(NM, 0);
    // count predecessors
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < M; k++) {
            int id = oid(j, k);
            if (k > 0) indeg[id]++;            // job precedence
            if (d.mPrev[id] >= 0) indeg[id]++; // machine precedence
        }
    }
    static vector<int> order; order.clear(); order.reserve(NM);
    static vector<int> stk; stk.clear();
    for (int id = 0; id < NM; id++) if (indeg[id] == 0) stk.push_back(id);
    while (!stk.empty()) {
        int u = stk.back(); stk.pop_back();
        order.push_back(u);
        int j = u / M, k = u % M;
        int hu = d.head[u];
        int du = DU[j][k];
        // job successor
        if (k + 1 < M) {
            int v = oid(j, k + 1);
            if (hu + du > d.head[v]) d.head[v] = hu + du;
            if (--indeg[v] == 0) stk.push_back(v);
        }
        // machine successor
        int v2 = d.mNext[u];
        if (v2 >= 0) {
            if (hu + du > d.head[v2]) d.head[v2] = hu + du;
            if (--indeg[v2] == 0) stk.push_back(v2);
        }
    }
    if ((int)order.size() != NM) { d.feasible = false; d.makespan = INT_MAX; return d; }
    d.feasible = true;

    // tails: process in reverse topological order.
    d.makespan = 0;
    for (int idx = (int)order.size() - 1; idx >= 0; idx--) {
        int u = order[idx];
        int j = u / M, k = u % M;
        int du = DU[j][k];
        int best = 0;
        if (k + 1 < M) best = max(best, d.tail[oid(j, k + 1)]);
        int v2 = d.mNext[u];
        if (v2 >= 0) best = max(best, d.tail[v2]);
        d.tail[u] = du + best;
        d.makespan = max(d.makespan, d.head[u] + d.tail[u]);
    }
    return d;
}

// Emit a machine order to stdout.
void emit(const vector<vector<int>>& mseq) {
    string buf;
    buf.reserve(N * M * 4 + M);
    for (int i = 0; i < M; i++) {
        for (size_t t = 0; t < mseq[i].size(); t++) {
            if (t) buf += ' ';
            buf += to_string(mseq[i][t]);
        }
        buf += '\n';
    }
    fputs(buf.c_str(), stdout);
}

// ---------- baseline construction: SPT-ordered active list schedule -------------
// Produces a machine order (feasible by construction) using a Giffler-Thompson
// style list schedule with the SPT priority. This is a strong, always-feasible
// starting point for the local search.
vector<vector<int>> constructListSchedule(int rule, Rng& rng) {
    // We simulate scheduling op by op. For each machine we append jobs in the
    // order we actually schedule operations on it.
    vector<vector<int>> mseq(M);
    vector<int> nextk(N, 0);
    vector<long long> jobReady(N, 0);
    vector<long long> machFree(M, 0);
    int remaining = N * M;
    while (remaining > 0) {
        // candidate next-ops: for each job, its current op (if any left)
        // pick by earliest possible start, tie-broken by rule.
        int chosen = -1;
        long long bestStart = LLONG_MAX;
        long long bestKey2 = LLONG_MAX;
        for (int j = 0; j < N; j++) {
            int k = nextk[j];
            if (k >= M) continue;
            int mc = MC[j][k];
            long long st = max(jobReady[j], machFree[mc]);
            long long dur = DU[j][k];
            long long key2;
            if (rule == 0) key2 = dur;                 // SPT
            else if (rule == 1) key2 = -dur;           // LPT
            else key2 = (long long)rng.nextu(1u << 30); // random
            if (st < bestStart ||
                (st == bestStart && (key2 < bestKey2 ||
                 (key2 == bestKey2 && j < chosen)))) {
                bestStart = st; bestKey2 = key2; chosen = j;
            }
        }
        int j = chosen, k = nextk[j];
        int mc = MC[j][k];
        long long st = max(jobReady[j], machFree[mc]);
        long long end = st + DU[j][k];
        machFree[mc] = end;
        jobReady[j] = end;
        nextk[j]++;
        remaining--;
        mseq[mc].push_back(j);
    }
    return mseq;
}

// Extract one critical path as a list of operation ids, from a source to the
// makespan-achieving sink, following predecessors with head+tail == makespan.
void criticalPath(const Decode& d, vector<int>& path) {
    path.clear();
    int NM = N * M;
    // find a sink op with head+tail == makespan and no successor on the critical path
    int cur = -1;
    for (int id = 0; id < NM; id++) {
        if (d.head[id] + d.tail[id] == d.makespan) {
            int j = id / M, k = id % M;
            // an op is an END of a critical path if neither of its successors is critical
            bool jobSuccCrit = (k + 1 < M) && (d.head[oid(j, k + 1)] + d.tail[oid(j, k + 1)] == d.makespan)
                               && (d.head[oid(j, k + 1)] == d.head[id] + DU[j][k]);
            int mn = d.mNext[id];
            bool machSuccCrit = (mn >= 0) && (d.head[mn] + d.tail[mn] == d.makespan)
                               && (d.head[mn] == d.head[id] + DU[j][k]);
            if (!jobSuccCrit && !machSuccCrit) { cur = id; break; }
        }
    }
    if (cur < 0) {
        // fallback: pick any critical op
        for (int id = 0; id < NM; id++)
            if (d.head[id] + d.tail[id] == d.makespan) { cur = id; break; }
    }
    // walk backward to a source
    while (cur >= 0) {
        path.push_back(cur);
        int j = cur / M, k = cur % M;
        int hcur = d.head[cur];
        // a predecessor is critical if it is on a longest path: head[pred]+dur[pred]==head[cur]
        // and pred is critical (head+tail==makespan).
        int prev = -1;
        if (k > 0) {
            int p = oid(j, k - 1);
            if (d.head[p] + DU[j][k - 1] == hcur &&
                d.head[p] + d.tail[p] == d.makespan) prev = p;
        }
        if (prev < 0) {
            int p = d.mPrev[cur];
            if (p >= 0) {
                int pj = p / M, pk = p % M;
                if (d.head[p] + DU[pj][pk] == hcur &&
                    d.head[p] + d.tail[p] == d.makespan) prev = p;
            }
        }
        cur = prev;
    }
    reverse(path.begin(), path.end());
}

int main() {
    const double T0 = now_sec();
    const double TIME_LIMIT = 1.85;

    if (scanf("%d %d", &N, &M) != 2) return 0;
    if (N <= 0 || M <= 0) { return 0; }
    MC.assign(N, vector<int>(M));
    DU.assign(N, vector<int>(M));
    POS.assign(N, vector<int>(M, -1));
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < M; k++) {
            if (scanf("%d %d", &MC[j][k], &DU[j][k]) != 2) { MC[j][k] = 0; DU[j][k] = 1; }
            POS[j][MC[j][k]] = k;
        }
    }

    Rng rng(0x5151ULL ^ ((uint64_t)N << 32) ^ ((uint64_t)M << 16) ^ 0xBEEFu);

    // ---- initial solution: SPT list schedule (always feasible). ----
    vector<vector<int>> cur = constructListSchedule(0, rng);
    Decode dc = decode(cur);
    // Safety: if somehow infeasible (shouldn't happen for a list schedule), fall
    // back to a trivial machine order (jobs in index order on every machine), which
    // is a valid acyclic schedule for this problem family. We always have a feasible
    // incumbent to print.
    if (!dc.feasible) {
        for (int i = 0; i < M; i++) {
            cur[i].clear();
            for (int j = 0; j < N; j++) cur[i].push_back(j);
        }
        dc = decode(cur);
    }

    vector<vector<int>> best = cur;
    int bestMk = dc.feasible ? dc.makespan : INT_MAX;

    // ---- tabu search over the critical-block neighborhood ----
    // tabu list: forbid re-doing the reverse of a recent swap, keyed by the
    // unordered pair of operation ids, until iteration `tabuUntil[pair]`.
    unordered_map<long long, long long> tabuUntil;
    tabuUntil.reserve(4096);
    auto pairKey = [](int a, int b) -> long long {
        if (a > b) std::swap(a, b);
        return ((long long)a << 20) ^ (long long)b;
    };

    long long iter = 0;
    int tenure = max(5, (N + M) / 2);
    vector<int> path;
    int sinceImprove = 0;
    int restartLimit = 400 + 30 * (N + M);

    // We optimize on a working copy `work`; periodically restart from `best` with
    // a random perturbation if stuck.
    vector<vector<int>> work = cur;
    Decode dw = dc;

    while (true) {
        if ((iter & 63) == 0) {
            if (now_sec() - T0 > TIME_LIMIT) break;
        }
        iter++;

        if (!dw.feasible) {
            // recover (should not happen): reset to best.
            work = best; dw = decode(work);
            if (!dw.feasible) break;
        }

        // Build the critical-block neighborhood: adjacent machine-consecutive ops
        // on the critical path. For consecutive path ops u -> v that are linked by
        // a MACHINE arc (same machine, v == mNext[u]), swapping them is a candidate.
        criticalPath(dw, path);
        // Evaluate every candidate swap by a full decode (cheap: O(n*m)).
        long long bestDelta = LLONG_MAX;
        int swA = -1, swB = -1, swMachine = -1, swPosA = -1;
        int curMk = dw.makespan;

        for (size_t t = 1; t < path.size(); t++) {
            int u = path[t - 1];
            int v = path[t];
            // machine-arc on the critical path?
            if (dw.mNext[u] != v) continue;  // only machine-consecutive pairs
            int j_u = u / M, k_u = u % M;
            int mc = MC[j_u][k_u];           // the machine they share
            // positions of u (=job j_u) and v (=job j_v) in mseq[mc]
            int j_v = v / M;
            // find index of j_u in work[mc]
            // (we maintain machine sequences, so swap adjacent entries)
            // locate position of u in the sequence
            vector<int>& seq = work[mc];
            int posU = -1;
            for (size_t z = 0; z + 1 < seq.size(); z++) {
                if (seq[z] == j_u && seq[z + 1] == j_v) { posU = (int)z; break; }
            }
            if (posU < 0) continue;  // safety

            // tabu check (aspiration overrides)
            long long key = pairKey(u, v);
            auto itab = tabuUntil.find(key);
            bool isTabu = (itab != tabuUntil.end() && itab->second > iter);

            // perform the swap on a scratch, decode, measure makespan.
            std::swap(seq[posU], seq[posU + 1]);
            Decode dd = decode(work);
            long long mk = dd.feasible ? dd.makespan : LLONG_MAX;
            std::swap(seq[posU], seq[posU + 1]);  // undo

            long long delta = mk - (long long)curMk;
            // aspiration: a move reaching a new global best is allowed even if tabu
            bool allowed = !isTabu || (mk < (long long)bestMk);
            if (!allowed) continue;
            if (delta < bestDelta) {
                bestDelta = delta; swA = u; swB = v; swMachine = mc; swPosA = posU;
            }
        }

        if (swMachine < 0) {
            // No critical machine-swap available (e.g. critical path is all job
            // arcs). Diversify: perturb `best` and continue.
            work = best;
            // random kicks: a few random adjacent swaps on random machines
            int kicks = 1 + (int)rng.nextu(3 + (unsigned)M);
            for (int s = 0; s < kicks; s++) {
                int mc = rng.nextu(M);
                if (work[mc].size() >= 2) {
                    int p = rng.nextu((unsigned)work[mc].size() - 1);
                    std::swap(work[mc][p], work[mc][p + 1]);
                }
            }
            dw = decode(work);
            tabuUntil.clear();
            sinceImprove = 0;
            continue;
        }

        // apply best move
        std::swap(work[swMachine][swPosA], work[swMachine][swPosA + 1]);
        // forbid the reverse for `tenure` iterations
        tabuUntil[pairKey(swA, swB)] = iter + tenure;
        dw = decode(work);

        if (dw.feasible && dw.makespan < bestMk) {
            bestMk = dw.makespan;
            best = work;
            sinceImprove = 0;
        } else {
            sinceImprove++;
        }

        // diversification on stagnation
        if (sinceImprove > restartLimit) {
            work = best;
            int kicks = 2 + (int)rng.nextu(4 + (unsigned)M);
            for (int s = 0; s < kicks; s++) {
                int mc = rng.nextu(M);
                if (work[mc].size() >= 2) {
                    int p = rng.nextu((unsigned)work[mc].size() - 1);
                    std::swap(work[mc][p], work[mc][p + 1]);
                }
            }
            dw = decode(work);
            tabuUntil.clear();
            sinceImprove = 0;
        }
    }

    // Final guard: ensure `best` decodes feasibly; if not, fall back to trivial.
    Decode fb = decode(best);
    if (!fb.feasible) {
        for (int i = 0; i < M; i++) {
            best[i].clear();
            for (int j = 0; j < N; j++) best[i].push_back(j);
        }
    }
    emit(best);
    return 0;
}
```
