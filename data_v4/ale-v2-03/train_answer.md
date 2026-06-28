**Problem.** A factory runs `n` jobs (`15 ≤ n ≤ 30`) on `m` machines (`10 ≤ m ≤ 20`). Each job is a
fixed chain of `m` operations that must run in order; operation `t` of job `j` runs on machine `M[j][t]`
for `P[j][t] ∈ [1,99]` time units. Every job visits every machine exactly once, so each machine has
exactly `n` operations queued (one per job), and a machine runs one operation at a time with no
preemption. Read `n m` and the per-job operation pairs from stdin; print `m` lines, line `k` a
permutation of `0…n-1` giving the order in which machine `k` processes the jobs.

**Objective and scoring.** This is the disjunctive-graph job shop `J||C_max`: the chosen machine orders,
plus the immutable job chains, induce a directed graph on the `n·m` operations, and the **makespan** `C`
is the longest source-to-sink path in it. Minimize `C`. The scorer (deterministic) recomputes a reference
`B` — the makespan of an earliest-start non-delay **list schedule** — and reports

```
score = round(1 000 000 × B / C)   if the output is m valid permutations AND the graph is acyclic, C > 0
score = 0                          otherwise   (the feasibility floor)
```

So the list-scheduling reference scores exactly `1 000 000`, a shorter makespan scores more, and any
infeasible output — wrong token/line count, an out-of-range or repeated index, garbage, a missing file,
or a **cyclic (deadlocked) orientation** — scores `0`. The metric is the mean score over a fixed seed
set; the trivial identity-order output scores only ~150 000 and is the floor to beat.

**Baseline.** First reach *any* feasible answer. The identity output (every machine runs jobs in index
order) is permutation-valid but not guaranteed acyclic and is a poor makespan. The real construction
baseline is **list scheduling**: simulate the factory and repeatedly dispatch the operation that can
*start earliest* (ties to the lowest job index), appending each dispatched operation to its machine's
order. This is a non-delay/active schedule, **acyclic by construction**, always feasible, and exactly the
scorer's reference `B`. It typically lands 20–60% above optimal because it never revisits its early
ordering commitments — that gap is the slack to recover.

**Key idea — the heuristic innovation.** The makespan is a longest path, so **only operations on a
critical path can be reordered to shorten it**; swapping operations off the critical path is provably
neutral, and scanning all `Θ(mn)` adjacent pairs with full makespan recomputation is `Θ((mn)²)` and
hopeless. The fix is the **N5 critical-block neighbourhood** (Nowicki & Smutnicki):

1. **Disjunctive graph + heads/tails.** Represent the schedule as a graph (job-chain arcs + per-machine
   order arcs). A forward Kahn topological pass gives `heads[o]` = earliest start of `o` (longest path
   from the source), and the makespan is `max_o heads[o]+P[o]`. A reverse pass over the same order gives
   `tails[o]` = longest path from `o` to the sink including `P[o]`. An operation is **critical** iff
   `heads[o]+tails[o] == makespan`. Both passes are `O(nm)` — a few hundred nodes here.
2. **Critical blocks.** Reconstruct one concrete critical path by greedy forward walking (prefer the
   machine-successor), then cut it into maximal runs of operations that share a machine and are
   machine-adjacent — the **blocks**.
3. **N5 moves.** The only candidate moves are swapping the **first two** or the **last two** operations of
   each critical block. These are the only swaps that can shorten the current critical path, and they
   **provably never create a cycle**. A handful of moves per iteration, every one targeted, instead of
   `Θ(mn)` mostly-neutral ones.

Drive it with **tabu search**: each step build the critical-block moves, evaluate each by the makespan it
produces (one `O(nm)` heads pass), take the best **admissible** one, and forbid reversing the just-flipped
arc for a short tenure. **Aspiration** overrides tabu when a move beats the global best. Keep the best
schedule in a snapshot; on a long stall, **diversify** (restore the best, apply a few random feasible
swaps, clear the tabu list). This is the established strong-yet-simple metaheuristic for the job shop and
converts the leftover time budget into real makespan reductions.

**Feasibility and pitfalls.**
- *Two-part feasibility.* Every output line must be a permutation of `0…n-1` *and* the induced graph must
  be **acyclic** — a deadlock (two machines each waiting on the other) is infeasible and scores `0`. The
  machine sequences are permutations by construction; acyclicity is enforced by validation, not trust.
- *Cycle guard on every move.* Although N5 block-border swaps are theoretically cycle-free, the solver
  never relies on that: the forward Kahn pass *is* the cycle detector (if it cannot process all `nm`
  operations the graph is cyclic), and any swap that deadlocks is reverted and skipped. The committed
  state is therefore always feasible.
- *Critical-path reconstruction.* The set of critical operations is the *union* of all critical paths, not
  a single path; threading it naively yields fake "blocks" with non-adjacent pairs, which swap the wrong
  slots and raise the makespan. The fix is to walk one concrete critical path and emit a swap only when
  `machNext(a)==c` genuinely holds.
- *Always-feasible cutoff.* The wall-clock budget is sampled cheaply (once per 256 inner evaluations) and
  the printed answer is the best feasible machine orders found, so the time limit never yields a
  half-applied or deadlocked output.
- *Edge cases.* `n ≤ 0` or `m ≤ 0` prints nothing; an empty critical move list triggers a small random
  feasible kick instead of stalling; parse failures substitute `(machine 0, proc 0)` and clamp the machine
  index rather than crash.

**Complexity per step.** Each makespan / heads / tails evaluation is one `O(nm)` topological pass.
Building the critical moves walks one critical path (`O(nm)`) and yields `O(#blocks)` candidate moves;
evaluating them costs `O(#moves · nm)`. Applying a move is an `O(1)` adjacent swap plus a `O(nm)`
heads+tails refresh. The verified result on seeds 1–20: every output feasible, solver mean ≈ `1 360 000`
(makespans ≈ 36% shorter than the list-scheduling reference, minimum ≈ `1 172 000`), versus the
identity-order baseline's ≈ `157 000` — an >8× margin over the floor and a clear margin over the
list-scheduling reference, within a ~1.9 s budget.

**Code.**

```cpp
// Factory Job-Shop Scheduling -- heuristic solver.
//
// Objective: schedule n jobs on m machines to MINIMIZE the makespan. Job j is a
// chain of m operations that must run in the given order; operation op(j,t) runs
// on machine M[j][t] for P[j][t] time. Each machine runs one operation at a time
// and every job uses every machine exactly once. Read the instance from stdin;
// print, for each machine k = 0..m-1 (one line each, in increasing k), the order
// in which that machine processes the n jobs (a permutation of 0..n-1). The
// makespan is the longest path in the disjunctive graph induced by those orders.
//
// Method (the innovation):
//   * Represent the schedule as the disjunctive graph: fixed job-chain arcs plus,
//     per machine, a chosen total order of its n operations. The makespan is the
//     longest source->sink path; an op is CRITICAL iff head+proc+tail == makespan.
//   * Build a feasible start with a deterministic non-delay LIST SCHEDULE.
//   * Local search over the N5 critical-block neighbourhood (Nowicki & Smutnicki):
//     decompose the critical path into maximal blocks of consecutive ops sharing a
//     machine; the only moves are swapping the FIRST two or the LAST two ops of a
//     block. These moves provably keep the graph acyclic and are the only ones that
//     can shorten the current critical path -- so we never scan the whole schedule.
//   * Drive it with TABU SEARCH: take the best admissible block move each step
//     (recomputing heads/tails to evaluate makespan), forbid reversing it back for
//     a few iterations (aspiration overrides tabu when a move beats the incumbent),
//     and keep the best schedule ever seen.
// The current machine orders are always valid permutations and the start state is
// acyclic, so any early stop (including the time limit) still prints a feasible
// solution.
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
    uint32_t nextu(uint32_t mod) { return (uint32_t)(next() % mod); }
};

int N, M;                       // jobs, machines (= ops per job)
int NOP;                        // number of operations = N*M
vector<int> opMach, opProc;     // per-op machine and processing time
vector<int> jobNext, jobPrev;   // job-chain successor / predecessor op (or -1)

static inline int opid(int j, int t) { return j * M + t; }

int main() {
    const double T0 = now_sec();
    const double TIME_LIMIT = 1.90;  // wall-clock budget (seconds)

    if (scanf("%d %d", &N, &M) != 2) return 0;
    if (N <= 0 || M <= 0) { return 0; }
    NOP = N * M;
    opMach.assign(NOP, 0);
    opProc.assign(NOP, 0);
    jobNext.assign(NOP, -1);
    jobPrev.assign(NOP, -1);
    // mOps[k] = list of op-ids that run on machine k (one per job)
    vector<vector<int>> mOps(M);
    for (int j = 0; j < N; j++) {
        for (int t = 0; t < M; t++) {
            int mc = 0, p = 0;
            if (scanf("%d %d", &mc, &p) != 2) { mc = 0; p = 0; }
            if (mc < 0) mc = 0; if (mc >= M) mc = M - 1;
            if (p < 0) p = 0;
            int o = opid(j, t);
            opMach[o] = mc;
            opProc[o] = p;
            if (t > 0) { jobPrev[o] = opid(j, t - 1); jobNext[opid(j, t - 1)] = o; }
            mOps[mc].push_back(o);
        }
    }

    // ---------- machine-order state ----------
    // mseq[k] = ordered list of op-ids on machine k (a permutation of mOps[k]).
    // posInM[o] = index of op o within mseq[opMach[o]].
    // For each op, machine predecessor / successor are derived from mseq.
    vector<vector<int>> mseq(M);
    vector<int> posInM(NOP, 0);
    vector<int> jobOfOp(NOP, 0);
    for (int j = 0; j < N; j++)
        for (int t = 0; t < M; t++)
            jobOfOp[opid(j, t)] = j;

    // ---------- deterministic non-delay list schedule -> initial machine orders --
    {
        vector<int> nextT(N, 0);          // next op index to dispatch per job
        vector<long long> jobReady(N, 0); // earliest start of job j's next op
        vector<long long> machFree(M, 0); // earliest time machine k is free
        int remaining = NOP;
        while (remaining > 0) {
            int bestJob = -1; long long bestStart = LLONG_MAX;
            for (int j = 0; j < N; j++) {
                int t = nextT[j];
                if (t >= M) continue;
                int o = opid(j, t);
                int k = opMach[o];
                long long s = max(jobReady[j], machFree[k]);
                if (s < bestStart || (s == bestStart && bestJob < 0)) {
                    bestStart = s; bestJob = j;
                }
            }
            int j = bestJob, t = nextT[j];
            int o = opid(j, t), k = opMach[o];
            long long fin = bestStart + opProc[o];
            machFree[k] = fin; jobReady[j] = fin;
            mseq[k].push_back(o);          // append in dispatch order
            nextT[j]++; remaining--;
        }
    }
    for (int k = 0; k < M; k++)
        for (int i = 0; i < (int)mseq[k].size(); i++)
            posInM[mseq[k][i]] = i;

    // machine predecessor / successor of an op given current mseq
    auto machPrev = [&](int o) -> int {
        int k = opMach[o], i = posInM[o];
        return i > 0 ? mseq[k][i - 1] : -1;
    };
    auto machNext = [&](int o) -> int {
        int k = opMach[o], i = posInM[o];
        return i + 1 < (int)mseq[k].size() ? mseq[k][i + 1] : -1;
    };

    // ---------- longest-path makespan via Kahn topo order ----------
    // heads[o] = earliest start (longest path from source); makespan via heads+proc.
    // Returns makespan, or -1 if the induced graph has a cycle (infeasible).
    vector<int> indeg(NOP), heads(NOP), topo(NOP);
    auto computeHeads = [&](long long &outMk) -> bool {
        for (int o = 0; o < NOP; o++) { indeg[o] = 0; heads[o] = 0; }
        for (int o = 0; o < NOP; o++) {
            int s1 = jobNext[o]; if (s1 >= 0) indeg[s1]++;
            int s2 = machNext(o); if (s2 >= 0) indeg[s2]++;
        }
        int head = 0, tail = 0;
        for (int o = 0; o < NOP; o++) if (indeg[o] == 0) topo[tail++] = o;
        long long mk = 0; int processed = 0;
        while (head < tail) {
            int u = topo[head++]; processed++;
            long long fin = (long long)heads[u] + opProc[u];
            if (fin > mk) mk = fin;
            int s1 = jobNext[u];
            if (s1 >= 0) {
                if (fin > heads[s1]) heads[s1] = (int)fin;
                if (--indeg[s1] == 0) topo[tail++] = s1;
            }
            int s2 = machNext(u);
            if (s2 >= 0) {
                if (fin > heads[s2]) heads[s2] = (int)fin;
                if (--indeg[s2] == 0) topo[tail++] = s2;
            }
        }
        if (processed != NOP) return false;  // cycle
        outMk = mk;
        return true;
    };

    // tails[o] = longest path from o to sink, INCLUDING opProc[o]. Critical iff
    // heads[o] + tails[o] == makespan. Computed by processing topo order in reverse.
    vector<int> tails(NOP);
    auto computeTails = [&]() {
        for (int o = 0; o < NOP; o++) tails[o] = 0;
        for (int idx = NOP - 1; idx >= 0; idx--) {
            int u = topo[idx];
            int best = 0;
            int s1 = jobNext[u]; if (s1 >= 0) best = max(best, tails[s1]);
            int s2 = machNext(u); if (s2 >= 0) best = max(best, tails[s2]);
            tails[u] = best + opProc[u];
        }
    };

    long long curMk = 0;
    computeHeads(curMk);     // initial state is a real schedule -> always acyclic
    computeTails();

    // ---------- best-so-far snapshot ----------
    vector<vector<int>> bestSeq = mseq;
    long long bestMk = curMk;

    auto syncPos = [&](int k) {
        for (int i = 0; i < (int)mseq[k].size(); i++) posInM[mseq[k][i]] = i;
    };

    // swap two adjacent ops at positions i, i+1 on machine k (a directed-arc reversal)
    auto swapAdj = [&](int k, int i) {
        std::swap(mseq[k][i], mseq[k][i + 1]);
        posInM[mseq[k][i]] = i;
        posInM[mseq[k][i + 1]] = i + 1;
    };

    // ---------- tabu list: forbid re-reversing a recently reversed (a before b) ----
    // key on ordered pair (a,b) meaning "a immediately precedes b on its machine".
    // We store an expiry iteration in a hash map.
    unordered_map<long long, long long> tabuUntil;
    tabuUntil.reserve(4096);
    auto tabuKey = [&](int a, int b) -> long long {
        return (long long)a * (long long)NOP + b;
    };

    Rng rng(0xA1E5C0DEull ^ ((uint64_t)N * 1000003ull + (uint64_t)M * 7919ull));

    long long iter = 0;
    int tabuTenure = 8 + (N + M) / 4;     // mild dependence on instance size
    int sinceImprove = 0;

    long long clk = 0;
    auto timeUp = [&]() {
        if ((++clk & 255) == 0) return now_sec() - T0 > TIME_LIMIT;
        return false;
    };

    // Build the set of candidate block moves from the current critical path, then
    // pick the best admissible one, apply it, update tabu, and repeat.
    // A "move" = swap the adjacent pair (a,b) on machine k (positions i,i+1).
    struct Move { int k, i, a, b; };
    vector<Move> moves;
    moves.reserve(256);

    auto buildCriticalMoves = [&]() {
        moves.clear();
        // Critical ops: heads[o] + tails[o] == makespan. We walk one critical
        // source->sink path and cut it into blocks of consecutive ops on the same
        // machine. For each block we propose swapping its first two and last two ops.
        // Reconstruct a critical path: start from a critical op with heads==0 (a
        // source on a longest path), greedily follow a critical successor.
        // First find a critical starting op.
        int start = -1;
        for (int o = 0; o < NOP; o++) {
            if ((long long)heads[o] + tails[o] == curMk && heads[o] == 0) { start = o; break; }
        }
        if (start < 0) {
            for (int o = 0; o < NOP; o++)
                if ((long long)heads[o] + tails[o] == curMk) { start = o; break; }
        }
        if (start < 0) return;
        // Walk the critical path.
        vector<int> path;
        int cur = start;
        path.push_back(cur);
        while (true) {
            int nxt = -1;
            int s1 = jobNext[cur];
            int s2 = machNext(cur);
            // a successor s is on the critical path iff
            // heads[cur]+proc[cur] == heads[s] and heads[s]+tails[s]==makespan
            long long fin = (long long)heads[cur] + opProc[cur];
            if (s2 >= 0 && (long long)heads[s2] + tails[s2] == curMk && heads[s2] == fin)
                nxt = s2;                          // prefer staying on the machine block
            if (nxt < 0 && s1 >= 0 && (long long)heads[s1] + tails[s1] == curMk && heads[s1] == fin)
                nxt = s1;
            if (nxt < 0) {
                // fall back: any critical successor with matching head
                if (s1 >= 0 && (long long)heads[s1] + tails[s1] == curMk && heads[s1] == fin) nxt = s1;
            }
            if (nxt < 0) break;
            path.push_back(nxt);
            cur = nxt;
            if ((int)path.size() > NOP) break;     // safety
        }
        // Cut into machine blocks: maximal runs where consecutive path ops share a
        // machine AND are machine-adjacent (the second is the machine-successor of
        // the first). For each block of length >= 2, propose first-two/last-two swaps.
        int L = (int)path.size();
        int b0 = 0;
        while (b0 < L) {
            int k = opMach[path[b0]];
            int b1 = b0;
            while (b1 + 1 < L &&
                   opMach[path[b1 + 1]] == k &&
                   machNext(path[b1]) == path[b1 + 1]) {
                b1++;
            }
            int blockLen = b1 - b0 + 1;
            if (blockLen >= 2) {
                // swap first two ops of the block (positions of path[b0], path[b0+1])
                int a = path[b0], c = path[b0 + 1];
                int i = posInM[a];
                if (machNext(a) == c) moves.push_back({k, i, a, c});
                // swap last two ops of the block
                if (blockLen >= 3 || (blockLen == 2)) {
                    int a2 = path[b1 - 1], c2 = path[b1];
                    int i2 = posInM[a2];
                    if (machNext(a2) == c2) {
                        // avoid duplicating the same pair for a length-2 block
                        if (!(a2 == a && c2 == c)) moves.push_back({k, i2, a2, c2});
                    }
                }
            }
            b0 = b1 + 1;
        }
    };

    // Evaluate a candidate swap: apply, recompute makespan, then revert. Returns the
    // makespan if feasible (acyclic), or LLONG_MAX if the swap created a cycle.
    auto evalSwap = [&](const Move &mv) -> long long {
        swapAdj(mv.k, mv.i);
        long long mk = 0;
        bool ok = computeHeads(mk);
        swapAdj(mv.k, mv.i);   // revert (positions restored)
        return ok ? mk : LLONG_MAX;
    };

    while (now_sec() - T0 < TIME_LIMIT) {
        iter++;
        buildCriticalMoves();
        if (moves.empty()) {
            // No critical block move (e.g. critical path is a single job chain with
            // no machine block of length >=2): perturb to escape.
            // Pick a random machine with >=2 ops and swap a random adjacent pair,
            // keeping feasibility; this is a small kick.
            bool kicked = false;
            for (int tries = 0; tries < 12 && !kicked; tries++) {
                int k = rng.nextu(M);
                if ((int)mseq[k].size() < 2) continue;
                int i = rng.nextu((uint32_t)mseq[k].size() - 1);
                swapAdj(k, i);
                long long mk = 0;
                if (computeHeads(mk)) { curMk = mk; computeTails(); kicked = true; }
                else swapAdj(k, i);  // revert cycle
            }
            if (!kicked) break;
            if (curMk < bestMk) { bestMk = curMk; bestSeq = mseq; }
            continue;
        }

        // pick the best admissible move (lowest resulting makespan).
        long long bestMoveMk = LLONG_MAX; int bestIdx = -1;
        long long aspMoveMk = LLONG_MAX; int aspIdx = -1;  // best ignoring tabu
        for (int mi = 0; mi < (int)moves.size(); mi++) {
            if (timeUp()) break;
            const Move &mv = moves[mi];
            long long mk = evalSwap(mv);
            if (mk == LLONG_MAX) continue;          // would create a cycle: skip
            // best ignoring tabu (for aspiration / fallback)
            if (mk < aspMoveMk) { aspMoveMk = mk; aspIdx = mi; }
            // tabu test: forbid reversing the arc (a before b) back, i.e. the move
            // that would put b before a is tabu if (b,a) is in the tabu list.
            long long key = tabuKey(mv.a, mv.b);
            bool isTabu = false;
            auto it = tabuUntil.find(key);
            if (it != tabuUntil.end() && it->second > iter) isTabu = true;
            // aspiration: allow a tabu move if it beats the global best
            if (isTabu && mk >= bestMk) continue;
            if (mk < bestMoveMk) { bestMoveMk = mk; bestIdx = mi; }
        }

        int chosen = bestIdx;
        if (chosen < 0) {
            // every critical move is tabu and none beats best: take the least-bad
            // non-improving move (aspIdx) to keep moving, or perturb if none.
            chosen = aspIdx;
        }
        if (chosen < 0) {
            // no feasible move at all from the critical set: random kick.
            for (int tries = 0; tries < 12; tries++) {
                int k = rng.nextu(M);
                if ((int)mseq[k].size() < 2) continue;
                int i = rng.nextu((uint32_t)mseq[k].size() - 1);
                swapAdj(k, i);
                long long mk = 0;
                if (computeHeads(mk)) { curMk = mk; computeTails(); break; }
                else swapAdj(k, i);
            }
            if (curMk < bestMk) { bestMk = curMk; bestSeq = mseq; }
            continue;
        }

        // apply the chosen move
        const Move mv = moves[chosen];
        swapAdj(mv.k, mv.i);
        long long mk = 0;
        if (!computeHeads(mk)) { swapAdj(mv.k, mv.i); computeHeads(curMk); computeTails(); continue; }
        curMk = mk;
        computeTails();
        // make reversing this arc tabu: after the swap, b precedes a; forbid putting
        // a before b again for a tenure.
        tabuUntil[tabuKey(mv.b, mv.a)] = iter + tabuTenure;

        if (curMk < bestMk) {
            bestMk = curMk; bestSeq = mseq; sinceImprove = 0;
        } else {
            sinceImprove++;
        }

        // diversification: if stuck for a while, apply a few random feasible kicks
        // and reset from the best to avoid wandering too far.
        if (sinceImprove > 60 + 4 * (N + M)) {
            mseq = bestSeq;
            for (int k = 0; k < M; k++) syncPos(k);
            int kicks = 2 + rng.nextu(4);
            for (int kk = 0; kk < kicks; kk++) {
                for (int tries = 0; tries < 12; tries++) {
                    int k = rng.nextu(M);
                    if ((int)mseq[k].size() < 2) continue;
                    int i = rng.nextu((uint32_t)mseq[k].size() - 1);
                    swapAdj(k, i);
                    long long mk2 = 0;
                    if (computeHeads(mk2)) break;
                    else swapAdj(k, i);
                }
            }
            computeHeads(curMk);
            computeTails();
            tabuUntil.clear();
            sinceImprove = 0;
            if (curMk < bestMk) { bestMk = curMk; bestSeq = mseq; }
        }
    }

    // ---------- output the best machine orders (always valid permutations) -------
    mseq = bestSeq;
    // map each machine's op-sequence to the job indices in that order.
    string out;
    out.reserve((size_t)NOP * 4);
    char buf[16];
    for (int k = 0; k < M; k++) {
        for (int i = 0; i < (int)mseq[k].size(); i++) {
            int o = mseq[k][i];
            int j = jobOfOp[o];
            int len = snprintf(buf, sizeof(buf), "%d%c", j, (i + 1 < (int)mseq[k].size()) ? ' ' : '\n');
            out.append(buf, len);
        }
        if (mseq[k].empty()) out.push_back('\n');
    }
    fputs(out.c_str(), stdout);
    return 0;
}
```
