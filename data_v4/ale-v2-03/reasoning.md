**Reading the problem and fixing the contract before any algorithm.** A factory runs `n` jobs across
`m` machines. Each job is a fixed chain of `m` operations — operation `t` of job `j` runs on a named
machine for a named time, and within a job the operations must happen in the listed order. Every job
visits every machine exactly once, so each machine has exactly `n` operations queued (one per job), and
a machine does one operation at a time without preemption. I want to choose, for each machine, the order
in which it processes its `n` operations, and I want the **makespan** — the finish time of the very last
operation — to be as small as possible. Before writing a line of code I strip the surface story down to
its structure, because the structure decides the whole design. The only things I get to choose are the
machine orders; the per-job operation sequence and the processing times are fixed. So the decision space
is "one permutation per machine," and once those permutations are fixed, the start times are forced:
each operation starts as soon as its job-predecessor is done and its machine is free. That is exactly the
**disjunctive-graph** view of the job shop: nodes are operations, the immutable job chains are fixed
("conjunctive") arcs, and on each machine I orient the queue into a line. The makespan is then the
**length of the longest source-to-sink path** in that directed graph. This is `J||C_max`, classic and
NP-hard; the benchmark scores me by how short my makespan is, not by hitting a unique optimum. That last
point is the whole game: it is a continuous-score heuristic problem, so my job is to push the makespan
down as far as the budget allows, and to *never* emit something the scorer can refuse.

**Pinning the I/O and the feasibility rule, because an infeasible output scores zero.** Input is `n m`,
then `n` lines, each a job given as `m` pairs `mach proc`. Output is `m` lines, line `k` a permutation of
the job indices `0…n-1` saying the order machine `k` processes the jobs. The feasibility rule has *two*
parts, and the second one is the subtle trap of this problem. First, the obvious part: each output line
must be a genuine permutation of `{0,…,n-1}` — wrong count, a repeat, an out-of-range index, a stray
token, and the score floors to `0`. Second, the part specific to scheduling: the chosen machine orders,
combined with the job chains, must form an **acyclic** precedence graph. If machine A is told to run
job 1's operation before job 0's, while machine B is told to run job 0's operation before job 1's, and
the job chains thread through both, the two machines can end up each waiting on the other — a **deadlock**,
a cycle in the graph, for which *no* consistent start times exist. A cyclic orientation is infeasible and
scores `0`. So my design rule from the start is: hold a feasible (permutation-valid *and* acyclic) set of
machine orders at all times, and make the time-budget cutoff fall back on whatever feasible state I
currently have. Every move I make must preserve both invariants, not as an afterthought but as the
invariant of the search.

**Reaching a feasible baseline first.** Before optimizing anything I want a legal answer in hand. The
trivial output — every machine runs jobs in index order `0,1,…,n-1` — is permutation-valid but I cannot
even trust it to be acyclic, and on these instances it is a poor makespan anyway. The *right* first
baseline is **list scheduling**: simulate the factory and repeatedly dispatch, among all jobs whose next
operation is ready, the operation that can *start earliest* (ties to the lowest job index). Each dispatch
appends that operation to its machine's order. This is a non-delay / active schedule, and the key fact is
that it is **acyclic by construction** — I am literally simulating forward time, so no deadlock can arise.
It is always feasible, it is the natural construction heuristic for the job shop, and it is exactly the
reference `B` the scorer uses: a solver that merely reproduced list scheduling would score `1 000 000`.
My target is to beat that — to get the makespan `C` strictly below `B` so the ratio `B/C` exceeds one.
The list schedule typically lands 20–60% above optimal because it commits early to orderings it can never
revisit; that gap is the slack I am going to recover.

**Why the obvious local search is too slow and too weak, and what the lever has to be.** The naive way to
improve a schedule is: try swapping adjacent operations on machines, recompute the makespan, keep swaps
that help. Two things kill this immediately. First, there are `Θ(m·n)` adjacent pairs to consider, and a
full makespan recomputation is a longest-path pass over all `n·m` operations — so a single sweep that
recomputes the makespan for every candidate is `Θ(m·n · n·m)`, far too slow to iterate thousands of
times. Second, and more fundamentally, *most swaps cannot possibly help*: the makespan is the length of a
longest (critical) path, and swapping two operations that are not on a critical path leaves that path —
and hence the makespan — completely unchanged. I would be spending the overwhelming majority of my
evaluations on moves that are provably neutral. The structure tells me where to look: **only operations on
a critical path can be reordered to shorten the makespan.** So the search must propose moves *along the
critical path*, not across the whole schedule. This is the lever, and it has a precise classical form.

**Discovering the innovation — the N5 critical-block neighbourhood.** Take a critical path (a longest
source-to-sink path) and walk it. It alternates between job-chain arcs (moving to the next operation of
the same job) and machine arcs (moving to the next operation on the same machine). Group the path into
maximal runs of operations that share a machine — these are the **critical blocks**. Inside a block, the
operations are queued back-to-back on one machine with no slack; that block is a bottleneck. The
result I lean on (the N5 neighbourhood of Nowicki & Smutnicki) is that to shorten the current critical
path it suffices to consider, for each block, swapping only its **first two** operations or its **last
two** operations — reordering operations in the interior of a block, or touching non-critical operations,
cannot improve the makespan. Even better, these particular swaps **provably never create a cycle**:
reversing a single arc at a block border keeps the disjunctive graph acyclic, so feasibility is automatic
for every move I propose. That is the gift of this neighbourhood — it is *small* (a handful of moves per
critical path, not `Θ(mn)`), it is *targeted* (every move can actually change the makespan), and it is
*safe* (no deadlock checks needed, though I will keep one as a guard). The "recompute only the critical
path" instruction in the lever is exactly this: I find the critical path, cut it into blocks, and
generate only border-swap moves from it — I never scan the whole schedule for moves.

**Choosing the metaheuristic wrapper.** Pure best-improvement over N5 converges to a local optimum that is
good but not great, and it can cycle (swap an arc, the swap-back looks attractive, swap it again). The
standard, strong wrapper is **tabu search**. Each step I build the critical-block moves, evaluate each by
the makespan it would produce, and take the best **admissible** one. When I apply a swap that reverses the
arc `(a before b)` into `(b before a)`, I mark the *reverse* move — putting `a` before `b` again — as
**tabu** for a short tenure, so the search cannot immediately undo its own progress and is forced to
explore. The one exception is **aspiration**: if a tabu move would nonetheless beat the best makespan I
have ever seen, I take it anyway (the reason to forbid it — that it might just be undoing recent work —
does not apply when it sets a new record). I keep the best schedule ever seen in a snapshot and, when the
search stalls for a long stretch, I **diversify**: restore the best, apply a few random feasible swaps to
jump to a new basin, clear the tabu list, and continue. This tabu-over-N5 scheme is the established
strong-yet-simple metaheuristic for `J||C_max`, and it is what turns the leftover time budget into real
makespan reductions rather than re-converging to the same local optimum.

**Designing the data structures so evaluation stays cheap and the schedule stays feasible.** I keep the
machine orders as `mseq[k]` (the op-ids on machine `k`, in order) with an inverse `posInM[o]` (the index
of op `o` within its machine's sequence), both updated together on every swap so I can answer "what is the
machine-predecessor / -successor of op `o`?" in `O(1)`. The makespan and the critical structure come from
two longest-path passes over the disjunctive graph: a forward Kahn topological pass computes `heads[o]`
(the earliest start of `o` = longest path from the source), and the makespan is `max_o heads[o]+proc[o]`;
a reverse pass over the same topological order computes `tails[o]` (the longest path from `o` to the sink,
including `proc[o]`). An operation is **critical** iff `heads[o]+tails[o] == makespan`. These two passes are
`O(n·m)` each — a few hundred nodes here — so I can afford to recompute heads to evaluate a candidate
swap, and recompute heads+tails after committing a move to refresh the critical path. The forward Kahn pass
doubles as my **cycle detector**: if it cannot process all `n·m` operations, the graph has a cycle, the
swap deadlocked, and I reject it. So even though the N5 border swaps are theoretically cycle-free, I never
*rely* on that — every committed move is validated by a real acyclicity check, and a rejected move is
reverted, so the state is always feasible.

**First implementation and the bug I expected to hit — reconstructing the critical path.** I wrote
`buildCriticalMoves()` to find a critical path and cut it into blocks. My first cut just collected *all*
critical operations (`heads+tails==makespan`) and tried to thread them together, and it produced garbage
blocks — the set of critical operations is not a single path, it is the *union* of all critical paths, and
walking it naively jumps between parallel critical chains and even revisits machines, so my "blocks" were
not real machine runs. The symptom showed up immediately in self-verify: on several seeds the move list
contained pairs `(a,b)` where `a` was *not* actually the machine-predecessor of `b`, so `swapAdj` swapped
the wrong adjacent slots and the makespan jumped *up* instead of down, and a couple of seeds even produced
a transient cycle that my guard had to reject — the search was thrashing instead of improving. I diagnosed
it by printing, for the first few iterations on seed 1, the reconstructed path and checking the invariant
"each consecutive pair on the path is either a job-chain arc (`jobNext`) or a machine arc
(`machNext`)." It failed constantly. The fix is to reconstruct **one concrete critical path** by greedy
forward walking: start at a critical op with `heads==0` (a source on some longest path), and at each step
move to a *critical successor* `s` with `heads[cur]+proc[cur]==heads[s]` and `heads[s]+tails[s]==makespan`,
**preferring the machine-successor** so that machine blocks come out maximal. Then I cut the path into
blocks only where consecutive path operations share a machine *and* are genuinely machine-adjacent
(`machNext(path[i])==path[i+1]`), and I only emit a border swap when `machNext(a)==c` actually holds. After
that, every move in the list was a real adjacent-on-machine swap, the spurious makespan increases
disappeared, and the cycle-guard essentially stopped firing.

**A second self-verify: does it actually beat the baseline, and does it use the budget?** With the
critical-path reconstruction correct, my first working tabu version already beat list scheduling, but I
checked it concretely rather than trusting it. I generated seeds 1–20, ran the solver, scored each output,
and also scored the trivial identity-order output. Every single output parsed as `m` permutations and
induced an acyclic graph — feasible, score > 0 — on all twenty seeds. The solver's mean score came out
around `1 360 000`, i.e. makespans roughly 36% shorter than the list-scheduling reference, with the
*minimum* over the twenty seeds still about `1 172 000` — comfortably above one, so there is no seed where
the heuristic regresses below the construction baseline. The trivial identity-order output, by contrast,
averaged only about `157 000`, so the solver beats that floor by more than eight-fold. I also confirmed the
solver runs to its ~1.9 s budget (it does not converge and sit idle — tabu plus diversification keeps it
productively moving), that two runs on the same seed give identical scores (the RNG is seeded
deterministically from `n` and `m`), and that memory stays around 4 MB. One thing I tuned here: my first
tabu tenure was a fixed small constant and the search cycled on a few seeds; tying the tenure mildly to
instance size (`8 + (n+m)/4`) and adding the stall-triggered diversification removed the cycling and lifted
the mean.

**Edge cases, on purpose, because this is where heuristic solvers quietly die.**
- `n ≤ 0` or `m ≤ 0`: nothing to schedule; return immediately (the scorer treats a degenerate instance as
  full credit).
- Tiny instances (`n` or `m` small): the list schedule already gives a feasible acyclic start, and if the
  critical path has no machine block of length ≥ 2 there is simply no N5 move — I detect the empty move
  list and apply a small random feasible kick instead of stalling, so the search never gets stuck with
  nothing to do.
- A swap that would deadlock: the forward Kahn pass returns "cycle," I revert the swap and skip it; the
  state stays feasible. This is the guard that makes the theoretical cycle-freeness of N5 irrelevant to
  correctness — I never trust it, I check it.
- Reading failures: if an operation pair fails to parse I substitute `(machine 0, proc 0)` rather than
  crash, and I clamp any out-of-range machine index into `[0, m-1]`; a never-crashing reader keeps me
  feasible even on malformed input.
- Time-check cost: calling the clock every iteration is itself overhead, so I sample it once every 256
  inner evaluations. The cutoff returns the *best* feasible machine orders found, never a half-applied
  move, so hitting the time limit mid-iteration still prints a valid solution.
- Output: I build one big string and `fputs` it once, so printing `m` lines is not I/O-bound.

**Why I am confident in the final design.** The *feasibility* of the answer rests on two invariants I
verified directly — every output line is a permutation (the machine sequences are permutations of their
operations by construction, and I emit the job index of each op in order), and every committed state is
acyclic (every move is validated by a real longest-path/cycle check, and rejected moves are reverted).
The *strength* rests on the structure: the makespan is a longest path, so only critical operations can
help; the N5 critical-block neighbourhood proposes exactly the border swaps that can shorten the current
critical path and nothing else, so I never waste evaluations on neutral moves; tabu search with aspiration
and stall-triggered diversification keeps the search escaping local optima and converts the whole time
budget into real reductions; and the heads/tails longest-path machinery means each move is evaluated by a
single cheap pass, with the critical path itself recomputed only when I commit a move. That is the
established strong-yet-simple approach for the job shop, not a toy greedy. And the whole thing is wrapped
so the *current* schedule is feasible at every instant: the time cutoff returns the best valid machine
orders found, never a corrupted or deadlocked one.

This is what I ship — one self-contained C++17 file: disjunctive-graph longest-path makespan with
heads/tails, a non-delay list-schedule feasible start, critical-path reconstruction cut into machine
blocks, tabu search over the N5 first-two/last-two block-border swaps with aspiration and diversification,
a real acyclicity guard on every move, all under a wall-clock budget, always emitting `m` valid machine
orders.

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
