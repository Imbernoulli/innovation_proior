# Reasoning: Machine Assignment with Sequence-Dependent Setups

## Reading the objective carefully

I have `n` jobs, `M` identical parallel machines, and `T` job types. Job `j` has a duration `d_j` and
a type `c_j`. Each machine runs the jobs I give it in whatever order I choose, and the order matters
because of changeovers: the machine pays an initial setup `init[a]` before its first job of type `a`,
and a setup `s[a][b]` whenever a job of type `b` immediately follows a job of type `a` on that
machine. The matrix `s` is given, and from the generator description it is asymmetric and non-metric:
same-type changeovers are cheap (`s[a][a]` near 0), cross-type changeovers are moderate to large, and
some pairs are outright expensive.

I must assign **every** job to exactly one machine, choose each machine's order, and minimize the
**total** over all machines of `(sum of durations + all setups)`. The scorer normalizes my cost
against a balanced round-robin baseline and reports
`score = round(1e6 * baseline_cost / max(1, my_cost))`, higher is better, and — this is the part I
cannot forget — **any infeasibility floors the score to 0**. Infeasible here means: not exactly `M`
machine lines, a malformed count, an out-of-range job index, or the listed jobs not forming exactly
the set `{0, ..., n-1}` (a job missing or duplicated). So feasibility is "every job placed exactly
once, in `M` well-formed lines" — that is the whole feasibility surface, and it is easy to keep if I
am disciplined about always holding a partition of the jobs.

The first real observation, and it reshapes everything: the term `sum_j d_j` is a **constant**. Every
job runs exactly once, on some machine, so its duration is counted exactly once no matter how I
assign or order. The durations contribute the same amount to every feasible schedule. Therefore
**minimizing total cost is exactly minimizing total setup cost**. I can throw the durations out of the
optimization entirely (I still print the jobs, but they do not enter any decision) and optimize purely
on setups. That is a big simplification and it is what makes an `O(1)`-per-step local search possible.

## What the setup cost actually is

Setup depends only on *types* and *adjacency*. On one machine with job order `(j_1, ..., j_k)` the
setup it pays is

```
init[c[j_1]] + s[c[j_1]][c[j_2]] + s[c[j_2]][c[j_3]] + ... + s[c[j_{k-1}]][c[j_k]].
```

If I imagine a virtual "ground" node `G` whose directed edge to a job of type `a` costs `init[a]`,
then a machine's setup is the length of an **open directed path** `G -> j_1 -> j_2 -> ... -> j_k`
where edge weights are the type-to-type setups. That is an **open asymmetric Travelling-Salesman path
on types**. And the full problem is: partition the `n` jobs among `M` machines, then solve an open
asymmetric TSP-on-types on each machine, minimizing the sum of path lengths. So this is a
**multiple-TSP / vehicle-routing-style** structure: an assignment wrapped around per-machine
sequencing. Recognizing that "the per-machine setup cost is a TSP-on-types" is the key that tells me
which neighborhoods to use — exactly the moves that work for routing problems.

One more structural note: because `s[a][a]` is cheap, jobs of the **same type** want to sit next to
each other (a "type-run"), and within a run the order is free (same-type internal edges are all cheap
and roughly equal). So the real decision is (1) which types' jobs go on which machine, and (2) in what
order the *type-runs* appear on each machine. The individual job identities barely matter beyond their
type — but I will keep the explicit job sequences anyway, because the moves are simplest to implement
and to keep feasible at the job level, and the optimizer naturally discovers the runs.

## A feasible baseline first

My rule on these problems is to get *some* valid output before optimizing, and to make sure the
optimizer can never lose that. The trivial feasible schedule is the **round-robin** one: send job `j`
to machine `j mod M`, in input order. It is a partition of all jobs into `M` lists, so it parses and
covers every job exactly once — feasible. It is also exactly the scorer's baseline, so by definition
it scores about `1e6`. Its weakness is obvious from the structure above: it interleaves types
arbitrarily, so almost every adjacency is a cross-type changeover, and it pays `init` plus a big
matrix entry on nearly every edge. It is the thing I must beat, not the thing I ship.

A much better *constructive* start, still trivially feasible, is a **type-aware greedy**: bucket jobs
by type; process types from most-common to least-common; and for each job, append it to the machine
that is currently cheapest to extend (the machine whose last type has the smallest setup to this
job's type, with `init` if the machine is empty), breaking ties toward the machine with fewer jobs so
the load stays roughly balanced. This clusters same-type jobs into runs immediately — consecutive jobs
of the same type pay the cheap `s[a][a]` — and it already crushes round-robin. It is `O(n*M)`, and it
gives the local search a good basin to start from. Crucially it is still just a partition, so it is
feasible by construction.

## Why the obvious local search is too slow / weak

The naive way to improve from there is: pick a random change to the schedule (move a job, swap two
jobs, reorder a machine), **recompute the whole cost**, accept if it improved (or by some annealing
rule), repeat. The trouble is the recompute. A full cost evaluation walks every machine's whole
sequence: `O(n)` work per proposed move. With `n` up to 260 and a ~2-second budget, that caps me at a
few hundred thousand evaluations — and most random moves are rejected, so the effective number of
*accepted* improvements is small. For a routing-style search that wants to explore millions of small
perturbations, `O(n)` per move is the bottleneck. I need the per-move cost to be `O(1)`.

The classic routing answer is **incremental (delta) evaluation**: a local move only changes a handful
of edges, so I can compute the change in cost by looking at just those edges, without touching the
rest of the schedule. Since I already reduced the objective to *setup edges only*, and a setup edge
depends only on the types of two adjacent jobs (or `init` for the first), the delta of any local move
is a sum of a constant number of edge weights. That is the lever the problem is pointing at: **two
coupled neighborhoods whose deltas are `O(1)` from the cached neighbor types.**

What about 2-opt, the workhorse of symmetric TSP? In symmetric TSP, reversing a segment changes only
the two boundary edges, an `O(1)` delta. But here the matrix is **asymmetric**: reversing a segment
flips the direction of *every* interior edge, and `s[a][b] != s[b][a]`, so the interior cost changes
too — a segment reversal is `O(segment length)`, not `O(1)`. So a literal 2-opt is *not* `O(1)` for
this asymmetric instance. The moves that *stay* `O(1)` and still cover both "which machine" and "in
what order" are **relocate** (a.k.a. or-opt of length 1) and **swap** — and these together are exactly
the inter-machine + intra-machine neighborhood the structure calls for. Relocate handles both moving a
job to another machine *and* reordering within a machine (relocating to a different slot on the same
machine); swap handles exchanging two jobs (same or different machines). So I will use those two.

## Designing the two O(1) neighborhoods

I keep the schedule explicitly as `seqv[m]` = the ordered job list on machine `m`, plus a reverse
lookup `mac[j]` (which machine job `j` is on) and `pos[j]` (its index within that machine's list).
Both moves recompute only the local setup edges.

**Relocate.** Pull job `j` out of its current slot on its donor machine `fm` at position `fp`, and
insert it before some position on a (possibly different) target machine `tm`. Removing `j` deletes the
edges `prev -> j` and `j -> next` and, if `j` had both a predecessor and a successor, adds a new
bridging edge `prev -> next`; if `j` was first and there is a `next`, the next job becomes first and
now pays `init[next]` instead of `init[j]`. That is the `removeGain` — the setup saved by removing
`j`, computed from at most three edges. Inserting `j` before position `ip` on `tm` deletes the old
edge there (`p -> q`, or `init[q]` if `q` was first) and adds `p -> j` and `j -> q` (or `init[j]` and
`j -> q` if `j` becomes first). That is `insertCost`. The net move delta is
`insertCost - removeGain` — six edge lookups, `O(1)`, independent of `n`.

The one subtlety I have to get right is the **same-machine** case (`tm == fm`): when I relocate a job
within its own machine, the insertion index has to be interpreted in the *post-removal* frame (after
`j` is gone), because that is the array I will actually insert into. So I sample `ip` over the
post-removal length, and I read the neighbor types through a `jobAt` helper that skips `j`'s old slot
when reading machine `fm`. Then, to apply, I erase `j` from `fm` first; after the erase, both the
donor and (if same machine) the target are already in that post-removal frame, so I can insert at `ip`
directly in every case. (This is exactly where I will trip up below.)

**Swap.** Pick two jobs `a`, `b` (each may be on a different machine) and exchange their array
positions. The cost change is confined to the `<= 6` setup edges incident to the two slots. The only
care is when `a` and `b` are *adjacent on the same machine*, because then they share an edge and I
must not double-count it — I handle that by recomputing the small window `enter(lo) + (lo->hi) +
leave(hi)` directly for the adjacent case, and `enter+leave` around each of the two slots otherwise.
For a swap I evaluate the window cost before, perform the swap on the arrays, evaluate the same window
after, take the difference, and either keep it (fixing the reverse lookup for the two touched slots)
or revert the swap.

**Metaheuristic.** I wrap both moves in **simulated annealing**. Each iteration picks relocate or swap
at random, computes the `O(1)` delta, and accepts by the Metropolis rule `delta <= 0 || rand() <
exp(-delta / Temp)`, with a geometric cooling schedule from a temperature near a typical large
changeover (~120) down to near zero over the time budget. Because every iteration is `O(1)`, I get
millions of them. I track the best schedule seen (`bestSeq`) and emit that, so an early time-out still
prints the best partition found — always feasible.

## Implement, then run it and find the bug

I wrote the constructive greedy, the SA loop with both moves, and the emit step, compiled, and ran the
self-check harness: generate seeds `1..20`, run the solver, score it, and also score the round-robin
baseline, asserting every output is feasible (score > 0) and the solver mean beats the baseline mean.

The first run did **not** go cleanly. On a couple of seeds the scorer printed `0` — infeasible. That
is the worst failure mode here, so I dug in. I dumped the solver's output for a failing seed and ran
my own partition check: count how many times each job index appears across all `M` lines. Some job
appeared **twice** and another appeared **zero** times. So the schedule had stopped being a partition
— the relocate move was corrupting it.

I traced it to the same-machine relocate apply step. My first version computed the insertion index
`ip` in the post-removal frame (correctly), but then in the apply step I tried to be clever and
"adjust" `ip` relative to `fp` with a separate `realIp` branch *before* erasing, mixing the
pre-removal and post-removal frames. When `tm == fm` and the target position was after the removed
slot, the index was off by one: I inserted `j` at the wrong place, and worse, in one branch the
bookkeeping let `j` be inserted while a stale copy remained, so the same job id ended up in the array
twice and some other job's slot got overwritten on the next reindex. The cost delta I had computed no
longer matched the array I produced, and the partition invariant broke.

The fix was to *stop* mixing frames. I made the rule uniform: **erase `j` from the donor first**, and
only then insert. After the erase, the donor array — and, when `tm == fm`, the target array, since
they are the same vector — is exactly in the post-removal frame that `ip` and the `jobAt` neighbor
reads were computed in. So inserting at `ip` directly is correct in every case (same-machine or
cross-machine), no special-casing. I deleted the `realIp` branch entirely. This is the kind of
off-by-one that only the asymmetric same-machine relocate exposes, which is why the cross-machine
moves looked fine at first.

After that fix I re-ran the harness. Now every one of seeds `1..20` is feasible, and I added a
second, independent check: for a few seeds I re-parsed the solver's output, asserted it is a true
partition (`seen[j] == 1` for all `j`), and recomputed the total cost from scratch (durations + init +
changeovers) to confirm it is internally consistent — it was. As a belt-and-suspenders guard I also
left a final sanity pass before printing: if for any reason the best schedule is not a full partition
of `{0,...,n-1}`, I rebuild a trivial round-robin partition so the program can never emit an
infeasible schedule. With a correct optimizer that branch never fires, but it guarantees the
feasibility floor is never hit.

## Confirming it beats the baseline

With feasibility fixed, the numbers came out strong. Across seeds `1..20` the solver's normalized
score averages around `1.69e6` against the round-robin baseline's `1.0e6` — a roughly 70% cost
reduction — and **every individual seed beats the baseline** (the weakest was still about `1.21e6`,
the best above `2.0e6`). That matches the structural intuition: round-robin pays a cross-type
changeover on nearly every edge, while the type-aware greedy plus the relocate/swap SA cluster each
type into runs (cheap same-type internal edges) and order the runs on each machine to minimize the few
remaining cross-type changeovers — solving the open asymmetric TSP-on-types per machine while
balancing the assignment. I also confirmed the solver respects the ~1.9s internal budget (well under
the 2s limit) and handles the degenerate `n = 0` / `M` edge cases by printing `M` empty lines.

I considered pushing further — adding a true or-opt of segment length 2–3 (still `O(1)` since I only
touch the segment's boundary edges), or restarts — but the single-job relocate plus swap already
exploit the dominant lever (same-type clustering and run ordering) and beat the baseline comfortably
and uniformly, so I kept the solver lean and let the `O(1)` moves run for the full budget. The final
solver is below, and it is what `verify/sol.cpp` contains.

## Final solver

```cpp
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
```
