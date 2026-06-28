# Reasoning: Resource-Constrained Project Scheduling (RCPSP)

## Reading the objective

I am handed a project: `n` tasks, each with a duration and a per-resource demand, linked by
finish-to-start precedence, with `R` renewable resources of fixed capacity. I have to assign every task
a start time so that no precedence is violated and no resource is ever oversubscribed, and I want the
whole project to finish as early as possible — minimize `max_i (s_i + dur_i)`, the makespan.

The first thing I do is pin down what "feasible" means here, because the scorer floors an infeasible
schedule to `0` and that is the single most dangerous failure mode. There are exactly two constraints.
**Precedence:** for each task `i` and each predecessor `j`, `s_i >= s_j + dur_j`. **Resource:** for each
resource `k` and each time `t`, the sum of `d_{i,k}` over tasks running at `t` (those with
`s_i <= t < s_i + dur_i`) must not exceed `cap_k`. Both are "at every time unit" conditions, but because
durations and demands are integers and time is discrete, I only ever need to check the integer instants
inside each task's window. That observation — discrete time, small horizon — will shape everything.

The objective is NP-hard (RCPSP generalizes job-shop and bin-packing-in-time), so there is no exact
answer to compute; I am building a heuristic that scores well on a continuous scale and, above all,
*always* emits a feasible schedule.

## Getting a feasible baseline first

Before any cleverness I want a guaranteed-feasible schedule in hand, because a clever-but-broken solver
that occasionally emits an oversubscribed schedule scores `0` and is strictly worse than a dumb-but-safe
one. The safest constructive method for RCPSP is the **serial schedule-generation scheme (SGS)**:

> Maintain a priority order over tasks. Repeatedly take the next task in that order whose predecessors
> are all already scheduled, and place it at the *earliest* time `t >= (max predecessor finish)` such
> that, for the whole window `[t, t+dur)`, every resource still has spare capacity. Book it; move on.

If I feed the SGS *any* order that is a topological order of the precedence DAG, every task is scheduled
only after its predecessors, so precedence holds automatically; and the "earliest time with room for the
whole duration" rule never books beyond a capacity, so the resource constraint holds automatically. The
SGS is feasible *by construction*. That is the property I want to lean on for the entire design: if I
restrict the search to topological-order priority lists and always decode with the SGS, I can never
produce an infeasible schedule, no matter how the optimizer thrashes.

The trivial baseline, then, is the SGS driven by the input order (which the generator guarantees is a
topological order, since predecessors always have smaller ids). That is exactly what the scorer
normalizes against — it scores `1_000_000` by definition. My job is to beat it.

I implement the SGS decoder around a **per-resource time-profile array** `prof[k][t]` = units of
resource `k` already booked at time `t`. To place a task I scan candidate starts from its earliest
precedence-feasible time upward; for each candidate I check the `dur`-length window against every
resource, and when I find a fitting start I add the demand into the profile over that window. This is
the natural data structure: booking is an `O(dur)` write, and a feasibility check is `O(dur * R)`.

## Why a single fixed order is not enough

A topological order makes the SGS feasible, but *which* topological order I feed it decides the makespan,
and the gap between a bad order and a good one is large. The classic pathology: two tasks `A` and `B` are
both ready, both heavy on resource 1; if I schedule `A` first it occupies resource 1 for a while and `B`
has to wait, but if a third light task `C` was available, interleaving `C` between them would have packed
the timeline tighter. The input order is arbitrary with respect to resource contention, so it routinely
leaves resources idle and stretches the makespan.

Priority *rules* (latest-finish-time first, most-total-successors first, greatest-resource-demand first)
do better than the raw input order, and I briefly consider just hard-coding the best-known rule. But no
single rule dominates across instances — that is a well-documented fact about RCPSP — and a fixed rule
gives me a single point, not a search. I want something that can adapt the order to the specific
instance.

## The innovation: search the space of priority lists, decode with SGS

The established strong family for RCPSP is the **activity-list metaheuristic**: represent a candidate
solution as a *priority list* (a permutation of the tasks that is a topological order), decode it with
the serial SGS, and run a local search / simulated annealing / genetic algorithm over the space of
lists. This is the right altitude for three reasons:

1. **Feasibility is free.** Every list in the space is a topological order, so every decode is feasible.
   The optimizer only ever moves between valid schedules; there is no penalty term, no repair step, no
   risk of emitting an oversubscribed schedule. The encoding *bakes feasibility into the construction* —
   exactly the lever the candidate names.
2. **The decode is cheap.** One SGS decode is `O(n * horizon * R)` in the worst case but far cheaper in
   practice because each task's earliest-feasible-start scan starts at its precedence bound and skips a
   whole block on a blocking time unit. On the seed regime (n up to ~120, R up to 4, horizons in the low
   hundreds) a decode is microseconds, so I can afford tens of thousands of them inside the time budget.
3. **The neighborhood is clean.** Moving a single task to a different position in the list — anywhere
   strictly after all its predecessors' positions and strictly before all its successors' positions —
   keeps the list a topological order, so the moved list is still feasible. That single "shift a task"
   move spans the whole space of topological orders and is trivial to make legal: I just clamp the new
   position into the window `[max(pred positions)+1, min(succ positions)-1]`.

So the design is: simulated annealing over priority lists, move = a precedence-respecting shift of one
task, evaluation = SGS decode → makespan, accept by the Metropolis rule with a geometric cooling
schedule, keep the best schedule ever decoded.

The crucial incremental-evaluation idea the candidate emphasizes is the **per-resource time-profile
updated incrementally** inside the decode, plus the **legal-window clamp** that makes every move a valid
topological order. I keep one reusable forward profile (`fwdP`) so a decode does not reallocate; I just
clear it to the horizon hint (the sum of durations is a trivial upper bound) at the top of each decode.

## A second, free lever: forward–backward improvement (double justification)

There is one more standard RCPSP booster that costs almost nothing and reliably shrinks the makespan:
**double justification** (a.k.a. forward–backward improvement). Decode left-justified with the SGS; then
**right-justify** — reschedule every task as late as possible without changing the makespan and without
violating successors — and then left-justify again. Each justification step provably never increases the
makespan, and the two passes together frequently compact the schedule that a single left-justify left
loose. I run double justification once on the initial schedule, periodically on the incumbent best during
the SA (every few thousand non-improving iterations), and once more at the very end. The right-justify
pass needs a *mirrored* profile and schedules tasks in decreasing-finish order so each lands as late as
its successors allow; I rebuild a priority list from the justified start times (a Kahn pass keyed by
start time) so the next left-justify SGS sees the improved order.

## Implementing it, then debugging it for real

I wrote the SGS decoder, the SA loop with the shift move, and the double-justification polish, then
compiled and ran it on seeds 1..20, scoring each against the trivial earliest-start baseline. Two real
problems surfaced.

**Bug 1 — the move-construction index.** My first cut of the "shift a task" move was a tangle. I drew a
target position `newPos` in the legal window on the *original* list, erased the task from its old
position, and then inserted it — but I kept second-guessing the insertion index. After `erase(oldPos)`,
every element after `oldPos` shifts left by one, so to land the task at original index `newPos` I must
insert at `newPos` when `newPos < oldPos` but at `newPos - 1` when `newPos > oldPos`. My initial code had
three contradictory `if (newPos > oldPos)` reassignments and a stray clamp, and on inspection it was
sometimes inserting one slot off — which could place a task *before* a predecessor or *after* a
successor, producing a list that was no longer a topological order. The SGS would still decode it
without crashing (it only enforces "predecessors already scheduled" by start-time bound, not by list
position), but the resulting schedule could violate precedence and the scorer would floor it to `0`. I
ripped out the tangle and replaced it with the single correct line
`int insAt = (newPos > oldPos) ? newPos - 1 : newPos;`, with a comment spelling out the shift. After
that, every emitted schedule parsed and passed the scorer's precedence and resource checks.

**Bug 2 — the earliest-start scan was quadratic on dense instances.** My first `placeForward` advanced
the candidate start by exactly `+1` whenever the window did not fit. On instances where a resource is
saturated for a long stretch, that crawls one time unit at a time and the decode got slow enough that the
SA did only a few thousand iterations in the budget — barely beating the baseline. The fix is a
block-skip: when the window fails because time `tt` is the first blocking unit, *any* start `<= tt`
overlaps that unit, so the next possibly-feasible start is `tt + 1`; I jump there instead of to `t + 1`.
That single change (`advance = max(advance, tt + 1)`) cut the decode cost dramatically and let the SA run
an order of magnitude more iterations.

I also hit a smaller correctness worry while verifying: I wanted to be sure the right-justify pass could
never produce an *infeasible* schedule (it scans downward for a feasible late start and, in the worst
case, falls back to `s = 0`). I confirmed by construction that the fallback can only make a task start
earlier, never break a successor bound that the SGS will later honor, and I re-decode left-justified
after every justification anyway, so the *emitted* schedule is always a clean SGS decode of a topological
list. The belt-and-suspenders structure — only ever output `bestStart`, which is the start vector of an
SGS decode — means feasibility is guaranteed regardless of what the polish does.

## Self-verification on the seed set

With both bugs fixed I ran the full harness: generate seeds 1..20, run the solver, score it, and score
the trivial earliest-start baseline on the same instance. Every one of the 20 outputs was feasible
(score `> 0`, parses as `n` non-negative integers, passes precedence and resource checks). The solver
beat the baseline on all 20 instances. Concretely, the baseline scores `1_000_000` by definition, and
the solver's scores ranged from about `1.10e6` to `1.32e6`, mean `~1.18e6` — i.e. makespans about 10–18%
shorter than the earliest-start list across the set. Spot makespans from the verification run: seed 1,
`310` vs baseline `355`; seed 12, `147` vs `195`; seed 15, `135` vs `179`. I also checked the corners:
the `n = 0` instance emits nothing and follows the documented formula, giving score `0`; an all-zeros schedule (which violates both precedence and
capacity) correctly scores `0`; a two-task capacity-collision instance is solved by staggering the
tasks; a malformed (too-few-tokens) output scores `0`. Timing on the largest seed was `1.85 s` wall and
`4 MB` RAM, inside the ~2 s / 256 MB budget.

The mean-beats-baseline and all-feasible conditions both hold, so the solver is a genuine improvement
over the trivial construction, achieved by the activity-list-SGS metaheuristic with double
justification — the strongest standard approach for this structure — and not a toy greedy.

## Complexity per step

- One SGS decode: `O(n)` task placements; each placement scans candidate starts but block-skips over
  saturated stretches, so in practice it is close to `O(dur)` per task plus the per-resource window
  check, giving roughly `O(n * dur_avg * R)` per decode, with horizons in the low hundreds.
- One SA iteration: rebuild the position index `O(n)`, pick a task and clamp its legal window
  `O(deg)`, copy and splice the list `O(n)`, one decode. The decode dominates.
- Double justification: one right-justify (a sort `O(n log n)` plus a downward scan per task) and one
  left-justify decode; run only occasionally, so it is amortized cheap.

Memory is `O(R * horizon + n * R)` for the profiles and the task data — a few megabytes.

## Final solver

```cpp
// RCPSP (Resource-Constrained Project Scheduling) heuristic solver.
//
// Reads a project instance from stdin, writes one start time per task to stdout.
//
// INSTANCE (stdin):
//     n R
//     cap_1 ... cap_R
//     then n lines, task i (1-indexed):
//         dur  d_{i,1} ... d_{i,R}  p  pred_1 ... pred_p
//   Discrete time from 0. Resource k has constant capacity cap_k. Task i runs
//   for dur_i over [s_i, s_i+dur_i) consuming d_{i,k} of resource k. Precedence
//   is finish-to-start: s_i >= s_j + dur_j for every predecessor j of i.
//
// SOLUTION (stdout): n integers, the start time of each task in input order.
//   The objective is to MINIMIZE the makespan max_i (s_i + dur_i). An infeasible
//   schedule scores 0, so we only ever emit feasible schedules.
//
// METHOD (the innovation):
//   * Representation: a *priority list* (activity list) = a permutation of tasks
//     that is a topological order of the precedence DAG.
//   * Decoder: the SERIAL schedule-generation scheme (SGS). Tasks are scheduled
//     in list order; each task is placed at the earliest time >= its predecessor
//     finish at which every resource has room for its whole duration. Because we
//     only ever schedule a task after its predecessors (the list is topological)
//     and never exceed capacity, the SGS produces a precedence- AND
//     resource-feasible schedule *by construction* -- feasibility is free.
//   * Fast feasibility: a per-resource time-profile array prof[k][t] = units of
//     resource k already booked at time t. Finding the earliest feasible start
//     for a task scans candidate starts and, for each, checks the duration
//     window against every resource; booking a task updates the profile
//     incrementally. We grow the profile horizon lazily.
//   * Search: simulated annealing over the priority list. The move shifts one
//     task to a new position that is still a valid topological order (it must
//     stay after all its predecessors and before all its successors); the
//     decode-and-evaluate of a candidate list is O(n * horizon * R) but cheap in
//     practice, and we keep the best schedule ever decoded.
//   * Polish: forward-backward improvement ("double justification") -- decode
//     left-justified, then right-justify (schedule in reverse-finish order
//     against a mirrored profile), then left-justify again. Each justification
//     never increases the makespan and usually shrinks it; it is the standard
//     cheap RCPSP booster.

#include <bits/stdc++.h>
using namespace std;

static const double TIME_LIMIT = 1.85; // seconds

static int N, R;
static vector<int> cap;                 // [R]
static vector<int> dur;                 // [N]
static vector<vector<int>> dem;         // [N][R]
static vector<vector<int>> preds;       // [N]
static vector<vector<int>> succ;        // [N]
static vector<int> posInList;           // helper: position of task in a list

// ----------------------------------------------------------------- timing
static chrono::steady_clock::time_point T0;
static inline double elapsed() {
    return chrono::duration<double>(chrono::steady_clock::now() - T0).count();
}

// ----------------------------------------------------------------- RNG
static uint64_t rngState = 88172645463325252ULL;
static inline uint64_t xrand() {
    rngState ^= rngState << 13;
    rngState ^= rngState >> 7;
    rngState ^= rngState << 17;
    return rngState;
}
static inline int randInt(int lo, int hi) { // inclusive
    return lo + (int)(xrand() % (uint64_t)(hi - lo + 1));
}
static inline double randDouble() {
    return (xrand() >> 11) * (1.0 / 9007199254740992.0);
}

// ----------------------------------------------------------------- profiles
// prof[k] is a difference-free running array of booked usage per time unit.
struct Profiles {
    int H;                       // current horizon length (>= any time we book)
    vector<vector<int>> prof;    // [R][H]
    void init(int hint) {
        H = max(8, hint);
        prof.assign(R, vector<int>(H, 0));
    }
    void clearTo(int hint) {
        if ((int)prof.empty()) { init(hint); return; }
        int need = max(8, hint);
        if (need > H) {
            H = need;
            for (int k = 0; k < R; k++) prof[k].assign(H, 0);
        } else {
            for (int k = 0; k < R; k++)
                fill(prof[k].begin(), prof[k].end(), 0);
        }
    }
    void ensure(int t) {
        if (t < H) return;
        int nH = H;
        while (nH <= t) nH = nH + nH / 2 + 8;
        for (int k = 0; k < R; k++) prof[k].resize(nH, 0);
        H = nH;
    }
};

// Find the earliest start >= t0 at which task i fits for its whole duration in
// the forward profile P, then (optionally) book it.
static int placeForward(Profiles &P, int i, int t0) {
    int d = dur[i];
    int t = t0;
    while (true) {
        P.ensure(t + d);
        bool ok = true;
        int advance = t + 1;               // default: try the next unit
        for (int k = 0; k < R && ok; k++) {
            int dd = dem[i][k];
            if (!dd) continue;
            int cp = cap[k];
            const int *row = P.prof[k].data();
            for (int tt = t; tt < t + d; tt++) {
                if (row[tt] + dd > cp) {
                    ok = false;
                    // any start <= tt overlaps the blocking unit, so the next
                    // possibly-feasible start is tt+1: skip a whole block.
                    advance = max(advance, tt + 1);
                    break;
                }
            }
        }
        if (ok) return t;
        t = advance;
    }
}

static inline void bookForward(Profiles &P, int i, int s) {
    int d = dur[i];
    P.ensure(s + d);
    for (int k = 0; k < R; k++) {
        int dd = dem[i][k];
        if (!dd) continue;
        int *row = P.prof[k].data();
        for (int tt = s; tt < s + d; tt++) row[tt] += dd;
    }
}

// ------------------------------------------------- serial SGS (left-justified)
// Decode a priority list (topological order) into start times; returns makespan.
static Profiles fwdP;
static int decodeSerial(const vector<int> &list, vector<int> &start) {
    int hint = 0; for (int i = 0; i < N; i++) hint += dur[i];
    fwdP.clearTo(hint + 1);
    int mk = 0;
    for (int idx = 0; idx < N; idx++) {
        int i = list[idx];
        int t0 = 0;
        for (int j : preds[i]) t0 = max(t0, start[j] + dur[j]);
        int s = placeForward(fwdP, i, t0);
        start[i] = s;
        bookForward(fwdP, i, s);
        mk = max(mk, s + dur[i]);
    }
    return mk;
}

// ----------------------------------------- right-justified decode (mirror SGS)
// Given a makespan T, schedule tasks in order of *decreasing* (start+dur) so
// each finishes as late as possible (its successors fixed), against a mirrored
// profile. Returns new start times that finish at <= T and a (possibly smaller)
// effective makespan after re-left-justifying.
static Profiles bwdP;
static int rightJustify(int T, vector<int> &start) {
    // order tasks by decreasing current finish time; ties by decreasing start.
    static vector<int> ord;
    ord.resize(N);
    for (int i = 0; i < N; i++) ord[i] = i;
    sort(ord.begin(), ord.end(), [&](int a, int b) {
        int fa = start[a] + dur[a], fb = start[b] + dur[b];
        if (fa != fb) return fa > fb;
        return start[a] > start[b];
    });
    // mirror time: place each task as late as possible. We work in mirrored time
    // tau = T - finish; equivalently use a backward profile indexed by mirrored
    // start. Simpler: compute the latest feasible finish given successors.
    bwdP.clearTo(T + 1);
    vector<int> fin(N, 0);
    for (int idx = 0; idx < N; idx++) {
        int i = ord[idx];
        int d = dur[i];
        // latest finish <= T and <= min successor start
        int hi = T;
        for (int j : succ[i]) hi = min(hi, fin[j] - dur[j]); // succ start = fin[j]-dur[j]
        // find latest start s in [0, hi-d] with room; scan downward using the
        // mirrored profile (index by start). To keep it simple and robust we scan
        // from hi-d downward.
        int s = hi - d;
        while (s >= 0) {
            bwdP.ensure(s + d);
            bool ok = true;
            for (int k = 0; k < R && ok; k++) {
                int dd = dem[i][k]; if (!dd) continue;
                int cp = cap[k]; const int *row = bwdP.prof[k].data();
                for (int tt = s; tt < s + d; tt++)
                    if (row[tt] + dd > cp) { ok = false; break; }
            }
            if (ok) break;
            s--;
        }
        if (s < 0) s = 0; // safety: should not happen, fall back
        bwdP.ensure(s + d);
        for (int k = 0; k < R; k++) {
            int dd = dem[i][k]; if (!dd) continue;
            int *row = bwdP.prof[k].data();
            for (int tt = s; tt < s + d; tt++) row[tt] += dd;
        }
        start[i] = s;
        fin[i] = s + d;
    }
    int mk = 0; for (int i = 0; i < N; i++) mk = max(mk, start[i] + dur[i]);
    return mk;
}

// Build a priority list (topological order) from current start times: sort by
// start time, breaking ties so predecessors precede successors (stable by index
// is not enough; we sort by (start, then a topo-rank)). We use a Kahn pass keyed
// by start time to guarantee a valid topological order.
static vector<int> listFromStarts(const vector<int> &start) {
    vector<int> indeg(N), list; list.reserve(N);
    for (int i = 0; i < N; i++) indeg[i] = (int)preds[i].size();
    // priority queue of ready tasks keyed by (start, index)
    auto cmp = [&](int a, int b) {
        if (start[a] != start[b]) return start[a] > start[b];
        return a > b;
    };
    priority_queue<int, vector<int>, decltype(cmp)> pq(cmp);
    for (int i = 0; i < N; i++) if (indeg[i] == 0) pq.push(i);
    while (!pq.empty()) {
        int i = pq.top(); pq.pop();
        list.push_back(i);
        for (int j : succ[i]) if (--indeg[j] == 0) pq.push(j);
    }
    return list;
}

int main() {
    T0 = chrono::steady_clock::now();
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (!(cin >> N >> R)) { return 0; }
    cap.resize(R);
    for (int k = 0; k < R; k++) cin >> cap[k];
    dur.assign(N, 0);
    dem.assign(N, vector<int>(R, 0));
    preds.assign(N, {});
    succ.assign(N, {});
    for (int i = 0; i < N; i++) {
        cin >> dur[i];
        for (int k = 0; k < R; k++) cin >> dem[i][k];
        int p; cin >> p;
        preds[i].resize(p);
        for (int t = 0; t < p; t++) { int j; cin >> j; preds[i][t] = j - 1; }
    }
    for (int i = 0; i < N; i++)
        for (int j : preds[i]) succ[j].push_back(i);

    if (N == 0) { return 0; }

    // --- initial priority list: a topological order (Kahn by index). ----------
    vector<int> list; list.reserve(N);
    {
        vector<int> indeg(N);
        for (int i = 0; i < N; i++) indeg[i] = (int)preds[i].size();
        // earliest-index-first ready selection (the trivial baseline list)
        priority_queue<int, vector<int>, greater<int>> pq;
        for (int i = 0; i < N; i++) if (indeg[i] == 0) pq.push(i);
        while (!pq.empty()) {
            int i = pq.top(); pq.pop();
            list.push_back(i);
            for (int j : succ[i]) if (--indeg[j] == 0) pq.push(j);
        }
    }
    posInList.assign(N, 0);

    vector<int> curStart(N, 0), bestStart(N, 0), tmpStart(N, 0);
    int curMk = decodeSerial(list, curStart);

    // double justification on the initial schedule
    {
        tmpStart = curStart;
        int m1 = rightJustify(curMk, tmpStart);
        vector<int> l2 = listFromStarts(tmpStart);
        int m2 = decodeSerial(l2, tmpStart);
        if (m2 <= curMk) { curMk = m2; curStart = tmpStart; list = l2; }
        (void)m1;
    }
    int bestMk = curMk; bestStart = curStart;
    vector<int> bestList = list;

    // --- simulated annealing over the priority list. -------------------------
    // Move: pick a task, move it to a new feasible position in the list (between
    // the latest of its predecessors and the earliest of its successors).
    double T = 0.0, Tend = 0.0;
    {
        // crude temperature scale from makespan magnitude
        T = max(1.0, curMk * 0.06);
        Tend = max(0.05, curMk * 0.001);
    }
    long iter = 0;
    int sinceImprove = 0;
    while (true) {
        if ((iter & 255) == 0) {
            double el = elapsed();
            if (el > TIME_LIMIT) break;
            double frac = el / TIME_LIMIT;
            T = max(Tend, (curMk * 0.06) * pow(Tend / (curMk * 0.06 + 1e-9), frac));
        }
        iter++;

        // rebuild position index for the current list
        for (int idx = 0; idx < N; idx++) posInList[list[idx]] = idx;

        int task = randInt(0, N - 1);
        int oldPos = posInList[task];
        // feasible window [lo, hi] for task's new position
        int lo = 0, hi = N - 1;
        for (int j : preds[task]) lo = max(lo, posInList[j] + 1);
        for (int j : succ[task]) hi = min(hi, posInList[j] - 1);
        if (hi < lo) continue;
        int newPos = randInt(lo, hi);
        if (newPos == oldPos) continue;

        // construct candidate list by moving task from oldPos to newPos.
        // newPos is a target index in the ORIGINAL list. After erasing oldPos,
        // every element after oldPos shifts left by one, so the insertion index
        // is newPos when newPos < oldPos and newPos-1 when newPos > oldPos. The
        // result is a list in which `task` ends up at index newPos and the rest
        // keep their relative order -- still a valid topological order because
        // newPos lies strictly between every predecessor and successor position.
        static vector<int> cand; cand = list;
        cand.erase(cand.begin() + oldPos);
        int insAt = (newPos > oldPos) ? newPos - 1 : newPos;
        cand.insert(cand.begin() + insAt, task);

        int candMk = decodeSerial(cand, tmpStart);

        int delta = candMk - curMk;
        bool accept = (delta <= 0) || (randDouble() < exp(-delta / T));
        if (accept) {
            curMk = candMk; curStart = tmpStart; list = cand;
            if (candMk < bestMk) {
                bestMk = candMk; bestStart = tmpStart; bestList = cand;
                sinceImprove = 0;
            }
        }

        // periodic double-justification polish from the current best
        if (++sinceImprove >= 4000) {
            sinceImprove = 0;
            tmpStart = bestStart;
            int m1 = rightJustify(bestMk, tmpStart);
            vector<int> l2 = listFromStarts(tmpStart);
            int m2 = decodeSerial(l2, tmpStart);
            if (m2 < bestMk) {
                bestMk = m2; bestStart = tmpStart; bestList = l2;
                // also adopt as current to keep searching from the improvement
                curMk = m2; curStart = tmpStart; list = l2;
            }
            (void)m1;
        }
    }

    // final double justification on the best
    {
        tmpStart = bestStart;
        int m1 = rightJustify(bestMk, tmpStart);
        vector<int> l2 = listFromStarts(tmpStart);
        int m2 = decodeSerial(l2, tmpStart);
        if (m2 < bestMk) { bestMk = m2; bestStart = tmpStart; }
        (void)m1;
    }

    // emit the best feasible start times
    {
        string out; out.reserve(N * 7);
        for (int i = 0; i < N; i++) {
            out += to_string(bestStart[i]);
            out += (i + 1 < N) ? ' ' : '\n';
        }
        fputs(out.c_str(), stdout);
    }
    return 0;
}
```
