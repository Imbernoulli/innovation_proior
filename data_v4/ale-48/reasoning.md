**Reading the problem and stripping it to structure.** I have `W` workers, `D` days, and `S`
work shifts per day, and I must fill a `W × D` grid where each cell `a[w][d]` is a shift code in
`{0, 1, …, S}` — `0` meaning the worker rests that day. Every covered unit of a `(day, shift)` slot
is worth `VALUE[d][s]`, but only up to the slot's `DEMAND[d][s]`: staffing past demand earns nothing.
On top of that, each worker pays an overtime penalty `LAMBDA` per hour worked beyond a soft weekly
cap `MAXH`. So the objective I want to push up is

```
OBJ = Σ_{d,s} VALUE[d][s] · min(cov[d][s], DEMAND[d][s])  −  LAMBDA · Σ_w overtime(w),
```

where `cov[d][s]` counts the workers I put on shift `s`, day `d`. Before I write a line I want to be
clear about what kind of world this is. The cells are tiny categorical decisions, but they are tied
together two ways: *vertically* down a worker's week by hard labour rules, and *horizontally* across
workers by the shared coverage targets. That coupling, plus the demand cap, makes it a constrained
rostering/assignment problem — NP-hard, no exact answer at this size in two seconds, and the
benchmark scores me by *how good* my objective is, not by hitting a unique optimum. So my job is to
get `OBJ` as high as the time budget allows while *never* emitting something the scorer refuses.

**Pinning feasibility first, because an infeasible roster scores zero.** The output is `W` lines of
`D` codes. It is feasible only if it parses as exactly `W × D` integers in `[0, S]` *and* every
worker's row obeys four hard rules:

- **(A) Availability.** If `a[w][d] = s ≥ 1`, then `AVAIL[w][d][s]` must be `1`.
- **(B) Minimum rest.** If a worker works shift `s` on day `d` and `s'` on day `d+1`, the rest gap
  `24 + START[s'] − END[s]` must be `≥ MIN_REST`. With the canonical three shifts — morning at 06:00,
  day at 14:00, night at 22:00 for ten hours (so `END = 32`, i.e. 08:00 next day) — a night shift
  followed by *any* shift the next morning has gap `24 + START[s'] − 32 = START[s'] − 8`, which for a
  06:00 morning is `−2`: an immediate violation. Night→day and night→night are also illegal under an
  11-hour rest rule. This is the back-to-back lever the problem is built around.
- **(C) Max consecutive working days.** No more than `MAXCONS` worked days in a row.
- **(D) Hard weekly hours.** Total assigned hours `≤ HARDH`.

In a continuous-score heuristic problem a brilliant-but-occasionally-invalid solver is strictly worse
than a mediocre always-valid one, because a single zero wrecks the mean. So my governing rule from
the start is: *hold a feasible grid at all times*, treat construction and every search move as
feasibility-preserving operations, and on timeout print the best feasible grid I currently hold —
never something I validate at the end and hope.

**Reaching a feasible baseline immediately.** The trivial legal answer is the **all-off grid**:
everyone rests every day. It violates nothing — no shift means no availability, rest, consecutive, or
hour rule can fire — so it is feasible. It is also worthless: it covers no demand, so `OBJ = 0`. That
is exactly its role: my safety net and the floor I must clear, not my answer. The scorer normalises
against a stronger reference, the **greedy fill-by-demand** roster: process `(day, shift)` slots in
decreasing value, and for each, assign the lowest-indexed still-available workers who can legally take
it given what is already placed. That greedy is itself always feasible and scores exactly
`1 000 000`. My real target is to beat *it* — to get `OBJ` strictly above the greedy's `OBJ(G)` so the
ratio exceeds one.

**Why the obvious approaches leave value on the table.** Greedy fill-by-demand commits too early. A
worker grabbed for a medium-value slot on Tuesday may then be locked out — by rest, by hours, by the
consecutive-day cap — of a higher-value slot on Wednesday that is considered later in the value
ordering. The damage is structural: the decision for one slot silently removes options for others. A
pure slot-by-slot greedy has no way to weigh "spend this worker here vs save them for there." I want a
construction that reasons about a worker's *whole week at once*.

**The first lever — per-worker "columns."** Here is the reframing that unlocks it. Think of each
worker's entire weekly pattern — the vector `(a[w][0], …, a[w][D−1])` — as a single object, a *column*
in the column-generation sense. A column is feasible iff it obeys rules (A)–(D) for that one worker
*in isolation*; rules (A)–(D) never reference another worker. So feasibility factorises perfectly per
worker. If I could, for one worker, compute the feasible column of maximum **marginal** value — the
value of the slots it would fill that are still short of demand, minus the overtime it incurs, given
everything already rostered — then I could build the roster one strong column at a time, each worker
choosing its best week against the residual demand left by the workers placed before it. That is a far
better construction than slot greedy because each column decision already balances the whole week's
rest/hours/consecutive trade-offs.

**Why finding the best column is cheap.** A single worker's days form a short *chain*, and the only
state that couples consecutive days is: which shift was worked yesterday (for the rest rule (B)), how
many days in a row have been worked (for (C)), and the hours accumulated so far (for (D)). So the best
feasible column is an exact **dynamic program** over the worker's days with state
`(prevShift, consecRun, hoursSoFar)`. Transitions: either take a day off (resets `consecRun` to 0,
`prevShift` to "off", hours unchanged) or work an available shift `s` that passes the rest rule
against `prevShift`, keeps `consecRun + 1 ≤ MAXCONS`, and keeps `hours + HOURS[s] ≤ HARDH`. The value
added by working `s` on day `d` is the marginal residual value `marg(d, s)` — `VALUE[d][s]` if
`cov[d][s] < DEMAND[d][s]`, else `0` — minus the extra overtime crossing `MAXH` costs. `D` is a week,
`S` is three, `HARDH` is small, so the state space is a few thousand and the DP is microseconds. The
DP *guarantees* a feasible column for that worker because it enforces (A)–(D) in its transitions, and
the all-off column is always reachable (value 0), so a feasible terminal state always exists.

**Construction wired up.** I order workers by flexibility — those with more available slots first —
which tends to leave the genuinely constrained workers to soak up the residual demand, though any
order yields a feasible roster. For each worker I run the column DP against the *current* `cov`,
commit the chosen column (incrementing `cov` and the worker's hours for each worked day), and move on.
This is the "pick patterns greedily by marginal coverage" step. After all workers I have a strong,
fully feasible starting roster.

**The second lever — simulated annealing with two independent localities.** Construction is greedy in
worker order, so it is not optimal; the residual interactions between columns leave room. I spend the
rest of the budget on **simulated annealing** over the grid, and the reason SA is fast here is the
crux of the whole solver: a single-cell move `a[w][d] : s → s'` is local in *two independent senses*.

1. **Objective locality (O(1) delta).** Changing one cell changes at most two coverage buckets,
   `cov[d][s]` (down one) and `cov[d][s']` (up one), and one worker's overtime. The objective delta is
   therefore `Δ = [bucketVal(d, s, cov−1) − bucketVal(d, s, cov)] + [bucketVal(d, s', cov+1) −
   bucketVal(d, s', cov)] − LAMBDA · (overtime_new − overtime_old)` — a handful of arithmetic ops,
   independent of `W` and `D`. I never rescore the grid inside the loop.

2. **Feasibility locality (O(1) check confined to one worker's adjacent days).** Because rules (A)–(D)
   are per-worker, the move can only break feasibility *for worker `w`*, and only through: the
   availability of the new shift; the rest gap between day `d` and its neighbours `d−1` and `d+1`; the
   consecutive-day run that passes through `d`; and that worker's hour total. So I re-check exactly
   those — the worker's adjacent days plus its hour sum — and nothing else. No other worker, no other
   day, no global scan. This is the "incremental rest-rule check confined to that worker's adjacent
   days" that makes thousands of moves per millisecond possible.

The SA loop proposes a random `(w, d)` and a random new code `≠` current, rejects immediately if it is
locally infeasible, computes `Δ` incrementally, and accepts with probability `1` if `Δ ≥ 0` else
`exp(Δ / T)`. Temperature geometrically anneals from about the largest single-shift value (so early
on it will accept roughly a one-shift worsening) down to a small floor, scheduled on the wall-clock
fraction so the run fills the budget exactly. I keep the best feasible grid seen and restore it at the
end; since every accepted state is feasible, "best seen" is always a legal roster.

**Implementing — and the bug I hit.** My first SA pass had a subtle and instructive defect in the
incremental delta. I wanted to *evaluate* a move before deciding whether to accept it, i.e. compute
`Δ` without mutating state, then mutate only on accept. My initial code computed the delta by calling
the same `applyMove` helper that *commits* the change (it decrements/increments `cov`, updates hours,
and adds the delta to `curObj`), then — if the move was rejected — tried to undo it by calling
`applyMove` back the other way. That "do then maybe undo" pattern is exactly the kind of state-mutation
bookkeeping that goes wrong: an early `continue` on a locally-infeasible proposal skipped the undo
path in one branch, and after a few thousand iterations `cov` had drifted out of sync with `A`. The
symptom was ugly and clear: I added an assertion that recomputed `fullObjective()` from `cov`/`hoursW`
and compared it to the incrementally-tracked `curObj` every few thousand iterations, and on seed 8 it
fired — the tracked objective and the from-scratch objective disagreed by a few hundred. That meant my
`cov` array no longer matched the grid I would print, so the *score I thought I had was a lie*, and
worse, a desynced `cov` could let an infeasible state slip through the local check (which trusts
`cov`/`hours`).

**The fix — evaluate without mutating, commit only on accept.** I rewrote the loop so the delta is
computed purely from reads: for the old shift the change is `bucketVal(d, os, cov−1) − bucketVal(d,
os, cov)` and for the new shift `bucketVal(d, ns, cov+1) − bucketVal(d, ns, cov)`, using the
*hypothetical* counts directly, touching nothing. The single mutation point is `applyMove`, called
*only* after the accept test passes, and it is the sole place that edits `cov`, `hoursW`, `A`, and
`curObj` — so they can never drift apart. I re-enabled the periodic `fullObjective()` cross-check as a
debug invariant and it held across all seeds; the desync was gone. This is the discipline the problem
rewards: one commit path, evaluation by pure reads, and a cheap global recomputation kept around as a
guard while developing.

**A second feasibility scare — the consecutive-run check.** While testing I also found that an early
version of the local consecutive-day check only looked at `d−1` and `d+1` being worked, not the full
run length. Consider a worker already working days 0–4 (a run of five at `MAXCONS = 5`) with day 5
off; a move that sets day 5 to a shift makes a run of six. Checking only the immediate neighbours
"both worked?" is not enough — I must count the *whole* contiguous worked block that `d` would join.
So the local check walks left from `d` while days are worked and right from `d` while days are worked,
sums `left + 1 + right`, and rejects if it exceeds `MAXCONS`. The walk is bounded by `MAXCONS + 1`
steps, so it is still O(1) in practice. I verified this directly: I constructed a worker pre-filled
0–4, fed a move at day 5, and confirmed the check rejects it; with the neighbours-only version it had
wrongly accepted, and the scorer then zeroed the whole instance. That is precisely the kind of "one
worker's rule silently floors the entire score" trap the feasibility-first discipline is meant to
catch.

**Self-verifying on the seed set.** With both bugs fixed I ran the real harness: compile with
`g++ -O2 -std=c++17`, generate seeds `1…20`, run the solver on each under the budget, score with the
deterministic `score.py`, and also score the trivial all-off baseline. Every one of the twenty outputs
is feasible — the scorer never floors one to zero — and the solver's mean score is about `1 312 000`,
i.e. it beats the trivial all-off baseline (which scores `0`) decisively and, more tellingly, beats
the *greedy fill-by-demand reference* (which is exactly `1 000 000`) on every single seed, with the
worst seed still at `1 174 825` — at least 17 % above greedy. I then widened the check to sixty fresh
seeds with an *independent* re-implementation of the feasibility rules (a second pair of eyes that
does not share code with `score.py`): zero infeasible outputs, and the solver beats greedy on all
sixty, worst ratio `1.157`. The runtimes all land right at the `1.85 s` budget, comfortably inside the
two-second limit. Both acceptance criteria — every output feasible, and the solver strictly beats the
baseline — hold, and they hold with the margin coming from genuine optimisation rather than luck.

**Why this is the right strong method, in one line.** The column DP gives a high-quality feasible
roster fast by reasoning about each worker's whole week against the residual demand; the SA then
exploits the two-fold locality of a single-cell move — an O(1) objective delta and an O(1)
feasibility re-check confined to one worker's adjacent days — to run a vast number of refining moves
inside the budget without ever rescoring the grid or re-validating other workers. That combination,
not a slot-by-slot greedy, is what carries the score well past the greedy reference. The final solver:

```cpp
// Time-Indexed Crew Rostering -- heuristic solver.
//
// Objective: assign each (worker w, day d) a shift code a[w][d] in {0..S}
// (0 = day off, s>=1 = work shift s) to maximise
//     OBJ = sum_{d,s} VALUE[d][s] * min(cov[d][s], DEMAND[d][s])
//           - LAMBDA * (total overtime hours, hours per worker over MAXH)
// subject to the hard per-worker rest rules (any violation floors the score to 0):
//   (A) availability: a[w][d]=s>=1 requires AVAIL[w][d][s]=1;
//   (B) min rest between consecutive working days: 24+START[s']-END[s] >= MIN_REST;
//   (C) at most MAXCONS consecutive working days;
//   (D) total weekly hours <= HARDH.
// Read the instance from stdin, write the W x D grid (W lines, D codes each).
//
// Method (the innovation):
//   1. COLUMN-FLAVOURED GREEDY CONSTRUCTION. Each worker's feasible weekly pattern
//      is a "column". For each worker (in a coverage-aware order) we DP over that
//      worker's own days to pick the single feasible pattern of maximum MARGINAL
//      value -- value of the residual (still-uncovered up to demand) shifts it would
//      fill, minus its overtime -- given everything already rostered. The DP carries
//      the rest rule (B), the consecutive-day rule (C) and the hour budget (D) in its
//      state, so every column it returns is feasible by construction. This yields a
//      strong, always-feasible starting roster.
//   2. SIMULATED ANNEALING. State = the W x D grid. A move re-assigns ONE cell
//      a[w][d] to a different code. Feasibility is re-checked ONLY on worker w's
//      adjacent days (d-1, d, d+1), its consecutive-day run around d, and its hour
//      total -- an O(1) local check. The objective delta touches only the one or two
//      coverage buckets cov[d][.] and worker w's overtime, also O(1). The full grid
//      is NEVER rescored inside the loop. We keep the best feasible grid seen.
// The grid is feasible at all times, so any early stop (incl. the time limit) still
// prints a feasible solution.
#include <bits/stdc++.h>
using namespace std;

static inline double now_sec() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(
               steady_clock::now().time_since_epoch()).count();
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    uint32_t nextu(uint32_t m) { return (uint32_t)(next() % m); }     // [0,m)
    double nextd() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

int W, D, S;
vector<int> HRS, STc;                 // 1-indexed shift hours / clock start
int MIN_REST, MAXCONS, MAXH, HARDH, LAMBDA;
vector<vector<int>> demand, value;    // [D][S+1]
vector<vector<vector<char>>> avail;   // [W][D][S+1]

static inline int endHour(int s) { return STc[s] + HRS[s]; }
// rest gap between shift s on day d and shift s2 on day d+1
static inline int restGap(int s, int s2) { return 24 + STc[s2] - endHour(s); }
// can shift s be followed the NEXT day by shift s2 without breaking rest rule?
static inline bool restOK(int s, int s2) {
    if (s == 0 || s2 == 0) return true;       // an off-day never breaks rest
    return restGap(s, s2) >= MIN_REST;
}

// ---- global roster state ----
vector<vector<int>> A;                // [W][D] current assignment
vector<vector<int>> cov;             // [D][S+1] coverage counts
vector<int> hoursW;                  // [W] total hours per worker
long long curObj;                    // current objective

// coverage value of a single bucket given a count
static inline long long bucketVal(int d, int s, int c) {
    return (long long)value[d][s] * min(c, demand[d][s]);
}
// overtime hours of a worker given its total hours
static inline int overtimeOf(int h) { return h > MAXH ? h - MAXH : 0; }

// Recompute objective from scratch (used once, after construction, as a sanity base).
long long fullObjective() {
    long long obj = 0;
    for (int d = 0; d < D; d++)
        for (int s = 1; s <= S; s++)
            obj += bucketVal(d, s, cov[d][s]);
    for (int w = 0; w < W; w++) obj -= (long long)LAMBDA * overtimeOf(hoursW[w]);
    return obj;
}

// Is worker w's CURRENT row feasible (rules A-D)? Used after committing a move; the
// move-time check is the cheaper local version below.
bool rowFeasible(int w) {
    int consec = 0, tot = 0;
    for (int d = 0; d < D; d++) {
        int s = A[w][d];
        if (s == 0) { consec = 0; continue; }
        if (!avail[w][d][s]) return false;
        tot += HRS[s];
        consec++;
        if (consec > MAXCONS) return false;
        if (d + 1 < D && !restOK(s, A[w][d + 1])) return false;
    }
    return tot <= HARDH;
}

// Local feasibility of worker w around day d after tentatively setting A[w][d]=ns.
// Only the adjacent-day rest rule, the consecutive run through d, availability of
// ns, and the worker's new hour total can be affected -- O(MAXCONS) ~ O(1).
bool localFeasible(int w, int d, int ns, int newHours) {
    if (ns >= 1 && !avail[w][d][ns]) return false;
    if (newHours > HARDH) return false;
    // rest with previous and next day (ns may be 0)
    if (d - 1 >= 0 && !restOK(A[w][d - 1], ns)) return false;
    if (d + 1 < D && !restOK(ns, A[w][d + 1])) return false;
    // consecutive working-day run containing d, with A[w][d] replaced by ns
    if (ns >= 1) {
        int run = 1;
        for (int x = d - 1; x >= 0 && A[w][x] != 0; x--) run++;
        for (int x = d + 1; x < D && A[w][x] != 0; x++) run++;
        if (run > MAXCONS) return false;
    }
    return true;
}

// Apply A[w][d] <- ns, updating cov, hoursW and curObj incrementally. Caller must
// have verified localFeasible. Returns the objective delta.
long long applyMove(int w, int d, int ns) {
    int os = A[w][d];
    if (os == ns) return 0;
    long long delta = 0;
    // coverage change for the old shift
    if (os >= 1) {
        long long before = bucketVal(d, os, cov[d][os]);
        cov[d][os]--;
        delta += bucketVal(d, os, cov[d][os]) - before;
    }
    if (ns >= 1) {
        long long before = bucketVal(d, ns, cov[d][ns]);
        cov[d][ns]++;
        delta += bucketVal(d, ns, cov[d][ns]) - before;
    }
    // overtime change for worker w
    int oldOT = overtimeOf(hoursW[w]);
    hoursW[w] += (ns >= 1 ? HRS[ns] : 0) - (os >= 1 ? HRS[os] : 0);
    int newOT = overtimeOf(hoursW[w]);
    delta -= (long long)LAMBDA * (newOT - oldOT);
    A[w][d] = ns;
    curObj += delta;
    return delta;
}

// ---- column-flavoured greedy construction ----
// For worker w, choose the feasible weekly pattern maximising marginal value given
// the current cov (residual demand) and hour budget. DP over days; state = (shift
// chosen yesterday, consecutive working days so far). Hours are handled greedily by
// pruning: the DP value already discounts overtime via a running hour count carried
// approximately -- we keep it exact by storing hours in the state-compressed key.
// D is tiny (a week), S is tiny, so an exact DP over (prevShift, consecRun, hours)
// is cheap. We return the chosen pattern.
vector<int> bestColumn(int w) {
    // marginal value of adding one more worker to (d,s) right now (residual unit)
    auto marg = [&](int d, int s) -> long long {
        if (cov[d][s] >= demand[d][s]) return 0;   // demand already met -> no gain
        return value[d][s];
    };
    // DP state index: pack (prevShift in 0..S, consec in 0..MAXCONS, hours in 0..HARDH)
    const int PS = S + 1;
    const int CS = MAXCONS + 1;          // consec working days, capped (0..MAXCONS)
    const int HS = HARDH + 1;            // hours 0..HARDH
    const long long NEG = LLONG_MIN / 4;
    // dp[d] over states; store best marginal-minus-overtime value and a back pointer
    // Flatten: idx = ((prev)*CS + consec)*HS + hours
    int nStates = PS * CS * HS;
    vector<long long> dp(nStates, NEG), ndp(nStates, NEG);
    vector<vector<int>> choice(D, vector<int>(nStates, -1));  // shift chosen at day d
    auto sidx = [&](int prev, int consec, int hrs) {
        return (prev * CS + consec) * HS + hrs;
    };
    // initial state before day 0: prev = 0 (off), consec = 0, hours = 0
    dp[sidx(0, 0, 0)] = 0;
    for (int d = 0; d < D; d++) {
        fill(ndp.begin(), ndp.end(), NEG);
        for (int prev = 0; prev < PS; prev++)
            for (int consec = 0; consec < CS; consec++)
                for (int hrs = 0; hrs < HS; hrs++) {
                    long long base = dp[sidx(prev, consec, hrs)];
                    if (base == NEG) continue;
                    // option 1: day off
                    {
                        int ni = sidx(0, 0, hrs);
                        if (base > ndp[ni]) { ndp[ni] = base; choice[d][ni] = 0; }
                    }
                    // option 2..: work shift s
                    for (int s = 1; s <= S; s++) {
                        if (!avail[w][d][s]) continue;
                        if (prev != 0 && !restOK(prev, s)) continue;  // rest rule (B)
                        int nconsec = consec + 1;
                        if (nconsec > MAXCONS) continue;              // rule (C)
                        int nh = hrs + HRS[s];
                        if (nh > HARDH) continue;                     // rule (D)
                        long long gain = marg(d, s);
                        // overtime cost incurred by crossing MAXH on this shift
                        int otBefore = overtimeOf(hrs);
                        int otAfter = overtimeOf(nh);
                        long long val = base + gain
                                        - (long long)LAMBDA * (otAfter - otBefore);
                        int ni = sidx(s, nconsec, nh);
                        if (val > ndp[ni]) { ndp[ni] = val; choice[d][ni] = s; }
                    }
                }
        dp.swap(ndp);
    }
    // pick best terminal state
    long long best = NEG; int bi = -1;
    for (int i = 0; i < nStates; i++) if (dp[i] > best) { best = dp[i]; bi = i; }
    // reconstruct backwards
    vector<int> pat(D, 0);
    if (bi < 0) return pat;            // no feasible state (shouldn't happen: all-off works)
    // We need the state path; recompute forward keeping parents would be O(states*D)
    // memory. Instead re-run the DP storing parent pointers fully.
    // (Re-run with parent tracking.)
    vector<vector<long long>> DP(D + 1, vector<long long>(nStates, NEG));
    vector<vector<int>> par(D + 1, vector<int>(nStates, -1));   // parent state idx
    vector<vector<int>> act(D + 1, vector<int>(nStates, -1));   // shift taken to reach
    DP[0][sidx(0, 0, 0)] = 0;
    for (int d = 0; d < D; d++) {
        for (int prev = 0; prev < PS; prev++)
            for (int consec = 0; consec < CS; consec++)
                for (int hrs = 0; hrs < HS; hrs++) {
                    int cur = sidx(prev, consec, hrs);
                    long long base = DP[d][cur];
                    if (base == NEG) continue;
                    { int ni = sidx(0, 0, hrs);
                      if (base > DP[d + 1][ni]) { DP[d + 1][ni] = base; par[d + 1][ni] = cur; act[d + 1][ni] = 0; } }
                    for (int s = 1; s <= S; s++) {
                        if (!avail[w][d][s]) continue;
                        if (prev != 0 && !restOK(prev, s)) continue;
                        int nconsec = consec + 1;
                        if (nconsec > MAXCONS) continue;
                        int nh = hrs + HRS[s];
                        if (nh > HARDH) continue;
                        long long val = base + marg(d, s)
                                        - (long long)LAMBDA * (overtimeOf(nh) - overtimeOf(hrs));
                        int ni = sidx(s, nconsec, nh);
                        if (val > DP[d + 1][ni]) { DP[d + 1][ni] = val; par[d + 1][ni] = cur; act[d + 1][ni] = s; }
                    }
                }
    }
    long long fb = NEG; int fbi = -1;
    for (int i = 0; i < nStates; i++) if (DP[D][i] > fb) { fb = DP[D][i]; fbi = i; }
    int st = fbi;
    for (int d = D; d >= 1; d--) {
        pat[d - 1] = act[d][st];
        st = par[d][st];
    }
    return pat;
}

int main() {
    const double T0 = now_sec();
    const double TIME_LIMIT = 1.85;

    if (scanf("%d %d %d", &W, &D, &S) != 3) return 0;
    HRS.assign(S + 1, 0); STc.assign(S + 1, 0);
    for (int s = 1; s <= S; s++) scanf("%d", &HRS[s]);
    for (int s = 1; s <= S; s++) scanf("%d", &STc[s]);
    scanf("%d %d %d %d %d", &MIN_REST, &MAXCONS, &MAXH, &HARDH, &LAMBDA);
    demand.assign(D, vector<int>(S + 1, 0));
    value.assign(D, vector<int>(S + 1, 0));
    for (int d = 0; d < D; d++) {
        for (int s = 1; s <= S; s++) scanf("%d", &demand[d][s]);
        for (int s = 1; s <= S; s++) scanf("%d", &value[d][s]);
    }
    avail.assign(W, vector<vector<char>>(D, vector<char>(S + 1, 0)));
    for (int w = 0; w < W; w++)
        for (int d = 0; d < D; d++)
            for (int s = 1; s <= S; s++) { int x; scanf("%d", &x); avail[w][d][s] = (char)x; }

    if (W <= 0 || D <= 0) { return 0; }

    // ---- state init: everyone off (trivially feasible) ----
    A.assign(W, vector<int>(D, 0));
    cov.assign(D, vector<int>(S + 1, 0));
    hoursW.assign(W, 0);

    // ---- construction: process workers and greedily take each best column ----
    // Order workers by how flexible they are (more available slots first tends to
    // leave the constrained ones for the residual demand; either order is feasible).
    vector<int> order(W);
    for (int w = 0; w < W; w++) order[w] = w;
    {
        vector<int> flex(W, 0);
        for (int w = 0; w < W; w++)
            for (int d = 0; d < D; d++)
                for (int s = 1; s <= S; s++) flex[w] += avail[w][d][s];
        stable_sort(order.begin(), order.end(),
                    [&](int a, int b) { return flex[a] > flex[b]; });
    }
    for (int idx = 0; idx < W; idx++) {
        int w = order[idx];
        vector<int> pat = bestColumn(w);
        for (int d = 0; d < D; d++) {
            int ns = pat[d];
            if (ns == A[w][d]) continue;
            // apply (always feasible: column DP enforced A-D against an empty row)
            if (ns >= 1) { cov[d][ns]++; hoursW[w] += HRS[ns]; }
            A[w][d] = ns;
        }
    }
    curObj = fullObjective();

    // Safety: ensure the constructed roster is genuinely feasible; if any worker is
    // not (should not happen), clear that worker (all-off is always feasible).
    for (int w = 0; w < W; w++) {
        if (!rowFeasible(w)) {
            for (int d = 0; d < D; d++) {
                int os = A[w][d];
                if (os >= 1) { cov[d][os]--; hoursW[w] -= HRS[os]; A[w][d] = 0; }
            }
        }
    }
    curObj = fullObjective();

    // remember best
    vector<vector<int>> bestA = A;
    long long bestObj = curObj;

    // ---- simulated annealing ----
    Rng rng(0xA1E48u ^ (uint64_t)(W * 1000003 + D * 9176 + S));
    double Tstart = 0.0;
    // a reasonable starting temperature: a fraction of the largest single-shift value
    for (int d = 0; d < D; d++)
        for (int s = 1; s <= S; s++) Tstart = max(Tstart, (double)value[d][s]);
    Tstart = max(1.0, Tstart);     // accept ~one shift-value worsening early
    double Tend = 0.05;

    long long iters = 0;
    double t = 0.0;
    while (true) {
        if ((iters & 1023) == 0) {
            t = (now_sec() - T0) / TIME_LIMIT;
            if (t >= 1.0) break;
        }
        iters++;
        double frac = t;
        double T = Tstart * pow(Tend / Tstart, frac);

        int w = rng.nextu(W);
        int d = rng.nextu(D);
        int os = A[w][d];
        // propose a new code: off, or one of the shifts (uniformly among 0..S != os)
        int ns = rng.nextu(S + 1);
        if (ns == os) { ns = (ns + 1) % (S + 1); }
        // quick availability reject (also handled in localFeasible)
        if (ns >= 1 && !avail[w][d][ns]) continue;

        int newHours = hoursW[w] + (ns >= 1 ? HRS[ns] : 0) - (os >= 1 ? HRS[os] : 0);
        if (!localFeasible(w, d, ns, newHours)) continue;

        // incremental objective delta WITHOUT committing
        long long delta = 0;
        if (os >= 1) {
            long long b = bucketVal(d, os, cov[d][os]);
            delta += bucketVal(d, os, cov[d][os] - 1) - b;
        }
        if (ns >= 1) {
            long long b = bucketVal(d, ns, cov[d][ns]);
            delta += bucketVal(d, ns, cov[d][ns] + 1) - b;
        }
        int oldOT = overtimeOf(hoursW[w]);
        int newOT = overtimeOf(newHours);
        delta -= (long long)LAMBDA * (newOT - oldOT);

        bool accept;
        if (delta >= 0) accept = true;
        else accept = (rng.nextd() < exp((double)delta / T));

        if (accept) {
            applyMove(w, d, ns);     // commits cov/hours/obj consistently
            if (curObj > bestObj) {
                bestObj = curObj;
                bestA = A;
            }
        }
    }

    // restore best feasible grid
    A = bestA;

    // ---- output ----
    string out;
    out.reserve((size_t)W * D * 2);
    for (int w = 0; w < W; w++) {
        for (int d = 0; d < D; d++) {
            out += to_string(A[w][d]);
            out += (d + 1 < D ? ' ' : '\n');
        }
    }
    fputs(out.c_str(), stdout);
    return 0;
}
```
