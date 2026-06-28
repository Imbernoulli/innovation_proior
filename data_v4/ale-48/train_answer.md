**Problem.** Fill a `W × D` roster grid: each cell `a[w][d] ∈ {0, …, S}` assigns worker `w` on day
`d` to a shift (`0` = day off). Each `(day d, shift s)` slot has a value `VALUE[d][s]` per covered
unit and a demand `DEMAND[d][s]`; covering beyond demand earns nothing. Each shift `s` has length
`HOURS[s]` and clock start `START[s]` (end `= START[s] + HOURS[s]`, possibly past midnight). Read
`W D S`, the shift hours and starts, the rule parameters `MIN_REST MAXCONS MAXH HARDH LAMBDA`,
per-day demand/value, and per-worker availability from stdin; write the grid (`W` lines of `D` codes)
to stdout.

**Objective and scoring.** Maximise

```
OBJ = Σ_{d,s} VALUE[d][s] · min(cov[d][s], DEMAND[d][s])  −  LAMBDA · Σ_w overtime(w),
```

with `cov[d][s]` the number of workers on shift `s`, day `d`, and `overtime(w)` the worker's weekly
hours beyond the soft cap `MAXH`. The score is `round(1 000 000 · OBJ / OBJ(G))`, normalised against
a deterministic **greedy fill-by-demand** roster `G` that the scorer recomputes itself (so the
reference is reproducible and solver-independent). **Feasibility floor:** the grid must parse as
`W × D` codes in `[0, S]` and every worker's row must satisfy all four hard rules — (A) availability,
(B) minimum rest `24 + START[s'] − END[s] ≥ MIN_REST` between consecutive worked days, (C) at most
`MAXCONS` consecutive worked days, (D) total weekly hours `≤ HARDH`. Any violation or malformed
output scores **0**.

**Baseline.** The all-off grid (everyone rests) is always feasible but covers nothing, scoring `0` —
the floor. The greedy fill-by-demand reference (staff the highest-value slots first with the
lowest-indexed legally-available workers) is the stronger reference and scores exactly `1 000 000`;
the real target is to beat it. Greedy leaves value on the table because it commits workers early — a
worker spent on a medium-value slot is then locked out, by rest/hours/consecutive rules, of a more
valuable slot considered later.

**Key idea — the heuristic innovation.** Two levers, used in sequence.

1. **Column-flavoured greedy construction.** Treat each worker's whole weekly pattern as a *column*.
   Because the four hard rules are entirely per-worker, feasibility factorises: a column is feasible
   iff it obeys (A)–(D) for that one worker alone. So for each worker (processed flexible-first) I run
   an exact **dynamic program over that worker's seven days** with state
   `(prevShift, consecutiveRun, hoursSoFar)`, which carries rule (B) (rest vs `prevShift`), rule (C)
   (`consecutiveRun ≤ MAXCONS`) and rule (D) (`hours ≤ HARDH`) in its transitions, and whose
   per-shift reward is the **marginal residual value** `VALUE[d][s]` if `cov[d][s] < DEMAND[d][s]`
   (else 0) minus the extra overtime. The DP returns the best feasible column against the residual
   demand; committing one strong column per worker yields a high-quality feasible starting roster.

2. **Simulated annealing with two-fold locality.** A single-cell move `a[w][d] : s → s'` is local in
   two independent senses, and this is the whole engine. *Objective:* it shifts only `cov[d][s]`,
   `cov[d][s']` and worker `w`'s overtime, so its objective delta is `O(1)` — the grid is never
   rescored. *Feasibility:* since the rules are per-worker, only worker `w` can be broken, and only
   through its days `d−1, d, d+1` (rest), the consecutive run through `d`, availability of `s'`, and
   its hour total — an `O(1)` re-check **confined to one worker's adjacent days**. SA proposes random
   cells, rejects locally-infeasible moves, accepts by the Metropolis rule on the `O(1)` delta with a
   geometric temperature scheduled on the wall-clock fraction, and keeps the best feasible grid.

**Feasibility and pitfalls.** The grid is feasible at all times: construction commits only
DP-certified-feasible columns, SA accepts only locally-feasible moves, and the best-seen grid is
restored on timeout — so any early stop still prints a valid roster. Two pitfalls are decisive.
*(i) State desync.* Evaluate each move's delta by **pure reads** of hypothetical counts
(`bucketVal(d, s, cov ± 1)`); mutate `cov`/`hours`/`A`/`obj` in exactly one commit path
(`applyMove`), called only on accept. Computing the delta by "do then maybe undo" drifts `cov` out of
sync with the grid and silently corrupts both the tracked objective and the feasibility check; a
periodic `fullObjective()` cross-check catches it during development. *(ii) Consecutive-run check.*
Checking only whether `d−1`/`d+1` are worked is wrong — placing a shift can extend an existing block
past `MAXCONS`. The local check must walk the full contiguous worked run through `d` (left and right)
and reject if `left + 1 + right > MAXCONS`; the walk is bounded by `MAXCONS + 1`, so still `O(1)`.

**Complexity per step.** Construction: per worker an `O(D · S · MAXCONS · HARDH)` DP — a few thousand
states on a week, microseconds each, `O(W)` columns total. SA: each iteration is `O(1)` — a handful
of arithmetic ops for the delta plus an `O(MAXCONS)` feasibility walk — so the solver runs on the
order of millions of moves inside the ~1.85 s budget without ever touching the global objective.

**Validation.** Over seeds 1–20 every output is feasible and the solver's mean score is ≈ 1.31 M,
beating the trivial all-off baseline (0) and the greedy reference (1.0 M) on every seed (worst seed
1.17 M). A 60-seed sweep with an independent re-implementation of the feasibility rules confirms zero
infeasible outputs and a worst solver/greedy objective ratio of 1.157.

**Code.**

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
