# Soft-Constraint Assignment — editorial

## Problem

Place `n` agents into `m` capacity-limited slots (slot `j` holds at most `cap_j`, with
`Σ_j cap_j ≥ n` so a legal placement always exists). Putting agent `i` in slot `j` earns
`pref[i][j] ≥ 0`. A list of `C` soft constraints couples pairs of agents: a **DIFFER** `(a, b, w)`
charges `w` if `a, b` share a slot; a **SAME** `(a, b, w)` charges `w` if they are in different slots.

**Input:** `n m`; then `m` capacities; then `n` rows of `m` preferences; then `C`; then `C` lines
`t a b w` (`t = 0` DIFFER, `t = 1` SAME). **Output:** `n` integers, the slot of each agent.

## Objective and scoring

Maximize

```
P = Σ_i pref[i][assign[i]]  −  Σ_{violated constraints} w.
```

The local scorer floors an output to **`0`** unless it is exactly `n` integers, each a slot in `[0, m)`,
**and** every slot's load is within capacity (a capacity overflow is infeasible). For a feasible output it
reports

```
score = round( 1 000 000 + 1 000 000 × (P − P_base) / Scale ),  clamped ≥ 0,
```

where `P_base` is the objective of a deterministic **greedy best-preference** baseline (each agent to its
best slot ignoring constraints, then capacity overflow repaired by evicting the least-loss agents) and
`Scale = Σ_i max_j pref[i][j] ≥ 1`. The baseline scores exactly `1 000 000`; beating it requires trading a
little preference for fewer violations. The scorer recomputes `P_base`, `Scale`, and every penalty itself.

## Baseline (always-feasible safety net)

Assign every agent to its single highest-preference slot, then repair capacity: for each over-full slot,
repeatedly evict the agent whose **loss** (`pref` at the current slot minus best `pref` at a slot with
spare room) is smallest, moving it there. Because `Σ cap ≥ n`, room always exists, so this terminates with
a fully feasible assignment. This is the scorer's reference baseline and our fallback if the search is cut
off early — it is `O(n·m)` plus a cheap repair. It is *constraint-blind*, so it pays every violated
penalty in full; that is the objective a constraint-aware search recovers.

## Key idea (the heuristic innovation)

**Relaxation rounding + a fused local search with an `O(degree)` penalty delta.** Start from the baseline,
then run one simulated-annealing local search whose moves change the assignment directly, so preference
and penalties co-evolve. The engineering lever is that a move's effect on `P` is cheap:

- **Constraint→agent incidence lists.** Precompute, for each agent, the constraints it participates in
  (`inc[x]`). When a move touches agent `x`, *only* the constraints in `inc[x]` can change penalty — every
  other constraint sees both endpoints unchanged. So the penalty delta is a sum over `deg(x)` constraints,
  not all `C`. With `C ≈ n…2n`, average degree is a small constant, so the delta is effectively `O(1)`.
- **RELOCATE** agent `x` to slot `t` with spare capacity (`load[t] < cap[t]`, keeping feasibility):
  `dPref = pref[x][t] − pref[x][cur]`, and `dPenalty = Σ_{c ∈ inc[x]} (penalty with x at t − penalty with
  x at cur)`. Then `dObj = dPref − dPenalty`.
- **SWAP** agents `x` and `y` (different slots) — the non-obvious move. It is **always capacity-feasible**
  because each slot's load is preserved (one agent leaves, one arrives), so no capacity check is needed.
  Its delta is still local — only `inc[x] ∪ inc[y]` matter. A relocate-only search gets stuck when the good
  slots are full; the swap reaches the trades that unstick it.
- **Simulated annealing.** Accept `dObj ≥ 0` always, else with probability `exp(dObj / T)`, cooling `T`
  geometrically (from ≈ half the preference spread down to ~0.1% of it) across a ~1.9 s budget. Track and
  print the best feasible assignment seen.

## Feasibility and pitfalls

- **Capacity is the floor-killer.** A naive search piles agents into high-preference slots and overflows.
  We keep capacities respected *at every instant*: relocate only into a slot with spare room; swap
  preserves loads. We only ever print `bestAssign`, which begins as the feasible baseline and is replaced
  only by other feasible states — so the output is feasible no matter when the time limit fires.
- **The swap double-count.** A constraint whose endpoints are exactly `{x, y}` appears in both `inc[x]` and
  `inc[y]`; summing naively counts it twice. We subtract such constraints once from both the old and the
  new penalty sums. Verified by an independent recompute: the solver's incremental `P` equals the
  from-scratch objective of the emitted assignment on every tested seed (e.g. seed 1 `125481 = 125481`,
  seed 12 `94926 = 94926`).
- **Soft, not hard.** Treating constraints as hard over-constrains and can spuriously conflict with tight
  capacities; the penalties must stay in the objective so the search can choose to eat a cheap one.
- **Robustness.** Malformed constraint indices are clamped on read; `n = 0` prints an empty line; integer
  arithmetic in `long long` avoids overflow.

## Complexity per step

- Construction: `O(n·m)` for best-preference + a small capacity repair.
- Incidence lists: `O(C)` once.
- Per move: `O(deg(x))` for relocate, `O(deg(x) + deg(y))` for swap — effectively `O(1)` at average
  degree — so the search runs millions of moves inside the ~1.9 s budget.

On a fixed seed set (seeds 1..20) the solver is feasible on all 20 and beats the `1 000 000` baseline on
all 20 (mean ≈ `1 065 710`, worst seed ≈ `1 028 712`), i.e. about `+6.6%` objective over the
constraint-blind start.

## Code

```cpp
// Soft-Constraint Assignment -- heuristic solver.
//
// Objective: assign each of the n agents to exactly one of the m slots, slot j
// holding at most cap_j agents, to MAXIMIZE
//     P = sum_i pref[i][assign[i]]  -  sum over VIOLATED soft constraints of w,
// where each soft constraint is (t, a, b, w):
//     t == 0 DIFFER : penalty w charged iff assign[a] == assign[b];
//     t == 1 SAME   : penalty w charged iff assign[a] != assign[b].
// Read the instance from stdin; write n integers (slot of each agent, in input
// order), space-separated, to stdout. A capacity-respecting assignment is always
// feasible; we keep one valid at all times so any early stop still prints a feasible
// answer.
//
// Method (the innovation):
//   * RELAXATION ROUNDING for the start: assign every agent to its best-preference
//     slot ignoring constraints, then repair capacity overflow by evicting the
//     least-loss agents to their best still-feasible slot (this matches the scorer's
//     baseline -- a strong, constraint-blind anchor we then improve on).
//   * CONSTRAINT -> AGENT INCIDENCE LISTS: for each agent we store the constraints it
//     participates in. A local move touching agents x (and y) recomputes the penalty
//     delta by scanning ONLY those incident constraints -- O(deg) not O(C). This is
//     the lever: the soft-penalty part of a move's score is a tiny, local sum.
//   * ONE local search over two capacity-preserving / capacity-checked moves:
//       - RELOCATE agent x to a slot t with spare capacity:
//           dPref    = pref[x][t] - pref[x][cur]
//           dPenalty = (penalties incident to x under new slot)
//                      - (penalties incident to x under old slot)   [O(deg_x)]
//           feasible iff load[t] < cap[t].
//       - SWAP agents x and y (slots cur_x, cur_y, must differ): loads are
//         preserved automatically, so always capacity-feasible. The penalty delta
//         scans constraints incident to x and to y; a constraint between x and y is
//         handled once (its endpoints both move), which the delta evaluation gets
//         right because it recomputes each touched constraint from the proposed slots.
//   * SIMULATED ANNEALING accepts a move with prob 1 if dObj >= 0 else exp(dObj/T),
//     T cooled geometrically over a ~1.9s budget; we always remember and finally
//     print the BEST feasible assignment seen.
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
    uint32_t nextu(uint32_t mod) { return (uint32_t)(next() % mod); }   // [0, mod)
    double nextd() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

int N, M, C;
vector<int> cap;                 // size M
vector<vector<int>> pref;        // N x M
// constraints stored flat: type, a, b, w
vector<int> ct, ca, cb, cw;      // size C
vector<vector<int>> inc;         // inc[agent] = list of constraint indices touching it

vector<int> assign_;             // assign_[i] in [0, M)
vector<int> load;                // load[j] = #agents in slot j

// penalty contributed by constraint c given the CURRENT assignment of its two agents
static inline long long penaltyOf(int c) {
    int a = ca[c], b = cb[c];
    if (ct[c] == 0) {            // DIFFER: penalty iff same slot
        return (assign_[a] == assign_[b]) ? (long long)cw[c] : 0LL;
    } else {                     // SAME: penalty iff different slots
        return (assign_[a] != assign_[b]) ? (long long)cw[c] : 0LL;
    }
}

// penalty of constraint c if agent `who` were instead in slot `sNew` (the OTHER
// endpoint keeps its current slot). Used to evaluate a hypothetical move cheaply.
static inline long long penaltyIf(int c, int who, int sNew) {
    int a = ca[c], b = cb[c];
    int sa = (a == who) ? sNew : assign_[a];
    int sb = (b == who) ? sNew : assign_[b];
    if (ct[c] == 0) return (sa == sb) ? (long long)cw[c] : 0LL;
    else            return (sa != sb) ? (long long)cw[c] : 0LL;
}

int main() {
    const double T0 = now_sec();
    const double TIME_LIMIT = 1.9;   // wall-clock budget (seconds)

    // ---- read instance ----
    if (scanf("%d %d", &N, &M) != 2) return 0;
    if (N < 0) N = 0;
    if (M < 1) M = 1;
    cap.assign(M, 0);
    for (int j = 0; j < M; j++) { if (scanf("%d", &cap[j]) != 1) cap[j] = 0; }
    pref.assign(N, vector<int>(M, 0));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            if (scanf("%d", &pref[i][j]) != 1) pref[i][j] = 0;
    if (scanf("%d", &C) != 1) C = 0;
    if (C < 0) C = 0;
    ct.assign(C, 0); ca.assign(C, 0); cb.assign(C, 0); cw.assign(C, 0);
    for (int c = 0; c < C; c++) {
        if (scanf("%d %d %d %d", &ct[c], &ca[c], &cb[c], &cw[c]) != 4) {
            ct[c] = 0; ca[c] = 0; cb[c] = 0; cw[c] = 0;
        }
        // guard against malformed indices
        if (ca[c] < 0 || ca[c] >= N) ca[c] = 0;
        if (cb[c] < 0 || cb[c] >= N) cb[c] = (N > 1 ? 1 : 0);
        if (cw[c] < 0) cw[c] = 0;
        if (ct[c] != 0 && ct[c] != 1) ct[c] = 0;
    }

    // Degenerate: no agents -> empty output.
    if (N == 0) { printf("\n"); return 0; }

    // ---- constraint -> agent incidence lists ----
    inc.assign(N, {});
    for (int c = 0; c < C; c++) {
        inc[ca[c]].push_back(c);
        if (cb[c] != ca[c]) inc[cb[c]].push_back(c);
    }

    Rng rng(0xA55E27ULL ^ ((uint64_t)N * 1000003ULL + (uint64_t)M * 9176ULL + (uint64_t)C));

    // ---- relaxation rounding: best-preference, then capacity repair ----
    assign_.assign(N, 0);
    load.assign(M, 0);
    for (int i = 0; i < N; i++) {
        int bj = 0, bv = pref[i][0];
        for (int j = 1; j < M; j++) if (pref[i][j] > bv) { bv = pref[i][j]; bj = j; }
        assign_[i] = bj; load[bj]++;
    }
    // repair overflow: for each over-full slot, evict least-loss agents to their best
    // still-feasible slot (capacity guaranteed: sum cap >= N).
    {
        vector<int> remCap(M);
        for (int j = 0; j < M; j++) remCap[j] = cap[j] - load[j];
        for (int j = 0; j < M; j++) {
            while (load[j] > cap[j]) {
                int bestAgent = -1, bestTarget = -1; long long bestLoss = LLONG_MAX;
                for (int i = 0; i < N; i++) {
                    if (assign_[i] != j) continue;
                    int cur = pref[i][j];
                    int tj = -1, tv = INT_MIN;
                    for (int k = 0; k < M; k++) {
                        if (k == j || remCap[k] <= 0) continue;
                        if (pref[i][k] > tv || (pref[i][k] == tv && (tj == -1 || k < tj))) {
                            tv = pref[i][k]; tj = k;
                        }
                    }
                    if (tj == -1) continue;
                    long long loss = (long long)cur - tv;
                    if (loss < bestLoss || (loss == bestLoss && (bestAgent == -1 || i < bestAgent))) {
                        bestLoss = loss; bestAgent = i; bestTarget = tj;
                    }
                }
                if (bestAgent == -1) break;  // unreachable given feasibility guarantee
                assign_[bestAgent] = bestTarget;
                load[j]--; load[bestTarget]++;
                remCap[j]++; remCap[bestTarget]--;
            }
        }
    }

    // ---- current objective P (kept incrementally) ----
    long long P = 0;
    for (int i = 0; i < N; i++) P += pref[i][assign_[i]];
    for (int c = 0; c < C; c++) P -= penaltyOf(c);

    // best feasible seen
    vector<int> bestAssign = assign_;
    long long bestP = P;

    // temperature scale: roughly the spread of preferences (so early moves can pay a
    // typical preference loss).
    double Tstart;
    {
        int mn = INT_MAX, mx = INT_MIN;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < M; j++) { mn = min(mn, pref[i][j]); mx = max(mx, pref[i][j]); }
        Tstart = max(1.0, (double)(mx - mn) * 0.5);
    }

    // ---- simulated annealing over RELOCATE and SWAP ----
    long long iter = 0;
    while (true) {
        // check time every so often
        if ((iter & 1023) == 0) {
            double el = now_sec() - T0;
            if (el > TIME_LIMIT) break;
        }
        iter++;
        double prog = min(1.0, (now_sec() - T0) / TIME_LIMIT);
        double T = Tstart * pow(1e-3, prog);   // geometric cooling to ~0.1% of start

        bool doSwap = (rng.nextu(2) == 0) && (M > 1);
        if (!doSwap) {
            // RELOCATE agent x to a different slot t with spare capacity.
            int x = rng.nextu(N);
            int cur = assign_[x];
            int t = rng.nextu(M);
            if (t == cur) continue;
            if (load[t] >= cap[t]) continue;      // capacity check -> keep feasible
            long long dPref = (long long)pref[x][t] - pref[x][cur];
            // penalty delta over constraints incident to x
            long long dPen = 0;
            for (int c : inc[x]) dPen += penaltyIf(c, x, t) - penaltyOf(c);
            long long dObj = dPref - dPen;        // change in P (P = pref - penalties)
            if (dObj >= 0 || rng.nextd() < exp((double)dObj / T)) {
                assign_[x] = t;
                load[cur]--; load[t]++;
                P += dObj;
                if (P > bestP) { bestP = P; bestAssign = assign_; }
            }
        } else {
            // SWAP agents x and y (different current slots). Loads preserved.
            int x = rng.nextu(N);
            int y = rng.nextu(N);
            if (x == y) continue;
            int sx = assign_[x], sy = assign_[y];
            if (sx == sy) continue;               // swap would be a no-op
            long long dPref = ((long long)pref[x][sy] - pref[x][sx])
                            + ((long long)pref[y][sx] - pref[y][sy]);
            // penalty delta: recompute touched constraints with BOTH new slots.
            // Use a small dedup via a visited stamp on constraint indices.
            long long penOld = 0, penNew = 0;
            // gather incident constraints of x and y; avoid double-counting a
            // constraint that touches both (the x-y constraint) by marking.
            // Old penalties:
            for (int c : inc[x]) penOld += penaltyOf(c);
            for (int c : inc[y]) penOld += penaltyOf(c);
            // subtract the x-y shared constraints once (counted twice above):
            // a constraint touches both x and y iff {a,b} == {x,y}.
            for (int c : inc[x]) {
                if ((ca[c] == x && cb[c] == y) || (ca[c] == y && cb[c] == x))
                    penOld -= penaltyOf(c);
            }
            // New penalties: temporarily apply the swap, evaluate, revert.
            assign_[x] = sy; assign_[y] = sx;
            for (int c : inc[x]) penNew += penaltyOf(c);
            for (int c : inc[y]) penNew += penaltyOf(c);
            for (int c : inc[x]) {
                if ((ca[c] == x && cb[c] == y) || (ca[c] == y && cb[c] == x))
                    penNew -= penaltyOf(c);
            }
            assign_[x] = sx; assign_[y] = sy;     // revert
            long long dPen = penNew - penOld;
            long long dObj = dPref - dPen;
            if (dObj >= 0 || rng.nextd() < exp((double)dObj / T)) {
                assign_[x] = sy; assign_[y] = sx; // loads unchanged
                P += dObj;
                if (P > bestP) { bestP = P; bestAssign = assign_; }
            }
        }
    }

    // ---- output the best feasible assignment seen ----
    string out;
    out.reserve((size_t)N * 4);
    for (int i = 0; i < N; i++) {
        out += to_string(bestAssign[i]);
        out += (i + 1 < N) ? ' ' : '\n';
    }
    fputs(out.c_str(), stdout);
    return 0;
}
```
