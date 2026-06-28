# Reasoning: Parameter Placement for a Simulated Controller

## What is actually being asked

I have a deterministic scalar plant — a point mass on a line with position `x` and
velocity `v` — and a reference trajectory `ref[0..T-1]` it should follow. I get to set
the gains of a controller, and the score I am graded on is how small I can make the
accumulated squared tracking error `COST = sum_t (ref[t] - x_after(t))^2`. The twist
that makes it a placement problem rather than a one-shot PID-tuning exercise: the
horizon is cut into `S` equal **segments**, and each segment carries its *own* gain
triple `(Kp, Kd, Kv)`. So my decision vector is `S*K = S*3` real numbers, and I report
each one as an integer code in `[0, Q]` that the scorer maps onto that gain's box. The
plant evolves as

```
e   = ref[t] - x
f   = Kp*e + Kd*(e - e_prev) + Kv*v
v  += f - DRAG*v + d[t]
x  += v
```

with `DRAG = 0.02` and an exogenous disturbance `d[t]`. Everything is deterministic;
the only freedom is the `S*K` codes.

Two things jump out immediately. First, there is no closed form: the gains in segment
`s` interact with the gains in every later segment through the carried-over state
`(x, v, e_prev)` at the boundary, and the squared-error objective over a nonlinear
feedback loop is non-convex. This is a black-box continuous optimization problem where
the oracle is a forward simulation. Second, the objective is cheap to evaluate once —
one pass is `O(T)` and `T <= 12*70 = 840` — so I can afford a *lot* of evaluations if
each one is cheap. That observation is the whole game, and I will come back to it.

I also need to pin down feasibility, because in ALE-Bench an infeasible output floors
the score to zero. A solution is feasible iff it parses as exactly `S*K` integers and
every code is in `[0, Q]`; additionally the simulation must not diverge (NaN/inf or a
magnitude past `1e15`). The codes-in-a-box construction means I can *guarantee*
feasibility on the parsing/bounds side for free if I simply never let a code leave
`[0, Q]`. Divergence is the only other way to fail, and I will watch for it.

## Getting a feasible baseline first

Before any cleverness I want a solution that always parses and always scores. The
simplest is "all-zero codes": every code is `0`, which is trivially in `[0, Q]`. With
the standard boxes that maps to `(Kp, Kd, Kv) = (0, 0, -1.2)` — pure velocity damping,
no tracking. It is feasible and it is what I will measure my solver against. (The score
itself normalizes against an even more passive reference, the literal zero-gain / open-
loop controller with `f == 0`, but as a *trivial-baseline output* the all-zero codes
are the honest thing to beat.)

So step zero of the solver is: read the instance, set `code` to all zeros, simulate
once to get `curCost`, and be ready to print those codes if everything else somehow
fails. That alone is a valid (if weak) submission. Good — I always have a feasible
answer in hand.

## Why the obvious search is too slow

The obvious approach is coordinate descent or simulated annealing directly over the
`S*K` codes: pick a coordinate, perturb it, run the simulation, accept if the cost
improved (or by Metropolis). The catch is the *cost of an evaluation*. If every
perturbation re-runs the full `O(T)` simulation, then with `T` up to ~840 and a 1.7-
second budget I get on the order of a few hundred thousand evaluations — which sounds
like a lot until I remember the decision vector has up to `12*3 = 36` coordinates and
the cost surface is bumpy, so annealing wants *millions* of single-coordinate trials
to converge well. A full-resim search is shallow: it spends its whole budget paying for
horizon steps that the perturbation never even touched.

Here is the structural fact that rescues it. Changing the gain of segment `s` cannot
possibly affect the trajectory *before* segment `s` begins — the plant up to step
`s*seg_len` was produced entirely by the gains of segments `0..s-1`, which I did not
touch. The simulation is a strict left-to-right recurrence on `(x, v, e_prev, cost)`.
So if I have the plant's state at the boundary entering segment `s`, I can re-run the
simulation from *there* and get the exact same total cost I would have gotten from a
full pass — but I only pay for steps `s*seg_len .. T-1`, i.e. `O((S-s)*seg_len)`.

This is the prefix-state cache. Treat the simulator as a black box, but cache its
per-boundary state. Concretely I keep `boundary[b]` = the tuple `(x, v, e_prev, cost)`
*just before* simulating segment `b`, for every `b in [0, S]`. `boundary[0]` is the
fixed initial state `(ref[0], 0, 0, 0)`. To evaluate a candidate after perturbing a
code in segment `s`, I call `resimulate(s)`: start from `boundary[s]`, walk forward to
the end, and *update* `boundary[s+1..S]` as I go (so the cache stays consistent with
the current codes). The returned `boundary[S].cost` is the exact total cost. A move in
the *last* segment costs one segment's worth of steps; a move in segment `0` still
costs the full horizon, but the average move costs about half the horizon, and the
*expected* speedup over full-resim is a factor of ~2x — and crucially, the late
segments (where annealing spends a lot of its fine-tuning) are nearly free. The cache is
what turns a shallow search into a deep one.

There is a subtlety I have to respect: the cache must always reflect the *current*
codes. After I accept a move I leave `boundary[]` as `resimulate` left it (consistent).
After I *reject* a move I have to restore both the code and the cached state — so I put
the old code back and call `resimulate(s)` once more to repair `boundary[s+1..S]`. If I
forgot that repair, the cache would silently describe a code vector I no longer hold,
and every subsequent incremental cost would be wrong. I will make sure to test this.

## The method I will implement

Three phases, all sharing the cached re-simulation:

1. **Greedy coordinate-descent construction.** Sweep segments left to right. For each
   gain, scan a coarse grid of ~21 codes spanning `[0, Q]` and keep the best, each trial
   evaluated by `resimulate(s)` (only the tail re-runs). After choosing a gain I
   re-simulate once to commit it and refresh the cache. This climbs out of the all-zero
   start into a genuinely good schedule — a strong warm start — and because it goes
   left to right, by the time I tune segment `s` the earlier segments are already good,
   so the boundary state entering `s` is realistic.

2. **Simulated annealing over single-code moves.** From the warm start, repeatedly pick
   a random coordinate, perturb its code by a step whose span *shrinks* as time runs
   out (large exploratory jumps early, `+-1` polish late), clamp into `[0, Q]`,
   `resimulate(s)`, and accept by Metropolis on the cost delta with a geometric
   temperature schedule anchored to the per-step cost scale. This escapes the
   coordinate-descent local minimum that phase 1 inevitably sits in. I keep the best
   feasible code vector ever seen.

3. **Coordinate-descent polish.** Restore the best vector and do a few rounds of
   small-offset coordinate descent (`+-1, +-5, +-20, +-60`) to clean up to a local
   optimum.

Feasibility is structural: every code is clamped into `[0, Q]` before it is ever used,
so I can never emit an out-of-range code, and the divergence guard inside `resimulate`
returns `+inf` (and marks the downstream boundaries unusable) for any gain combination
that blows the plant up — so annealing simply rejects those moves and the best-so-far
stays finite and feasible. I print the best code vector at the end.

## Building it, and the bugs I actually hit

I wrote the generator (sinusoids + steps for the reference, drift+jitter disturbance,
everything emitted as fixed-point `*1000` integers so C++ and Python read identical
numbers), the scorer (mirror the exact recurrence, map codes to gains, apply the
feasibility floor and the divergence floor), a trivial baseline (`all-zero codes`), and
the solver. Then I compiled — and immediately hit a wall.

**Bug 1: `ref` is ambiguous.** I had named the reference array `ref`, and with
`#include <bits/stdc++.h>` plus `using namespace std;` that collides with
`std::ref` from `<functional>`. The compiler spat a wall of "reference to 'ref' is
ambiguous" notes pointing at `std::reference_wrapper`. Easy fix but a real one: I
renamed the arrays to `rf` and `dst` throughout. It compiled clean after that. (Lesson
re-learned: never name a global `ref`, `data`, `count`, or `distance` under
`using namespace std`.)

**Bug 2 — the important one: the score did not discriminate.** I ran the first
self-verify over seeds 1..20. Every output was feasible and every solver score beat the
baseline, so on paper it "passed". But the numbers were suspicious: the solver scored a
flat `1000000` on every single seed and the baseline scored ~`999500` on every seed.
A solver that hits the exact ceiling on all 20 seeds is not measuring anything — the
metric had saturated. I probed the raw costs to see what was happening:

```
seed 1:  COST_zero(open-loop) = 1.86e10   COST_baseline = 8.16e6   COST_solver = 100.98
seed 5:  COST_zero(open-loop) = 2.66e10   COST_baseline = 8.88e6   COST_solver = 133.54
seed 13: COST_zero(open-loop) = 5.00e8    COST_baseline = 4.05e5   COST_solver = 24.48
```

The open-loop cost is enormous (with no control the disturbance accumulates and the
plant runs away to ~`10^10`), while *any* controller that does something gets COST down
to `10^2`–`10^6`. My first scoring formula was
`score = 1e6 * COST_ZERO / (COST_ZERO + COST)`. With `COST_ZERO ~ 1e10` and
`COST ~ 1e2`, that ratio is `1e10 / (1e10 + 1e2) = 0.99999999…`, which rounds to
`1000000` for the solver *and* sits at ~`999500` for the baseline whose `COST ~ 1e7`.
The five-order-of-magnitude gap between solver and baseline was completely invisible
because both are astronomically smaller than `COST_ZERO`. The score told me both
controllers were "essentially perfect," which is useless — a good ALE metric has to
spread out across the realistic controlled regime, not saturate the instant you leave
open loop.

The fix is to measure performance the way control engineers actually do — in *decades*,
on a log scale. I redefined the score as the open-loop-relative cost reduction in
`log10`:

```
gain_decades = log10( COST_ZERO / max(COST, 1e-9) )
score = round( 1e5 * max(0, gain_decades) )   if feasible, else 0.
```

Now a controller that takes COST from `1e10` to `1e2` earns ~8 decades → ~`800000`,
while the baseline that only reaches `1e7` earns ~3.3 decades → ~`330000`. The metric
is monotone in COST, smooth, floors at 0 when you do no better than open loop, and —
critically — *discriminates*. Re-running seeds 1..20:

```
mean solver score   = 776250
mean baseline score = 333024     (all 20 feasible, all 20 beat baseline)
```

That is a decisive, honest margin (~8 decades of tracking improvement versus ~3.3),
exactly the kind of gap a strong heuristic should open over a trivial output.

**Cross-check: does the solver optimize the same cost the scorer measures?** This is
the one correctness property I cannot hand-wave, because the solver's incremental
re-simulation and the scorer's full simulation are two separate code paths. If they
ever disagreed by a rounding policy, the solver would be hill-climbing a different
surface than the one it is graded on. I wrote a tiny standalone C++ that replays the
exact recurrence for a given code file and compared its COST to `score.py`'s COST on
five seeds:

```
seed  cpp_COST       py_COST        reldiff
1     100.9828606    100.9828606    0.000e+00
5     133.5438428    133.5438428    0.000e+00
13     24.4761792     24.4761792    0.000e+00
...
```

Zero relative difference — they agree to the last bit, because both read the same
fixed-point integers and divide by the same `1000.0`, and both use the same `DRAG` and
the same update order. Good; the prefix-cache re-simulation is exact, not approximate.

**Feasibility floors reachable.** I checked that an out-of-range code (`Q+1`), a
negative code, and a wrong token count each score `0`, and that a deliberately
divergence-prone instance (huge `Kp`/`Kd` boxes) with max-gain codes drives the plant
past `1e15` and scores `0` — the divergence floor is live, not dead code. On that same
nasty instance my solver still finds a feasible, high-scoring schedule (its divergence
guard makes annealing reject the blow-up moves), so it never crashes and never emits an
infeasible answer.

**Bug 3 — a timing spike.** In a back-to-back sweep one seed clocked 8.24 s wall, way
over budget. Re-timing that seed in isolation showed 1.82 s — it was system contention
during the parallel sweep, not the solver. Still, to be safe under a loaded judge I
tightened the internal budget to `1.7 s`, factored the elapsed-time check into a small
lambda, and added a fine-grained time guard *inside* the construction and polish loops
(not just once per outer round) so no single phase can overrun even on the largest
`S=12` instances. After that, isolated wall times are ~1.55 s across seeds and the
scores are unchanged (mean solver `776250`).

## Final verification

On seeds 1..30: every output feasible (parses, codes in range, no divergence), the
solver beats the all-zero baseline on every seed, mean solver score ~`776k` vs baseline
~`333k`, and the worst isolated wall time is comfortably under 2 s. The prefix-state
cache is doing exactly what it was meant to: late-segment moves are nearly free, so the
annealing pass fits millions of trials into the budget and converges to schedules whose
tracking cost is ~8 decades below open loop. That is the strongest standard approach for
this structure — black-box simulation as the oracle, but with the deterministic-prefix
caching that makes a single-parameter change re-simulate only from the first divergent
step.

## Final solver

```cpp
// Parameter Placement for a Simulated Controller -- strong heuristic solver.
//
// We tune S segments of K = 3 gains each (a PD-on-error + velocity-damping triple)
// that drive a deterministic scalar plant to track a reference trajectory. Each gain
// is an integer CODE in [0, Q] mapping to value LO_k + (HI_k-LO_k)*code/Q. The
// objective is to MINIMIZE the squared tracking cost
//     COST = sum_t (ref[t] - x_after(t))^2
// and the reported score grows as COST drops below the open-loop (zero-gain) cost.
//
// THE LEVER -- PREFIX-STATE CACHE / LOCALIZED RE-SIMULATION.
//   The plant is deterministic and each segment owns DISJOINT gains, so changing a
//   single segment s's gain perturbs the trajectory ONLY from the start of segment s
//   onward. We treat the forward simulation as a black-box oracle but CACHE the plant
//   state (x, v, e_prev) and the accumulated cost at every segment boundary. A
//   single-code perturbation in segment s then re-runs the sim from boundary s only --
//   O((S-s)*seg_len) work instead of a full O(T) re-evaluation. That cheap incremental
//   delta is what makes a dense coordinate-descent / simulated-annealing search over
//   the S*K codes affordable.
//
// METHOD.
//   (1) Greedy coordinate-descent construction. Sweep segments left to right; for each
//       gain try a coarse grid of codes and keep the best, re-simulating only the tail
//       from that segment using the cached prefix state. Gives a strong warm start.
//   (2) Simulated annealing over single-code moves with the same localized re-sim and
//       O(1)-amortized acceptance, escaping the coordinate-descent local minimum.
//   (3) Final coordinate-descent polish to a local optimum.
// Every code stays in [0, Q] at all times, so every intermediate -- and the final
// output -- is FEASIBLE by construction. The simulation matches the scorer exactly
// (same DRAG, same fixed-point reads, same divergence guard).
#include <bits/stdc++.h>
using namespace std;

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static inline uint64_t xr() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}
static inline double urand() { return (xr() >> 11) * (1.0 / 9007199254740992.0); }

static const double SCALE = 1000.0;
static const double DRAG = 0.02;
static const double DIVERGE = 1e15;

int S, K, Q, T, seg_len;
vector<double> LO, HI;         // per-gain box bounds (K entries)
vector<double> rf, dst;        // reference + disturbance, length T

// current codes: code[s*K + k]
vector<int> code;
// gain VALUE for a code on gain k
inline double codeVal(int k, int c) { return LO[k] + (HI[k] - LO[k]) * (double)c / (double)Q; }

// cached plant state at each segment boundary b in [0, S]:
// state just BEFORE simulating segment b (i.e. at global step b*seg_len).
struct St { double x, v, ep, cost; bool ok; };
vector<St> boundary;           // size S+1; boundary[0] is the fixed initial state

// Re-simulate from segment boundary `b0` to the end, using boundary[b0] as the start
// state. Fills boundary[b0+1 .. S]. Returns total COST (boundary[S].cost) or +inf if
// the simulation diverges. Uses the CURRENT `code` for the gains.
double resimulate(int b0) {
    St cur = boundary[b0];
    if (!cur.ok) return numeric_limits<double>::infinity();
    double x = cur.x, v = cur.v, ep = cur.ep, cost = cur.cost;
    for (int s = b0; s < S; s++) {
        double g0 = codeVal(0, code[s * K + 0]);
        double g1 = codeVal(1, code[s * K + 1]);
        double g2 = codeVal(2, code[s * K + 2]);
        int t0 = s * seg_len, t1 = t0 + seg_len;
        for (int t = t0; t < t1; t++) {
            double e = rf[t] - x;
            double f = g0 * e + g1 * (e - ep) + g2 * v;
            v = v + f - DRAG * v + dst[t];
            x = x + v;
            double err = rf[t] - x;
            cost += err * err;
            ep = e;
            if (!isfinite(x) || !isfinite(v) || !isfinite(cost) ||
                fabs(x) > DIVERGE || fabs(v) > DIVERGE || cost > DIVERGE) {
                // mark all downstream boundaries diverged
                for (int b = s + 1; b <= S; b++) boundary[b] = {0, 0, 0, 0, false};
                return numeric_limits<double>::infinity();
            }
        }
        boundary[s + 1] = {x, v, ep, cost, true};
    }
    return boundary[S].cost;
}

int main() {
    auto t_start = chrono::steady_clock::now();
    const double TIME_LIMIT = 1.7; // seconds, comfortably under the 2s budget
    auto elapsed = [&]() {
        return chrono::duration<double>(chrono::steady_clock::now() - t_start).count();
    };

    // ---- read instance ----
    if (scanf("%d %d %d", &S, &K, &Q) != 3) return 0;
    LO.assign(K, 0); HI.assign(K, 0);
    for (int k = 0; k < K; k++) {
        long long lo, hi;
        if (scanf("%lld %lld", &lo, &hi) != 2) return 0;
        LO[k] = lo / SCALE; HI[k] = hi / SCALE;
    }
    if (scanf("%d", &T) != 1) return 0;
    rf.assign(T, 0); dst.assign(T, 0);
    for (int t = 0; t < T; t++) { long long r; if (scanf("%lld", &r) != 1) return 0; rf[t] = r / SCALE; }
    for (int t = 0; t < T; t++) { long long d; if (scanf("%lld", &d) != 1) return 0; dst[t] = d / SCALE; }
    seg_len = T / S;

    // ---- fixed initial boundary state ----
    boundary.assign(S + 1, St{0, 0, 0, 0, false});
    boundary[0] = {rf[0], 0.0, 0.0, 0.0, true};    // x0 = ref[0], v0 = 0, e_prev0 = 0

    // ---- start from all-zero codes (feasible) ----
    code.assign(S * K, 0);
    double curCost = resimulate(0);

    // ---- (1) greedy coordinate-descent construction, left to right ----
    // For each segment/gain, scan a coarse code grid and keep the best, re-simulating
    // only the tail from that segment. The cached prefix state makes each trial cheap.
    {
        const int GRID = 21;
        bool stop = false;
        for (int s = 0; s < S && !stop; s++) {
            for (int k = 0; k < K; k++) {
                int idx = s * K + k;
                int bestC = code[idx];
                double bestCost = curCost;
                for (int gi = 0; gi <= GRID; gi++) {
                    int c = (int)llround((double)Q * gi / GRID);
                    if (c < 0) c = 0; if (c > Q) c = Q;
                    code[idx] = c;
                    double cst = resimulate(s);   // only seg s..S-1 re-run
                    if (cst < bestCost) { bestCost = cst; bestC = c; }
                }
                code[idx] = bestC;
                curCost = resimulate(s);          // commit the chosen code, refresh cache
                if (elapsed() > TIME_LIMIT * 0.4) { stop = true; break; } // warm start only
            }
        }
    }

    // best-so-far snapshot
    vector<int> best = code;
    double bestCost = curCost;

    // ---- (2) simulated annealing over single-code moves ----
    // A move perturbs one code in some segment s by a (decaying) step; we re-simulate
    // only segments s..S-1 from the cached boundary[s]. Accept by Metropolis on the
    // cost delta. Keep the best feasible coloring seen.
    {
        // characteristic cost scale for the temperature schedule
        double costScale = max(1.0, bestCost / max(1, T));
        double T0 = costScale * 50.0, T1 = costScale * 0.05;
        long long iter = 0;
        // we must keep boundary[] consistent with `code`; resimulate(0) once to sync
        curCost = resimulate(0);
        while (true) {
            if ((iter & 1023) == 0) {
                if (elapsed() > TIME_LIMIT * 0.9) break;
            }
            iter++;
            double frac = elapsed() / TIME_LIMIT;
            if (frac > 1) frac = 1;
            double Temp = T0 * pow(T1 / T0, frac);

            int idx = (int)(xr() % (uint64_t)(S * K));
            int s = idx / K;
            int oldC = code[idx];
            // step size shrinks over time; at least 1
            int span = max(1, (int)llround((double)Q * (0.5 - 0.45 * frac)));
            int step = (int)(xr() % (uint64_t)(2 * span + 1)) - span;
            int newC = oldC + step;
            if (newC < 0) newC = 0; if (newC > Q) newC = Q;
            if (newC == oldC) continue;

            code[idx] = newC;
            double cand = resimulate(s);          // localized re-sim from segment s
            double d = cand - curCost;
            if (d <= 0 || urand() < exp(-d / Temp)) {
                curCost = cand;                   // accept; boundary[] now matches code
                if (curCost < bestCost) { bestCost = curCost; best = code; }
            } else {
                code[idx] = oldC;                 // reject; restore code...
                resimulate(s);                    // ...and the cached boundary state
            }
        }
    }

    // ---- (3) final coordinate-descent polish on the best codes ----
    code = best;
    curCost = resimulate(0);
    {
        bool improved = true;
        int rounds = 0;
        bool stop = false;
        const int offs[] = {1, -1, 5, -5, 20, -20, 60, -60};
        while (improved && rounds < 50 && !stop) {
            improved = false; rounds++;
            for (int s = 0; s < S && !stop; s++) {
                for (int k = 0; k < K; k++) {
                    int idx = s * K + k;
                    int oldC = code[idx];
                    int bestC = oldC; double bc = curCost;
                    for (int o : offs) {
                        int c = oldC + o;
                        if (c < 0 || c > Q) continue;
                        code[idx] = c;
                        double cst = resimulate(s);
                        if (cst < bc) { bc = cst; bestC = c; }
                    }
                    code[idx] = bestC;
                    curCost = resimulate(s);
                    if (bestC != oldC) improved = true;
                    if (elapsed() > TIME_LIMIT) { stop = true; break; }
                }
            }
        }
        if (curCost < bestCost) { bestCost = curCost; best = code; }
    }

    // ---- output: S lines of K codes (always in [0,Q] -> feasible) ----
    code = best;
    string out;
    out.reserve(8 * S * K);
    for (int s = 0; s < S; s++) {
        for (int k = 0; k < K; k++) {
            out += to_string(code[s * K + k]);
            out.push_back(k + 1 < K ? ' ' : '\n');
        }
    }
    fputs(out.c_str(), stdout);
    return 0;
}
```
