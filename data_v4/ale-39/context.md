# Parameter Placement for a Simulated Controller

## Research question

A deterministic scalar plant must be driven to track a reference trajectory
`ref[0..T-1]` over a fixed horizon. The "controller" is a bank of gains: the
horizon is split into `S` equal-length **segments**, and each segment `s` owns its
own gain vector `g[s][0..K-1]` (here `K = 3`: a PID-like triple — proportional on
the error, derivative on the error, and a velocity-damping term). At each step the
controller in the active segment observes the tracking error and produces a force
that the plant integrates; the disturbance `d[t]` perturbs the state. The task is to
**choose the gains so as to minimize the accumulated squared tracking error**.

The gains are continuous, but they are reported on a fixed lattice: for each gain the
solver emits an integer **code** in `[0, Q]`, which the scorer maps to a real value
inside that gain's box `[LO_k, HI_k]`. So the search is a continuous tuning problem
that is finally *quantized to a grid* — pick `S*K` codes that drive a black-box
forward simulation to the lowest tracking cost. This is the controller-tuning
archetype: the simulator is cheap and deterministic, but the decision vector is
moderately high-dimensional and the cost surface is non-convex (gains interact across
segments through the carried-over plant state), so it is judged as a continuous
heuristic-optimization problem rather than solved in closed form.

## Input / output contract

- **Input (stdin):**
  - First line: three integers `S K Q` — number of segments, gains per segment
    (`K = 3`), and quantization levels (`Q = 1000`).
  - Second line: `K` pairs `LO_k HI_k`, the box bounds of gain `k` as fixed-point
    integers (the real bound is the integer divided by `1000`). With the standard
    boxes, gain 0 (proportional) and gain 1 (derivative) are non-negative and gain 2
    (velocity damping) is non-positive.
  - Third line: the horizon `T` (an exact multiple of `S`; `seg_len = T / S`).
  - Fourth line: `T` integers, the reference `ref[t]` as fixed-point (`*1000`).
  - Fifth line: `T` integers, the disturbance `d[t]` as fixed-point (`*1000`).
- **Output (stdout):** `S * K` integer codes — the natural layout is `S` lines of `K`
  codes, code `code[s][k] in [0, Q]` for gain `k` of segment `s`. (Any whitespace
  layout yielding exactly `S*K` integer tokens is accepted.)
- **Time limit:** about 2 seconds wall-clock per instance. Memory: 256 MB.

## Background

This is *controller-tuning by simulation*: we cannot write the optimal gains down,
but we can run the plant forward cheaply for any candidate gain schedule and read off
the cost. Several method families are on the table before committing to one:

- **A single global gain (naive baseline).** Use one gain triple for the whole
  horizon. Easy, but a reference that changes character across the horizon (sinusoids
  of different phase, level steps) needs different damping/aggressiveness at different
  times, so one global setting leaves cost on the table.
- **Black-box search over the full `S*K` vector** with Nelder–Mead / CMA / coordinate
  descent, each candidate evaluated by a **full** `O(T)` forward simulation. Correct,
  but every single-coordinate perturbation pays the entire horizon, so the number of
  evaluations that fit in the budget is small and the search is shallow.
- **Localized re-simulation with a prefix-state cache (the lever).** The plant is
  deterministic and each segment owns *disjoint* gains, so perturbing one segment `s`'s
  gain changes the trajectory only from the start of segment `s` onward. Caching the
  plant state `(x, v, e_prev)` and the accumulated cost at every segment boundary lets
  a single-code move re-run the simulation from boundary `s` only — `O((S-s)*seg_len)`
  work instead of `O(T)`. That cheap incremental delta is what makes a dense
  coordinate-descent + simulated-annealing search over the codes affordable, turning a
  shallow search into a deep one within the same budget.

The strongest practical recipe: a greedy coordinate-descent **construction** (sweep
segments left to right, grid-scan each gain) for a strong warm start, then
**simulated annealing** over single-code moves to escape the coordinate-descent local
minimum, then a coordinate-descent **polish** — all three reusing the cached prefix
state so each candidate costs only the affected tail.

## Evaluation settings

A solution is **feasible** iff (1) it parses as exactly `S*K` integer tokens and
(2) every code lies in `[0, Q]`. Each code maps to a gain value
`g[s][k] = LO_k + (HI_k - LO_k) * code[s][k] / Q`. The deterministic plant
(unit mass, `dt = 1`, drag `DRAG = 0.02`) is then simulated:

```
x = ref[0]; v = 0; e_prev = 0; COST = 0
for t in 0..T-1:
    s = t // seg_len                          # active segment
    e = ref[t] - x
    f = g[s][0]*e + g[s][1]*(e - e_prev) + g[s][2]*v
    v = v + f - DRAG*v + d[t]
    x = x + v
    COST += (ref[t] - x)^2
    e_prev = e
```

If at any step `x`, `v` or `COST` becomes NaN/inf or exceeds `1e15`, the simulation
has **diverged** and the solution is infeasible. The same plant is run with all gains
`= 0` (open loop, `f == 0`) to obtain `COST_ZERO`. Tracking performance spans many
orders of magnitude, so the score measures the open-loop-relative cost reduction in
**decades** (`log10`):

```
gain_decades = log10( COST_ZERO / max(COST, 1e-9) )
score = round( 1e5 * max(0.0, gain_decades) )   if the solution is FEASIBLE,
score = 0                                        otherwise (the feasibility -> 0 floor).
```

A controller no better than open loop scores `0`; every decade by which it drives the
tracking cost below open loop is worth `1e5` points (a controller cutting cost from
`~1e10` to `~1e2` scores `~8e5`). The score rises monotonically and smoothly as `COST`
falls. Any infeasible output — wrong token count, an out-of-range code, or a diverging
simulation — floors the score to `0`.

**Instances.** A generator seeds a `random.Random`, draws `S in [6,12]` segments of
length `[40,70]`, builds the reference as a sum of `2..4` sinusoids (varied amplitude,
period, phase) plus `1..3` level steps, and a small bounded disturbance (a low-frequency
drift plus jitter). The per-gain boxes make the proportional/derivative gains positive
and the damping gain negative. The seed set, generator, and scorer are frozen; the mean
score over the seeds is reported. The only editable thing is the solver.

## Code framework

A single self-contained C++17 program reading the instance on stdin and writing a
feasible code grid on stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int S, K, Q, T, seg_len;
vector<double> LO, HI;     // per-gain box bounds
vector<double> rf, dst;    // reference, disturbance (length T)

int main() {
    if (scanf("%d %d %d", &S, &K, &Q) != 3) return 0;
    LO.assign(K, 0); HI.assign(K, 0);
    for (int k = 0; k < K; k++) {
        long long lo, hi; scanf("%lld %lld", &lo, &hi);
        LO[k] = lo / 1000.0; HI[k] = hi / 1000.0;
    }
    scanf("%d", &T);
    rf.assign(T, 0); dst.assign(T, 0);
    for (int t = 0; t < T; t++) { long long r; scanf("%lld", &r); rf[t] = r / 1000.0; }
    for (int t = 0; t < T; t++) { long long d; scanf("%lld", &d); dst[t] = d / 1000.0; }
    seg_len = T / S;

    // Feasible start: all-zero codes (every code in [0, Q]).
    vector<int> code(S * K, 0);

    // TODO heuristic: cache the plant state at each segment boundary so a single-code
    //                 change re-simulates only the tail; coordinate-descent construction,
    //                 then simulated annealing, then a coordinate-descent polish.

    // Output S lines of K codes (all in [0, Q] -> feasible).
    for (int s = 0; s < S; s++)
        for (int k = 0; k < K; k++)
            printf("%d%c", code[s * K + k], k + 1 < K ? ' ' : '\n');
    return 0;
}
```
