# The Narrow Band: Localized Slow-Cooling for Glass Annealing

## Problem

A glass part must be cooled from `T_hot` down to `T_cold` through `M`
equal-width temperature segments (segment `i` spans `T[i-1] -> T[i]`,
where `T[0]=T_hot`, `T[M]=T_cold`, uniformly spaced). You choose a
positive duration `t_i` for each segment; the implied cooling rate is
`r_i = (T[i-1]-T[i]) / t_i`.

**Stress-relaxation-time.** The glass's local structural relaxation time
follows an Arrhenius law, evaluated at each segment's midpoint temperature
and held fixed across the segment: `tau_i = tau0 * exp(k / Tbar_i)`,
`Tbar_i = (T[i-1]+T[i])/2`. `tau` grows **exponentially** as temperature
falls.

**Structural-relaxation-lag.** The glass's internal ("fictive") structural
temperature `Tf` lags the true temperature whenever cooling isn't
quasi-static. Starting from `Tf_0 = T_hot` (fully relaxed liquid), each
segment updates `Tf` by the exact solution of `dTf/dt = (T(t)-Tf)/tau_i`
for a linearly-changing `T(t)`:

```
Tf_i = T[i] + r_i*tau_i + (Tf_{i-1} - T[i-1] - r_i*tau_i) * exp(-t_i / tau_i)
```

After all `M` segments, the residual structural **mismatch** is
`|Tf_M - T_cold|` and must be `<= tol`.

**Cooling-rate-gradient.** The oven cannot jump its rate between
consecutive segments by more than `rate_grad_max`: `|r_i - r_{i-1}| <=
rate_grad_max` for every `i`, where `r_0 := 0` (the oven starts at rest,
so segment 1's rate is gradient-capped too). Also `0 < r_i <= rate_max`
always.

Because `tau(T)` is exponential in `T`, the constraint that governs
feasibility only really *binds* in a narrow temperature band -- far above
it the glass relaxes essentially instantly (fast cooling is free there),
far below it the glass is already effectively frozen (slowing down
further barely changes the mismatch).

## Input (stdin)

```
M
T_hot T_cold
tau0 k
rate_max rate_grad_max
tol div
```
`M` is an integer. All other values are reals. `div>1` is used only by
the checker's own reference construction (see Scoring) -- it never
affects feasibility.

## Output (stdout)

`M` whitespace-separated positive reals `t_1 ... t_M`: the duration of
each segment, hottest first.

## Feasibility

- Exactly `M` tokens, each a finite positive real number.
- For every `i`: `0 < r_i <= rate_max` and `|r_i - r_{i-1}| <=
  rate_grad_max` (`r_0=0`).
- Final mismatch `|Tf_M - T_cold| <= tol`.
Any violation scores `Ratio: 0.0`.

## Objective and Scoring

Minimize `F = sum(t_i)`, the total anneal time. The checker also builds
its own reference schedule: the fastest **constant** (uniform) rate whose
resulting mismatch stays under a **stricter** budget `tol/div` (instead of
the real `tol`) -- i.e. an extra-cautious "just go uniformly slow"
construction, with total time `B`. It reports

```
Ratio = min(1, 0.1 * B/F)   ... printed as "Ratio: <value in [0,1]>"
```

so matching the reference exactly gives `Ratio ~= 0.1`, and a schedule
10x faster than the reference saturates at `Ratio = 1.0`.

## Constraints

`16 <= M <= 40`, `250 <= T_cold < T_hot <= 1300`, `0 < tau0 <= 1e-3`,
`k > 0`, `0 < rate_grad_max <= rate_max <= 400`, `tol > 0`, `1 < div <=
2`. Time limit 5s, memory 512MB.

## Example (worked score, illustrative shape only)

`M=2`, `T=[1000, 700, 400]`, `tau0=0.01`, `k=2000`, `rate_max=300`,
`rate_grad_max=100`, `tol=50`, `div=1.2`.

Submit `t = (4, 3)`. Rates: `r_1 = 300/4 = 75` (`<=300`, jump from rest
`75<=100` OK), `r_2 = 300/3 = 100` (jump `25<=100` OK). Simulating gives
mismatch `~= 37.94 <= 50` (feasible), so `F = 7`.

The checker's reference bisects the fastest uniform rate meeting
`tol/div ~= 41.67`; that turns out to be `r ~= 109.8`, giving `B ~=
5.73`. So `Ratio = min(1, 100*5.73/7)/100 ~= 0.082`.

The real test cases are calibrated so that a single global (uniform)
rate must be slow enough to survive the narrow band where `tau` is
comparable to the segment-traversal time -- and paying that rate for the
*whole* schedule costs far more total time than paying it only inside
that band.
