# Feedback Delay Network: Matching a Decay Curve and an Echo-Density Curve

## Problem
You are tuning a **feedback delay network (FDN)** reverb with `N` delay lines. The
architecture is fixed: line `i` is a circular buffer of integer length `L_i` samples,
and the lines are mixed every sample through a fixed `N x N` feedback matrix
`M = I - (2/N) * J`, where `J` is the all-ones matrix (a Householder reflection of the
all-ones vector; `M` is symmetric and orthogonal). Only the `N` delay lengths and `N`
per-line gains are yours to design.

At sample `n`, each line reads its current head sample `r_i`, the lines are mixed as
`fb = M @ r`, a unit impulse is added to `fb_0` only at `n = 0`, then each line writes
`g_i * fb_i` into its buffer (overwriting the value it just read) and advances its
pointer. The output is `y[n] = mean_i(r_i)`. Every pass a sample makes through line `i`
attenuates it by `g_i` and delays it by `L_i` — so an echo's arrival time is always a
sum `n = sum_i k_i * L_i` for nonnegative integers `k_i`.

## Input (stdin)
```
N T Lmin Lmax
K
t_1 t_2 ... t_K
db_1 db_2 ... db_K
dens_1 dens_2 ... dens_K
w_decay w_density
```
`N` delay lines, simulation length `T` samples, allowed delay-length range
`[Lmin, Lmax]`. `K` checkpoints `t_1 < ... < t_K = T` define the target **energy-decay
curve** (`db_k`, in dB, non-increasing) and the target **echo-density curve** (`dens_k`,
in `[0,1]`, non-decreasing). `w_decay`, `w_density` weight the two error terms.

## Output (stdout)
Two lines: `N` integers `L_1 ... L_N` with `Lmin <= L_i <= Lmax`, then `N` reals
`g_1 ... g_N` with `0.001 <= g_i <= 0.99`.

## Feasibility
- Exactly `N` length tokens then `N` gain tokens; all finite; lengths are integers in
  `[Lmin,Lmax]`; gains in `[0.001, 0.99]`.
- Stability: letting `G = diag(g)`, the spectral radius of `G @ M` must be `< 1`
  (checked directly via eigenvalues; it is in fact automatically implied by the
  per-line gain bound, but is checked explicitly).
Any violation scores `Ratio: 0.0`.

## Objective (minimize)
Simulate the network for `T` samples to get `y[0..T-1]`, then split `[0,T)` into `K`
segments at the checkpoints (segment `k` = samples `t_{k-1}..t_k-1`, `t_0=0`).
- **Decay error**: let `p_k` = mean(`y^2`) over segment `k`. The measured level at
  checkpoint `k` is `10*log10(p_k / p_1)` — i.e. in dB **relative to the first
  segment's own level** (this cancels any arbitrary overall output-amplitude scale;
  `db_1` is always `0` and `db_k` for `k>1` is the target, defined the same way from
  the underlying decay-envelope model). `decay_err` = mean absolute difference from
  `db_k` over the `K` checkpoints.
- **Echo-density error**: for a segment's samples, `N_eff = (sum y^2)^2 / sum y^4`
  estimates the effective number of comparable-magnitude echoes present;
  `density = min(1, N_eff / segment_length)`. `dens_err` = mean absolute difference
  from `dens_k` over the `K` segments.

`F = w_decay * decay_err + w_density * dens_err` (smaller is better).

## Scoring
The checker also simulates its own internal baseline to get `B`: delay lengths =
ascending multiples of `Lmin` (the density descriptor is left unaddressed), and a
single global gain for every line, fit crudely from the last checkpoint alone
(`g = 10^(slope * Lavg / 20)` with `slope = db_K / t_K` — a one-line, single-exponential
fit, not the two-stage reasoning). With your `F`,
```
sc    = min(1000, 100 * B / max(1e-9, F))
Ratio = sc / 1000
```
so matching the baseline scores about `0.1`; a substantially smaller `F` scores higher
(capped at `1.0`). There is no known closed-form optimum — decay shape and echo density
are two separate descriptors that trade off against each other under the fixed `[Lmin,
Lmax]` range and fixed matrix `M`, and different length/gain designs win on different
instances.

## Example (illustrative FORM only — not scored)
For `N=2`, if `L = [10, 21]` (coprime) and `g = [0.5, 0.9]`, the fast line 0 dominates
the early samples (contributing a steep initial slope) while line 1's longer, higher-gain
loop dominates later samples (a shallow tail) — a bent, two-stage decay curve, with
arrivals `10a + 21b` spreading out quickly since `gcd(10,21)=1`.

## Constraints
`4 <= N <= 8`, `1500 <= T <= 4500`, `K = 5`, `Lmin >= 20`, `Lmax <= 190`. Simulation and
scoring are `O(N^2 * T)`, well within the time limit.
