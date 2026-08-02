# Determined Combinations: Rate Constants From Equilibrated Snapshots

## Problem
A well-mixed reactor holds several **independent two-species modules**. Each
species pair `(u, v)` in a module is linked by first-order mass-action
edges, each edge `src -> dst` contributing `-k*[src]` to `d[src]/dt` and
`+k*[src]` to `d[dst]/dt`. There are two kinds of modules:

- **chain** module: a single forward edge `u -> v` with unknown rate
  constant `k`. Nothing else touches `u` or `v`.
- **pair** module: a *reversible* edge pair `u -> v` (rate `kf`) and
  `v -> u` (rate `kr`). Nothing else touches `u` or `v`.

Because no other reaction drains or feeds `u` or `v`, every module
**conserves its own total mass** `[u](t) + [v](t)` exactly, for all `t`,
regardless of the rate constants.

You are given the network topology and a handful of **noisy concentration
snapshots** taken at increasing times `t_1 < t_2 < ... < t_T` (`t_1 > 0`),
starting from an unknown initial condition. Your task: output one rate
constant per edge that reproduces the reactor's dynamics as faithfully as
possible -- including from *initial conditions you never observed*.

Illustrative FORM only (not the real instance shape): if a single isolated
species decayed as `[u](t) = [u](0) * exp(-k*t)`, two clean snapshots would
pin `k` exactly. Real instances mix chain and pair modules and add
measurement noise.

## Input (stdin)
```
testId
n_species n_modules n_edges n_snapshots K_MAX
n_modules lines: module_id type u v         (type: 0=chain, 1=pair)
n_edges   lines: edge_id module_id src dst
1 line:  t_1 t_2 ... t_T
n_snapshots lines: c_0 c_1 ... c_{n_species-1}   (concentrations at t_i)
```
For a chain module exactly one edge appears (`u -> v`). For a pair module
exactly two edges appear: `u -> v` (rate `kf`) then `v -> u` (rate `kr`).
Species/edge/module ids are 0-indexed. All rate constants lie in
`[0, K_MAX]`.

## Output (stdout)
Exactly `n_edges` lines, one per edge, in any order:
```
edge_id rate
```
`rate` must be a finite number in `[0, K_MAX]`. Every edge id from the input
must appear exactly once.

## Feasibility
An output is rejected (score 0) if: it is not parseable as `n_edges`
`"edge_id rate"` lines; any edge id is missing, duplicated, or unknown; or
any rate is non-finite or outside `[0, K_MAX]`.

## Objective
Maximize fidelity of the recovered rate constants, measured by how well they
reproduce the reactor's trajectory from a **held-out initial condition** you
never see the data for -- not by matching the visible snapshots. The
held-out probe deliberately starts some pair modules far from mass-action
equilibrium (all mass in one of the two species), so it directly tests
whether a module's *absolute* relaxation speed, not just its ratio, was
recovered soundly.

## Scoring
The checker simulates both the true and your submitted rate constants
forward (closed-form) from the held-out initial condition, at several fixed
times, and computes the RMS trajectory error normalized per-module by that
module's conserved total mass. Your score is this error compared against an
internal flat-rate baseline the checker builds itself:
`score = min(1, 100 * baseline_error / max(1e-9, your_error) / 1000)`.
Smaller trajectory error is better; matching the true reactor essentially
exactly is not achievable (T1-scale calibration slack is intentional), so
headroom remains above any reference solution.

## Constraints
`1 <= n_modules <= 8`, `n_species <= 16`, `2 <= n_edges <= 16`,
`4 <= n_snapshots <= 6`, `t_1 > 0`, `K_MAX = 400`. Time limit 5s, memory
512MB. 10 test cases, increasing measurement noise.

## Example (worked score, illustrative)
Suppose a pair module shows `[u]/[v]` essentially unchanged (say `1:3`)
across every snapshot, while a chain module's `[u]` visibly decays across
snapshots. A submission that (a) fits the chain edge's `k` from the decay
trend and (b) reports the pair's `kf:kr` as `1:3` at a scale large enough to
have already equilibrated by `t_1` will reproduce the held-out trajectory
closely, scoring well above the flat-guess baseline. A submission that
instead tries to curve-fit `kf` and `kr` independently to the (flat) pair
snapshots will typically drift toward a near-zero pair rate that visibly
fails to re-equilibrate the held-out probe.
