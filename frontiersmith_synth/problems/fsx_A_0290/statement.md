# Ridgeline Watch: Budgeted Fire-Tower Placement

**Family:** heuristic-contest-offline (AtCoder-heuristic-contest style, scored offline by a
deterministic contest formula) · **Format:** B (isolated heuristic evaluation) · **Objective:** maximize

## Story

A forestry service is fortifying a mountain valley against wildfire by raising a network of
fire-lookout towers. The valley is an `N x N` grid of forest cells. Each cell has an integer
**fire-risk weight** — dry, wind-exposed ridgelines score high, damp shaded gullies score low.

There is a fixed menu of `M` candidate tower **sites**. A tower built at site `j` sits at grid
position `(ty[j], tx[j])` and **watches** every cell within Chebyshev distance `tr[j]` of it: the
`(2*tr[j]+1) x (2*tr[j]+1)` square centred on the site, clipped to the grid. A taller tower (larger
patrol radius) watches more but **costs more** to raise and staff — site `j` costs `tc[j]`.

The service has a fixed capital **budget** `B`. Choose a subset of sites to build whose total cost
stays within budget so as to **maximize the total fire-risk weight of the forest cells watched by at
least one built tower**. A cell watched by several towers still counts **once** — redundant overlap
is wasted money.

This is *budgeted maximum coverage* (NP-hard) skinned as fire-tower siting. A coverage-blind rule is
easily beaten; plain marginal-gain greedy leaves value on the table where overlap and lumpy costs
interact; local search over add / drop / swap moves does better still. The offline "operation budget"
of the contest is enforced by the harness as an isolation timeout — your program is run in a sandboxed
subprocess and only the JSON answer it prints is scored, by a fixed deterministic formula.

## Candidate program contract (stdin -> stdout)

Read ONE JSON object (the **public instance**) from stdin and write ONE JSON object to stdout.

**Input (public instance):**
```json
{"name": "valley2901", "N": 24, "M": 34, "B": 190,
 "weight": [[w_00, w_01, ...], ...],
 "tx": [x_0, ...], "ty": [y_0, ...],
 "tr": [r_0, ...], "tc": [c_0, ...]}
```
- `weight` is an `N x N` grid of non-negative integers; `weight[row][col]` is cell `(row, col)`'s risk.
- Site `j` is at column `tx[j]`, row `ty[j]` (both in `0..N-1`), patrol radius `tr[j]`, build cost `tc[j]`.
- Site `j` watches every cell `(row, col)` with `|row - ty[j]| <= tr[j]` and `|col - tx[j]| <= tr[j]`.

**Output (answer):**
```json
{"build": [j0, j1, ...]}
```
the list of **distinct** site indices to build. Order is irrelevant.

A skeleton:
```python
import sys, json
inst = json.load(sys.stdin)
# ...choose sites within budget...
print(json.dumps({"build": chosen}))
```

## Validity

A plan is **valid** iff `build` is a list of **distinct** integers in `[0, M)` whose total cost
`sum(tc[j] for j in build) <= B`. A duplicate index, an out-of-range index, an over-budget plan, a
crash, a timeout, or non-JSON output makes that instance score **0.0**.

## Scoring (deterministic, no wall-time)

For each instance the evaluator computes two references in the parent process:
- `q_base` = watched weight of the internal **cost-ascending fill** (buy cheapest sites first while
  the budget lasts — coverage-blind).
- `q_full` = watched weight if **every** site were built (the union ceiling, ignoring budget — in
  general unaffordable, hence unreachable).

Let `q_cand` be the watched weight of your valid plan. Your per-instance score is the affine anchor
```
r = clamp( 0.1 + 0.9 * (q_cand - q_base) / max(1e-9, q_full - q_base),  0, 1 )
```
Matching the cost-ascending fill scores ~0.1; reaching the (budget-infeasible) all-towers ceiling
scores 1.0; doing worse than the fill scores below 0.1. The reported **Ratio** is the mean of `r`
over all instances (a mix of medium and larger, tighter-budget held-out valleys).

## Isolation

Your program is untrusted: it runs in a fresh sandboxed subprocess via `isorun.run_candidate` and only
ever sees the public instance above. The references and the scoring are computed by the parent process,
so introspection / frame-walking / source-reading buys nothing.
