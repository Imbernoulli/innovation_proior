# Hidden Pipe Network: Identifying a Shared Diffusion Operator from Impulse Responses

## Problem
A district heating utility has `n = 30` junctions connected by a **hidden**
sparse network of pipes (average degree about 2.5 -- every junction is linked
to only a couple of neighbours). Each pipe `(i, j)` carries an unknown,
nonnegative conductance `w_ij`. Heat obeys the standard leaky-diffusion update
at every one-minute tick:

```
x[t+1]_i = x[t]_i + sum_{j adjacent to i}  w_ij * (x[t]_j - x[t]_i)
```

This conserves total heat (no leak to the environment) and is exactly
`x[t+1] = A x[t]` for a fixed matrix `A` with `A_ii = 1 - sum_j w_ij` and
`A_ij = A_ji = w_ij` on pipes, `0` elsewhere. **The pipe layout and its
conductances are the same matrix `A` for every experiment below** -- you are
identifying one shared operator, not five unrelated curves.

Field technicians ran 5 experiments: at each of 5 chosen junctions they
injected one unit of heat (all other junctions start at 0) and logged every
junction's temperature for several further ticks, with small sensor noise.
Your job: submit a weighted pipe layout. It will be graded by simulating it
forward on impulses injected at **different, unseen** junctions.

**Illustrative FORM only -- not the real size or graph:** on a toy 4-junction
line 0-1-2-3 with pipes (0,1,w=0.30) and (2,3,w=0.20), an impulse at junction 0
spreads only toward 1, never reaching 2 or 3. Real instances have 30 junctions
and a genuinely sparse but connected layout that you must discover from data.

## Input (stdin)
```
t n S T
src_1 src_2 ... src_S
<T+1 rows of n floats: impulse-response of source src_1>
<T+1 rows of n floats: impulse-response of source src_2>
...
```
`t` is the test id, `n = 30`, `S = 5` training sources, `T` is the number of
post-impulse ticks recorded. For source `src_k`'s block, row `0` is the exact
impulse (`1.0` at `src_k`, else `0.0`); rows `1..T` are the noisy observed
temperatures at that tick.

## Output (stdout)
```
m
i_1 j_1 w_1
...
i_m j_m w_m
```
`m` is the number of pipes you claim exist; each line gives an edge between
junctions `i` and `j` (`0 <= i,j < n`, `i != j`) with conductance `w`. Submit
`m = 0` (just the line `0`) to claim no pipes at all.

## Feasibility
All of the following must hold, else the submission scores `Ratio: 0.0`:
- exactly `m` edge lines follow the header, all finite;
- `0 <= i,j < n`, `i != j`, and each unordered pair `(i,j)` appears at most once;
- every weight satisfies `0 < w <= 1.0`;
- for every junction, the sum of its incident weights is `<= 1.0` (a junction
  cannot lose more heat per tick than it holds).

## Objective (minimize)
Let `A_hat` be the operator implied by your edges. The grader re-runs impulses
at 4 held-out junctions (never used in your training data) for a longer
horizon under the TRUE hidden operator (plus a small irreducible noise floor)
and under your `A_hat`, and computes the mean squared error `F` between the
two rollouts, plus a small parsimony surcharge for edge count.

## Scoring
```
F = heldout_MSE(A_hat) * (1 + 0.02 * m)
B = heldout_MSE(no-diffusion baseline)          # A = identity, 0 edges
sc = min(1000, 100 * B / max(1e-9, F))
Ratio = sc / 1000
```
Claiming no pipes reproduces `B` exactly, scoring `Ratio = 0.1`. Recovering
the true sparse operator drives `F` far below `B`; sensor noise and estimation
error keep even an excellent submission below the cap, so `Ratio = 1.0` is not
reachable.

## Constraints
- `1 <= t <= 10^5` (test id); `n = 30`; `S = 5`; time limit 5s, memory 512m.
- Each `.in` file is well under 5 MB.

## Example (worked score)
Suppose the checker's no-diffusion baseline gives `heldout_MSE = 0.0400`
(`B = 0.0400`). A submission with `m = 40` edges achieves
`heldout_MSE(A_hat) = 0.0120`, so `F = 0.0120 * (1 + 0.02*40) = 0.0120*1.8 =
0.0216`. Then `sc = 100 * 0.0400 / 0.0216 = 185.2`, giving `Ratio = 0.1852`.
