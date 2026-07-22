# Water-Filling a Measurement Budget Across Coupled Quantities

A lab must estimate `Q` unknown quantities before a deadline. It has `M` possible
**measurement types**; running type `m` a total of `n_m` times (an integer dial,
`0 <= n_m <= cap_m`) costs `n_m` budget units, and total spend across all types must
not exceed a fixed integer `budget`. Every measurement has a concave, saturating
**cost-to-precision curve** — the first few runs teach a lot, later runs teach less.
Some measurements are **private** probes, informative about only one quantity; others
are **shared** probes whose single run is simultaneously informative about several
quantities at once (a coverage weight given explicitly per quantity — nothing is
truly hidden, but funding it well requires reading the numbers, not the story). You
decide, once and for all, how many times to run each type, then are graded on the
**worst** (largest) posterior variance across all `Q` quantities.

## Candidate program contract

Standalone program: read ONE JSON object (the public instance) from **stdin**, write
ONE JSON object (your answer) to **stdout**. It runs isolated and sees only the public
instance.

```python
import sys, json
inst = json.load(sys.stdin)
# ... decide alloc ...
print(json.dumps({"alloc": alloc}))
```

### Public instance (stdin)

```json
{
  "instance_id": "t3",
  "n_quantities": 2,
  "n_measurements": 4,
  "budget": 20,
  "prior_precision": [9.0, 0.35],
  "gain_a": [3.0, 2.6, 18.0, 0.6],
  "gain_b": [0.35, 2.6, 6.5, 0.4],
  "cap":    [4, 4, 22, 3],
  "coverage": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]
}
```

`prior_precision[q]`: information about quantity `q` known before any measurement.
`gain_a[m]`, `gain_b[m]`: the concave curve `gain_m(n) = gain_a[m] * n / (n + gain_b[m])`
— the *extra* information one measurement type `m` contributes once it has been run
`n` times (it saturates towards `gain_a[m]`, never reaches it). `cap[m]`: maximum
allowed runs of measurement `m`. `coverage[m][q]`: how much of measurement `m`'s
gain applies to quantity `q` (0 if unrelated; several nonzero entries in one row
means that measurement is a shared probe — its gain is added, unweighted-per-quantity,
to *every* quantity it covers, and it still only costs one run per unit spent).

### Answer (stdout)

`alloc`: a list of exactly `n_measurements` **non-negative integers**, `alloc[m] <=
cap[m]`, `sum(alloc) <= budget`. Any wrong length/type, out-of-range entry,
over-budget total, a crash, timeout, or non-JSON output scores that instance `0.0`.

## Mechanics

Total posterior information for quantity `q`:

```
info_q = prior_precision[q] + sum_m coverage[m][q] * gain_a[m]*alloc[m]/(alloc[m]+gain_b[m])
```

Posterior variance `var_q = 1 / info_q`. Lower is better for every quantity, but you
are graded on `max_q var_q` — the single worst-estimated quantity, so budget spent
making an already-precise quantity even more precise buys nothing towards the score,
while a shared probe that nudges *several* still-imprecise quantities at once is worth
more than its single-quantity cost suggests.

## Objective and scoring

**Minimize** the mean, over 10 fixed instances, of a normalized worst-quantity
posterior variance. Per instance the evaluator computes, itself: `obj_base` (an even,
round-robin split of the budget across all measurement types — the naive reference)
and `obj_ref` (a strong internal water-filling + integer-repair + local-swap-polish
procedure — a high but not provably-optimal ceiling). Both are variances (lower is
better), so `obj_base > obj_ref`. Your candidate's worst-quantity variance `obj_cand`
is scored:

```
r = clamp(0.1 + 0.9 * (obj_base - obj_cand) / max(1e-9, 1.5*(obj_base - obj_ref)), 0, 1)
```

Matching the naive baseline scores ≈0.1; matching the internal reference scores ≈0.7;
there is headroom above that (the internal reference is a strong heuristic, not a
proven optimum). The reported **Ratio** is the mean `r`; **Vector** lists the 10
per-instance scores.

## Suggested strategies

1. Split the budget evenly across all measurement types (the reference).
2. For each quantity, in isolation, find its single steepest-initial-slope
   measurement and fund quantities in that fixed priority order.
3. Water-fill one integer unit at a time: after every unit, recompute whichever
   quantity is *currently* worst (real up-to-date posterior information, prior
   included) and fund whichever available measurement gives it the largest marginal
   gain, so a shared measurement's cross-coverage is naturally funded once, not
   double-booked.
4. Local search: from a water-filled allocation, hill-climb by moving single budget
   units between measurement-type pairs, keeping only moves that lower the worst-case
   variance.
