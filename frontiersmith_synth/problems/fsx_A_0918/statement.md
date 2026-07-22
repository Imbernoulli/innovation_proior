# Rounding a Fractional Task-Assignment Plan Along Its Hidden Cycles

`N` tasks must each be sent to exactly one of `M` **desks**. Desk `k` has an integer
`capacity[k]`. Sending task `i` to desk `k` consumes `weight[i][k]` units of that desk's
capacity and earns `value[i][k]` — both numbers depend on **which desk** the task lands on,
not just on the task. One desk (always the *last* index, `M-1`) is a generous, low-value
"overflow" option that is always affordable, so a feasible assignment always exists; the real
decision is how to spend the tightly-capacitated desks well. You decide, once, which desk
every task goes to, and are graded on the **total earned value**.

## Candidate program contract

Standalone program: read ONE JSON object (the public instance) from **stdin**, write ONE
JSON object (your answer) to **stdout**. It runs isolated and sees only the public instance.

```python
import sys, json
inst = json.load(sys.stdin)
# ... decide assign ...
print(json.dumps({"assign": assign}))
```

### Public instance (stdin)

```json
{
  "instance_id": "t3",
  "n_tasks": 6,
  "n_agents": 4,
  "capacity": [11.0, 11.0, 18.0, 500.0],
  "weight": [[11.0, 10.0, 8.0, 1.0], [10.0, 12.0, 7.0, 1.0], ...],
  "value":  [[19.0, 18.0, 3.0, 2.0], [18.0, 20.0, 2.0, 2.0], ...]
}
```

`capacity[k]`: desk `k`'s budget. `weight[i][k]`, `value[i][k]`: the cost and payoff of
sending task `i` to desk `k`. Every entry is given explicitly for every `(task, desk)` pair —
nothing is marked "ineligible"; a desk that is simply a bad fit for a task will just have a
low value and/or a weight that eats most of its remaining budget.

### Answer (stdout)

`assign`: a list of exactly `n_tasks` **integers**, `assign[i]` in `[0, n_agents-1]`, such
that for every desk `k`, the sum of `weight[i][k]` over tasks routed to `k` does not exceed
`capacity[k]`. Any wrong length/type, out-of-range entry, capacity violation, a crash,
timeout, or non-JSON output scores that instance `0.0`.

## Mechanics — why this is not plain knapsack

Because `weight[i][k]` depends on **both** the task and the desk, the natural LP relaxation
— `x[i][k] in [0,1]`, `sum_k x[i][k] = 1` per task, `sum_i weight[i][k]*x[i][k] <= capacity[k]`
per desk, maximize `sum v*x` — does **not** have an all-integer optimum in general: it can
genuinely split a task's unit of "presence" across two desks. Since every task-row is a
*mandatory-total-1 equality* while every desk-column is only a *capacity inequality*, the
graph of fractional edges (`0 < x[i][k] < 1`) decomposes into simple alternating **paths and
cycles** (task–desk–task–desk–…). Some instances below plant such a cycle: two (or more)
desks each look like the best choice for two competing tasks at once. Rounding each `x[i][k]`
independently, or greedily grabbing whichever desk currently looks best task-by-task, locks
in a locally-fine choice for one task that starves another of a desk it badly needed —
landing far below what resolving the whole cycle together achieves.

## Objective and scoring

**Maximize** the mean, over 10 fixed instances, of a normalized total value. Per instance the
evaluator computes, itself: `obj_base` (send every task to the overflow desk — value ignored)
and `obj_ref` (an internal LP-relaxation + cycle-pipage-rounding + value-improving-swap-repair
procedure — a strong but not provably-optimal ceiling), with `obj_ref > obj_base`. Your
candidate's total value `obj_cand` is scored:

```
r = clamp(0.1 + 0.9 * (obj_cand - obj_base) / max(1e-9, 1.3*(obj_ref - obj_base)), 0, 1)
```

Matching the naive baseline scores ≈0.1; matching the internal reference scores ≈0.79; there
is headroom above that (the internal reference is a strong heuristic, not a proven optimum).
The reported **Ratio** is the mean `r`; **Vector** lists the 10 per-instance scores.

## Suggested strategies

1. Send everything to the overflow desk (the reference floor).
2. Process tasks in order; for each, grab its highest-value desk that still has room.
3. Relax the assignment LP (`0<=x[i][k]<=1`). A fractional task generically has exactly two
   fractional desks. Peel a task whose fractional desk no one else uses — send it fully to
   its *other* fractional desk. What remains decomposes into cycles; each cycle has exactly
   two consistent full roundings (alternate directions around it) — try both, keep the one
   earning more value.
4. After rounding, a desk can slightly overshoot capacity. Repair by moving or swapping
   tasks between desks, preferring the least value lost, falling back to overflow last.
