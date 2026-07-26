# One Purchase, Nine Mixes: Hedging a Workshop's Machine Portfolio

## Story

A workshop buys its machine fleet **once**, before it knows which job mix
the shop floor will run. There are 5 operation kinds (0..4) and 6 machine
*types*: types 0..4 are single-kind specialists (fast at exactly one kind,
slow at the rest), type 5 is a generalist (mediocre but usable at every
kind). Once bought, the fleet is fixed. The shop floor then runs one of 9
job mixes, and a fixed, none-too-clever scheduler processes it. You never
rebuy or re-plan — only pick a fleet that survives most of the mixes.

## Input (stdin)

```
T KINDS BUDGET
cost_0[0] cost_0[1] ... cost_0[KINDS-1] price_0
...
cost_{T-1}[0] ... cost_{T-1}[KINDS-1] price_{T-1}
K
n_jobs_1 oracle_1
L k_0 w_0 k_1 w_1 ... k_{L-1} w_{L-1}      (one line per job of scenario 1)
...
n_jobs_2 oracle_2
...
```
`T=6` machine types, `KINDS=5` operation kinds, integer `BUDGET`. Each type
line gives `cost_t[k]` (time per unit of work if kind `k` runs on a type-`t`
machine — smaller is faster) and `price_t` (cost to buy one unit). Then
`K=9` scenarios, each listing its jobs; job `j` is a **chain** of `L`
operations run in order, each with kind `k` and work `w` (duration on a
type-`t` machine is `w * cost_t[k]`). `oracle` is a shipped lower bound: the
largest total work-sum along any one job's chain — no schedule beats it,
regardless of spend.

## Output (stdout)

Print `T` non-negative integers: `n_0 n_1 ... n_{T-1}`, the number of
machines of each type you buy. Must satisfy `sum(n_t * price_t) <= BUDGET`
and `sum(n_t) >= 1`. Any wrong count, negative, non-integer, non-finite, or
over-budget output scores `0`.

## Scheduling (fixed, not yours to choose)

Your purchased machines are indexed round-robin across types in catalog
order (one machine from each type you bought at least 1 of, then a second
machine from each type you bought at least 2 of, and so on) — this keeps
tie-breaks from silently favoring whichever type you bought the most of.
For each scenario independently, the shop runs this frozen list scheduler:
repeatedly take the machine that becomes free earliest (ties -> lowest
machine index); among jobs whose next operation is already ready
(predecessor finished, or it's the job's first op) at that time, take the
**lowest-indexed** ready job and run its next operation on that machine.
The scenario's **makespan** is the time the last job finishes.

## Objective & scoring (minimize)

For each scenario compute `regret = makespan / oracle` (always >= 1). Sort
the 9 regrets from worst (highest) to best; your score is driven by the
**3rd-worst** value — you may silently sacrifice your two worst scenarios,
but not a third. Call this quantity `F` (lower is better). The checker also
computes `B`, the same quantity for a trivial reference fleet (the whole
budget spent on the single cheapest machine type). Final ratio:
`min(1000, 100*B/F) / 1000`, clamped to `[0,1]`.

## Notes

- Every machine type can process every kind, just slowly off its specialty
  — feasibility never fails because a kind is "impossible", only because you
  overspent, under-bought, or malformed the output.
- Buying only the specialist matching the scenarios' *average* demand is a
  trap: real scenarios don't overlap, and since only 2 of the 9 can be
  sacrificed, concentrating on one kind leaves several other kinds' spikes
  uncovered — each lands in the counted quantile. A fleet that hedges (a
  couple of specialists plus a few "wasteful" generalist units the scheduler
  can always fall back on) survives far more mixes.

## Worked example (illustrative shape only — not the shipped catalog)

Suppose 2 kinds, 2 types: type0 cost=[1,8] price=10; type1 cost=[8,1]
price=10; budget=20 (buy one of each). One scenario: job A = [(kind0, w=4)],
job B = [(kind1, w=4)]; oracle = 4. The scheduler runs A on type0 (4*1=4)
and B on type1 (4*1=4) in parallel — makespan=4, regret=1.0. Spending the
whole budget on two type0 units instead serializes B at cost 8 — regret=8.0,
far worse. Diversifying beat concentrating here.

In the shipped catalog, off-specialty cost is 14 and a generalist costs 2 at
every kind (price 48 vs 30) — a few generalist units give the scheduler a
decent fallback for under-provisioned kinds instead of dumping overflow onto
a badly-mismatched specialist at cost 14.
