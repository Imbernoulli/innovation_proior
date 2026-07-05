# Railway Freight Yard: Budget-Constrained Motive-Power Routing

You run the dispatch desk of a hump classification yard. Each morning a fixed daily
**trace** of freight **cuts** (blocks of cars) waits in the receiving yard; every cut
must be pulled over the hump and delivered before the cutoff. For each cut you may
assign exactly **one** motive-power option: a light yard switcher, a road-slug set, a
mainline unit, and so on. The options for a cut are listed in **increasing fuel order**.

Assigning cut `i` to option `j`:
- burns integer diesel `fuel[i][j]`, and
- delivers integer throughput `value[i][j]` (ton-miles moved before the cutoff).

Heavier power moves more tonnage, but with **diminishing returns**, and how steeply
value grows with fuel differs from cut to cut. You have a fixed daily diesel **budget**
`B`. Choose one option per cut, offline over the whole trace, to **maximise total
delivered value** subject to **total fuel used ≤ B**.

This is a multiple-choice knapsack over a fixed trace. Assigning every cut its cheapest
switcher (option 0) is always affordable but delivers little; a greedy that upgrades each
cut straight to its single best option ignores the diminishing-returns structure and
wastes budget on inefficient heavy units; a policy that weighs intermediate options
across the whole trace (e.g. Lagrangian relaxation on a fuel price, or a budget DP) does
markedly better. There is no easy optimum and several viable strategies.

## Program contract

Your program is a standalone process. Read ONE JSON **public instance** from stdin and
write ONE JSON **answer** to stdout:

```python
import sys, json
inst = json.load(sys.stdin)
# ...compute...
print(json.dumps(answer))
```

### Public instance schema
```json
{
  "n_cuts":   N,                      // number of freight cuts in the trace
  "n_options": M,                     // motive-power options per cut
  "budget":   B,                      // integer daily diesel budget
  "fuel":     [[f_00,...,f_0(M-1)], ...],   // N x M, strictly increasing per row
  "value":    [[v_00,...,v_0(M-1)], ...]    // N x M, integer delivered value
}
```

### Answer schema
```json
{ "assign": [j_0, j_1, ..., j_(N-1)] }   // option index chosen for each cut, each in [0, M-1]
```

## Objective and scoring

For a **feasible** assignment (`sum_i fuel[i][assign[i]] <= budget`) the objective is the
total delivered value `obj = sum_i value[i][assign[i]]` (higher is better).

Let `b` be the objective of the all-cheapest policy (every cut on option 0), which the
evaluator computes itself. Your normalized score on an instance is

```
r = min(1, 0.1 * obj / b)
```

so the all-cheapest policy scores exactly `0.1`. An assignment that is over budget, has
the wrong length, or contains an out-of-range / non-integer option index is **infeasible
and scores 0**. The final score is the mean of `r` over all instances (a fixed set of
12 traces, including four larger held-out ones). Scoring is fully deterministic.
