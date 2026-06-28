# Soft-Constraint Assignment

## Research question

You must place `n` **agents** into `m` **slots**. Slot `j` has a fixed **capacity** `cap_j` (it can hold
at most `cap_j` agents), and the capacities are generous enough that *some* legal placement always
exists (`Σ_j cap_j ≥ n`). Putting agent `i` in slot `j` earns an integer **preference** `pref[i][j] ≥ 0`
(higher is better). On top of the preferences there is a list of `C` **soft constraints**, each a pair of
agents that *prefer* a relationship with a penalty for breaking it:

- a **DIFFER** constraint `(a, b, w)` charges penalty `w` if `a` and `b` end up in the **same** slot;
- a **SAME** constraint `(a, b, w)` charges penalty `w` if `a` and `b` end up in **different** slots.

The task is to choose a slot for every agent so as to **maximize**

```
P = Σ_i pref[i][assign[i]]  −  Σ_{violated constraints} w,
```

subject to never exceeding any slot's capacity. This is a capacitated assignment with soft side
constraints — a generalized assignment problem whose constraint graph makes it NP-hard, with no known
efficient exact solution at this scale. The benchmark scores a placement by *how much objective it
earns* rather than by matching a unique optimum.

## Input / output contract

- **Input (stdin):**
  - the first line holds two integers `n m` (`200 ≤ n ≤ 600`, `8 ≤ m ≤ 40`);
  - the second line holds `m` integers `cap_0 … cap_{m−1}` (`cap_j ≥ 1`, `Σ cap_j ≥ n` guaranteed);
  - then `n` lines, line `i` (0-indexed) holding `m` integers `pref[i][0] … pref[i][m−1]`
    (`0 ≤ pref[i][j] ≤ 1000`);
  - then a line with one integer `C` (the number of soft constraints);
  - then `C` lines, each `t a b w` with `t ∈ {0, 1}` (`0` = DIFFER, `1` = SAME), `0 ≤ a, b < n`,
    `a ≠ b`, and `w ≥ 1` an integer penalty weight.
- **Output (stdout):**
  - exactly `n` integers `s_0 … s_{n−1}` (whitespace-separated), where `s_i ∈ {0, …, m−1}` is the slot
    chosen for agent `i`.
- **Time limit:** about 2 seconds wall-clock per instance. **Memory:** 256 MB.

A solution is **feasible** iff the output is exactly `n` integer tokens, each in `{0, …, m−1}`, **and**
no slot receives more than its capacity (`#{i : s_i = j} ≤ cap_j` for every `j`). Anything else — a wrong
token count, an out-of-range slot, a non-integer token, **a capacity overflow**, a missing file — is
**infeasible**.

## Background

Stripped of the surface story, the structure is: choose a function `assign : agents → slots` respecting
capacities to maximize a sum of per-agent rewards minus a sum of pairwise penalties on a constraint
graph. Two forces pull against each other — each agent has a slot it *prefers* (its high-preference
column), but the constraints couple agents, so satisfying a constraint may require pulling an agent away
from its favourite slot. Several approaches sit on the table before committing:

- **Greedy best-preference.** Assign every agent to its single highest-preference slot, ignoring the
  constraints, then repair any capacity overflow by bumping the least-loss agents to their next-best
  feasible slot. Always feasible and fast. But it is *constraint-blind*: it pays the full penalty of
  every violated DIFFER/SAME pair, leaving objective on the table exactly where the constraints bite.
  (This is the scorer's reference baseline.)
- **Constraints as hard rules + matching.** Treat the soft constraints as hard and solve a constrained
  matching. The trap: the constraints are *soft* — sometimes paying a small penalty is cheaper than the
  preference loss of satisfying it — so forcing them hard both over-constrains the problem and can make
  it infeasible against the capacities.
- **Decompose: pick slots, then fix violations.** First optimize preferences, then in a second phase
  flip agents to cut violations. Better, but the two phases fight: cutting a violation moves an agent and
  *re-pays* preference, and the best preference assignment depends on which violations you intend to
  tolerate — alternation stalls in a poor joint local optimum.
- **Relaxation rounding + a fused local search.** Start from the constraint-blind relaxation rounding
  (the greedy best-preference assignment above — a strong anchor), then run **one** local search whose
  moves change the slot assignment directly, so preference and penalties co-evolve. This is the established
  strong approach, and the engineering lever is a **cheap incremental objective delta** per move:
  preference changes in `O(1)`, and — the key — the penalty change is computed by scanning **only the soft
  constraints incident to the moved agent(s)** via a precomputed **constraint→agent incidence list**, so a
  move costs `O(deg)` instead of `O(C)`. The non-obvious move beyond a plain relocate is the **two-agent
  swap**: swapping the slots of two agents is *always* capacity-feasible (loads are preserved), and it
  reaches improving states that no single relocate can when both donor slots are full.

The decisive accelerators are the **incidence lists** (so each move's penalty delta is a tiny local sum)
and the **capacity-aware move set** (relocate only into a slot with spare capacity; swap preserves loads),
so the search never has to recheck global feasibility.

## Evaluation settings

- **Instances.** A generator (`verify/gen.py`, parametrized by an integer `seed`) deterministically chooses
  `n ∈ [200, 600]`, `m ∈ [8, 40]`, capacities summing to `1.1–1.6 ×` of `n` (always `≥ n`, so a feasible
  placement exists but slots are tight), a preference matrix where each agent has one "favourite" slot that
  spikes high over a noisy background, and `C ≈ 0.8 n … 2.0 n` soft constraints (a mix of DIFFER and SAME)
  whose penalty weights are set on the order of a *typical* preference value. That scaling is the
  load-bearing design choice: one violated constraint costs about as much as one agent's favourite slot, so
  the constraint-blind preference optimum is provably *not* optimal — a constraint-aware search must trade a
  little preference for fewer violations.
- **Scoring rule (deterministic, `verify/score.py`).** Read the instance and the submitted solution.
  - **Feasibility floor:** if the output is not exactly `n` integers each in `[0, m)`, **or any slot's
    capacity is exceeded**, the score is **`0`**.
  - Otherwise let `P` be the submitted solution's **objective** (`Σ chosen preferences − Σ violated
    penalties`). Let `P_base` be the objective of the scorer's own deterministic **greedy best-preference**
    baseline (assign each agent to its best slot ignoring constraints, then repair overflow by evicting the
    least-loss agents to their best still-feasible slot — recomputed inside the scorer, independent of the
    solver). Let `Scale = Σ_i max_j pref[i][j]` be the maximum attainable raw preference, a positive
    instance-scale normalizer (floored at `1`). The score is

    ```
    score = round( 1 000 000 + 1 000 000 × (P − P_base) / Scale )     (feasible), clamped to ≥ 0
    score = 0                                                          (infeasible)
    ```

    A higher score is better. The greedy best-preference baseline scores exactly `1 000 000`; a
    constraint-aware solution that improves the objective scores strictly more; a worse one scores less but
    never below `0`. The scorer recomputes `P_base`, `Scale`, and every penalty itself, so the reference is
    reproducible and independent of the solver.
- **Reported metric.** The mean score over a fixed seed set. A genuine relaxation-rounding-plus-local-search
  solver lands above `1 000 000` (≈ `1.03–1.13 ×` on these instances); the constraint-blind baseline is the
  `1 000 000` floor to beat, and any infeasible output scores `0`.

## Code framework

A single self-contained C++17 program reading the instance from stdin and writing a feasible solution to
stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    vector<int> cap(m);
    for (int j = 0; j < m; j++) scanf("%d", &cap[j]);
    vector<vector<int>> pref(n, vector<int>(m));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++) scanf("%d", &pref[i][j]);
    int C; scanf("%d", &C);
    vector<int> ct(C), ca(C), cb(C), cw(C);
    for (int c = 0; c < C; c++) scanf("%d %d %d %d", &ct[c], &ca[c], &cb[c], &cw[c]);

    // A feasible answer is ANY capacity-respecting slot per agent. Build the
    // greedy best-preference assignment (then repair overflow) as a safety net.
    vector<int> assign(n, 0);

    // TODO heuristic: relaxation rounding (best-preference + capacity repair) for the
    // start; build constraint->agent incidence lists; then run ONE local search over
    // RELOCATE (agent -> a slot with spare capacity) and SWAP (exchange two agents'
    // slots, loads preserved), each move scored by an O(1) preference delta plus an
    // O(deg) penalty delta over only the incident constraints, accepted by a
    // simulated-annealing rule under a ~1.9s budget. Keep a valid assignment at all
    // times and print the best feasible one seen.

    for (int i = 0; i < n; i++) printf("%d%c", assign[i], i + 1 < n ? ' ' : '\n');
    return 0;
}
```
