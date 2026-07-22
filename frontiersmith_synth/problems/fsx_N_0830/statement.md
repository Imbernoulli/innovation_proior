# The Calibrated Recall

You run a hiring desk that processes a queue of **N candidates**, arriving one
at a time in a fixed order. Each candidate `i` shows a single public
**interview score** `score[i]` in `[0, 1]`. For every candidate, immediately
upon arrival, you must choose one of two actions:

- **Accept** (`1`): hire them right now. This is **irreversible** and
  consumes one of your **K** total hiring slots. Their realized contribution
  is `v[i]`, computed by the formula below (never shown to you numerically).
- **Pass** (`0`): do not hire them now. They are not gone forever — they
  enter a **recall pool** for up to **W** further arrivals. At any later
  round `r` with `r - i <= W` you may still **recall** them (spend a slot to
  hire them retroactively), but their value decays: recalling at round `r`
  yields `v[i] * decay^(r-i)` instead of the full `v[i]`. Once `r - i > W`,
  they leave the pool forever and can never be hired.

## The hidden value rule (two possible regimes)

Every candidate's TRUE hiring value follows **one of two possible formulas**,
fixed for the whole instance but never stated directly:
```
Rising:  v_base(o) = 0.15 + 1.20*o      (higher interview score -> higher value)
Fading:  v_base(o) = 1.35 - 1.20*o      (higher interview score -> LOWER value)
```
Both regimes are equally likely a priori across instances. On top of
`v_base`, every candidate's value is scaled by a **known** per-instance drift
factor that rewards patience:
```
v[i] = max(0, v_base(score[i]) * (1 + drift * i/(N-1)) + noise[i])
```
`drift` is given in the input; `noise[i]` is small, unobservable,
per-candidate noise. **The trap**: a policy that treats the raw score as
value, assuming "higher is better," is badly fooled on every
**Fading**-regime instance, and also has no reason to save slots for the
richer, high-drift tail. The regime itself is never stated: you must infer
it from the *shape of the score stream* (Rising instances cluster scores
high; Fading instances cluster low) — spend a little of the early stream
learning which regime governs this run, then apply a threshold calibrated to
it for the rest, using the recall window as a backstop for anything passed
on while still uncertain.

## Candidate program contract
Standalone program: read one JSON object from stdin, write one JSON object
to stdout.
```python
import sys, json
inst = json.load(sys.stdin)
# ... decide ...
print(json.dumps({"actions": actions, "recalls": recalls}))
```
### Public instance (stdin)
```json
{"name": "desk7", "N": 160, "K": 14, "W": 6, "decay": 0.9, "drift": 0.35,
 "score": [0.41, 0.77, 0.12, ...]}
```
### Answer (stdout)
```json
{"actions": [0, 1, 0, ...],           // length N, each 0 (pass) or 1 (accept now)
 "recalls": [[r0, j0], [r1, j1], ...] // recall candidate j at round r
}
```
Each recall `[r, j]` must satisfy: `0 <= j < r < N`, `r - j <= W`,
`actions[j] == 0`, and each `j` used in at most one recall. Total hires
(`sum(actions) + len(recalls)`) must not exceed `K`. Any malformed output
(wrong shape/types, an invalid recall, exceeding `K` hires, a crash, a
timeout, or non-JSON) scores that instance `0.0`.

## Objective and scoring
**Maximize** total realized value: the sum of `v[i]` over immediate accepts
plus `v[j] * decay^(r-j)` over valid recalls, across a fixed, seeded family
of 10 instances (some Rising, several engineered as Fading traps, varying
`N`/`K`/`W`/`drift`). The evaluator replays your answer against the true,
hidden `v` array and normalizes:
```
r = clamp(0.1 + 0.9 * total_value / UB, 0, 1)
```
`UB` is a generous, loosely-reachable per-instance ceiling, so hiring nobody
scores `0.1` and real headroom remains above any reference solution.
**Ratio** is the mean of `r`; **Vector** lists the per-instance scores.

## Suggested strategies
1. Reject everyone (baseline).
2. Classic secretary sample-then-threshold rule applied directly to the raw
   interview score; never uses the recall window.
3. Infer the active regime from the early score stream's shape, convert
   scores to drift-adjusted estimated values under that regime, and
   dynamically retarget the acceptance bar against the remaining pool's
   estimated richness — recalling early passes that turn out promising.
4. Exact optimization over the estimated-value array respecting the
   round/slot/recall-window constraints, for a provably near-optimal policy
   under the inferred regime.
