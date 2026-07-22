# Ridgemont ER: Pre-Escalation Bay Triage

## Setting

Ridgemont ER has **N treatment bays** and receives **P patients** over a
**T-step shift**. Each patient `p` arrives at step `arrival[p]` with initial
acuity `L0[p]` and a personal **escalation rate** `rate[p]`: every step a
patient spends *waiting* (arrived, not currently being treated), their
acuity rises by `rate[p]`. The instant a bay starts treating a patient,
their acuity **freezes** for as long as that treatment continues. You
control, every step, which patient (if any) each bay is working on.

## The two costs you must manage

**Deadline-acuity-escalation.** A patient's outcome depends on their acuity
at the moment treatment *starts* (`start_acuity`), against the mid
threshold `C`:
- `start_acuity <= C`: full outcome, multiplier **1.0**, and treatment
  takes only `base_need` steps.
- `C < start_acuity <= D`: a reactive rescue, multiplier **0.35**, and
  treatment takes `base_need + slope * (start_acuity - C)` steps (rounded
  up) — reacting late costs a bay strictly *more* capacity, not less.
- If a *waiting* patient's acuity ever exceeds the death cap `D`, they die:
  reward **-weight[p]**, no rescue possible.
- A patient still un-resolved when the shift ends scores **0** (neutral).

**Preemption-switching-cost.** Reassigning a bay to a *different* patient
while it is mid-treatment is a preemption: the abandoned patient's progress
resets to zero and they resume waiting (and escalating); the bay is then
unusable for `S` steps of setup; and the evaluator subtracts
`switch_penalty` from the total score for that event. Freeing a bay to idle
(not switching to anyone) costs nothing extra but still drops the abandoned
patient's progress.

Total score: `obj = sum(patient rewards) - switch_penalty * (#preemptions)`.

## Input (stdin): ONE JSON object — the public instance

```json
{
  "name": "trap_duel_sleepers", "T": 60, "N": 2, "P": 11,
  "arrival": [0, 1, 2, 3, 6, ...],
  "L0":      [4.9, 5.3, 4.6, 5.1, 8.1, ...],
  "rate":    [1.35, 1.5, 1.4, 1.28, 0.83, ...],
  "weight":  [4, 3, 5, 3, 5, ...],
  "C": 10.0, "D": 17.0, "base_need": 3, "slope": 0.8,
  "S": 3, "switch_penalty": 1.3
}
```
All arrays have length `P`, index-aligned to patient `p = 0..P-1`.

## Output (stdout): ONE JSON object

```json
{ "assign": [[b_0, b_1, ..., b_{N-1}], ...] }
```
`assign` has exactly `T` rows, each of exactly `N` integers. `assign[t][b]`
is the **1-indexed patient id** (`p+1`) requested for bay `b` at step `t`,
or `0` for no request. Wrong shape/length, a non-integer or out-of-range
entry, non-finite output, a crash, a timeout, or non-JSON output scores
that instance **0.0**. Requesting a patient who hasn't arrived yet, is
already resolved (done/dead), or is being treated at another bay this step
is **not** a hard failure — it is simply ignored (treated as idle) for that
bay that step.

## Simulation (what the evaluator replays)

Each step, in order: (1) patients with `arrival[p] == t` become waiting at
`L0[p]`; (2) every already-waiting patient's acuity increases by `rate[p]`
(checked against `D` immediately — crossing it is death); (3) bays are
processed in index order `0..N-1`: continuing the same patient advances
progress by one step (the patient resolves once progress reaches
`need(start_acuity)`, earning `weight * multiplier(start_acuity)`);
requesting a new patient into an **idle** bay starts a fresh attempt;
requesting a **different** patient into a mid-treatment bay preempts it
exactly as above (the new request itself is wasted that step).

## Scoring

The evaluator computes a loose, deliberately-unreachable upper bound
`q_ideal = sum(weight)` (every patient resolved at multiplier 1.0 with zero
preemptions — never actually achievable given the seeded arrival bursts and
bay/time scarcity) and reports, per instance:

```
r = clamp( 0.1 + 0.9 * obj / q_ideal, 0, 1 )
```

Doing nothing scores exactly `0.1`. Net deaths push `obj` negative and `r`
toward (and clipped at) `0`. The final `Ratio` is the mean `r` over 10
seeded instances (some deliberately larger / held-out for generalization).
There is no clean optimum: bays, time, and switching are all finite, and
several instances are built so that a policy which only reacts to
*currently*-worst patients keeps thrashing bays while other patients
silently escalate — reading each patient's own deterministic curve to
schedule bays *before* they tip is what separates a strong policy from a
merely reactive one.
