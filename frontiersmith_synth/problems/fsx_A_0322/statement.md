# Ironhump Freight Yard: Single-Hump Shunting Priority

## Setting

The Ironhump classification yard runs its whole shift on a **single hump** (one shunting
engine). Over the shift, **freight cuts** — coupled groups of cars bound for the same
outbound train — roll onto the receiving lead and wait to be pushed over the hump and
sorted. You are the yardmaster, and you decide the **dispatch priority** the hump follows.

Each cut `i` has:

- `p` — **hump time**: how long the engine needs to push and sort the cut (`p >= 1`);
- `w` — **priority weight**: penalty per time unit its outbound train is kept waiting (`w >= 1`);
- `r` — **release time**: the cut cannot be humped before it has arrived (`r >= 0`);
- `d` — **cut-off**: the scheduled departure slot of its outbound train (the due date).

The hump works **one cut at a time** and never interrupts a cut once started
(non-preemptive). If cut `i` finishes at time `C_i`, it incurs a lateness penalty
`w_i * max(0, C_i - d_i)`. Your goal is to minimize the **total weighted lateness**

```
    sum_i  w_i * max(0, C_i - d_i).
```

This is the strongly NP-hard single-machine weighted-tardiness problem with release
dates, `1 | r_j | sum w_j T_j`.

## What you submit

You do **not** submit a full timetable. You submit a **dispatch priority order**: a
permutation of the cut indices, highest priority first. A **fixed, non-delay simulator**
turns your priority into an actual schedule:

- whenever the hump is idle and at least one cut has **already arrived** (`r <= t`), it
  immediately starts the arrived cut that stands **earliest in your priority order**;
- if no cut has arrived yet, the hump fast-forwards to the next arrival;
- it never leaves the hump idle while an arrived cut is waiting.

So your ranking is only consulted to break "who goes next" among the cuts already on the
lead — future arrivals cannot be pre-empted or waited for.

## Program contract (stdin → stdout)

Your program reads **one** JSON object (the public instance) from stdin and writes
**one** JSON object to stdout.

**Input (stdin):**

```json
{
  "name": "yard201",
  "n": 14,
  "horizon": 173,
  "cuts": [
    {"p": 5, "w": 3, "r": 2, "d": 11},
    {"p": 2, "w": 5, "r": 2, "d": 9}
  ]
}
```

- `n` — number of cuts; `cuts` has length `n` (index `i` = cut `i`).
- `horizon` — an informational upper bound (last release + total hump time).

**Output (stdout):**

```json
{"order": [1, 0, 5, 3, 2, 4, ...]}
```

`order` must be a **permutation of `0 .. n-1`**: exactly the `n` cut indices, each once.
`order[0]` is dispatched first among available cuts, and so on.

Anything that is not a valid permutation — wrong length, a repeat, an out-of-range or
non-integer entry, a crash, a timeout, or non-JSON output — scores **0.0** on that
instance.

## Scoring

Each instance is scored by the simulator's total weighted lateness `q_cand` for your
order, normalized against the **first-come-first-served** reference `q_fcfs` (dispatch
strictly by arrival time):

```
    r = min( 1.0, 0.1 * q_fcfs / max(q_cand, 1e-9) )
```

- Reproducing FCFS scores exactly **0.1**.
- Halving FCFS's total lateness scores about **0.2**; only cutting it to ≤10% of FCFS
  approaches **1.0**.
- Doing **worse** than FCFS scores **below 0.1**.

The instances are congested with tight cut-offs — no zero-lateness schedule exists — and
the problem is NP-hard, so even a strong local search stays well below `1.0`. Your final
score is the **mean of `r` over all 14 instances** (a mix of shift sizes, weight ranges,
arrival spacings and cut-off tightness, including a few larger held-out yards).

## Ideas

- **First-come-first-served** (the reference) ignores weights and cut-offs — the naive baseline.
- **Static dispatch rules**: weighted-shortest-processing-time (Smith's ratio `w/p`) or
  earliest-due-date (EDD) sharply beat FCFS.
- **Look-ahead rules**: Apparent-Tardiness-Cost (ATC) blends `w/p` with a slack discount.
- **Seeded local search / metaheuristics**: start from the best rule and relocate or swap
  cuts (insertion / 2-opt, perturb-and-restart), scoring candidates on the same non-delay
  simulator, to reach a strong local optimum.
