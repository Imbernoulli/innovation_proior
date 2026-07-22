# Job Router for a Shop of Aging Machines

## Story

A small job shop runs `M` machines. Jobs arrive one at a time, in a fixed
stream, in two kinds: **abrasive** (`"A"`, rough material removal) and
**finishing** (`"F"`, precision polishing). Each machine has its own fixed
**abrasion sensitivity** `a` and **polish sensitivity** `q`, plus a current
**speed** state starting at `1.0`. An abrasive job multiplies that machine's
speed by `(1 - a)` (it dulls); a finishing job multiplies it by `(1 + q)`
(it self-sharpens). Speed is clamped to `[0.15, 6.0]`. A machine that dulls
fast on abrasive jobs is often also one that self-sharpens fast on finishing
jobs -- a bad choice for one job type, great for the other.

You must assign each job to a machine **the moment it arrives, and this
choice is final** -- no reassignment, no peeking at future jobs, no undo.
Each machine processes the jobs assigned to it strictly in arrival order
(FCFS); all machines run in parallel, independently. **Assigning a job is an
investment in that machine's future condition, not just a queueing
decision.**

## You write a program, called once per job

Your program is standalone, invoked **fresh** (no memory between calls) once
per job. It reads **one** JSON object from **stdin** and writes **one** JSON
object to **stdout**.

### Input (stdin)

```json
{
  "step": 4, "n_jobs": 22, "n_machines": 2,
  "job": {"type": "A", "size": 3.7, "weight": 1.4},
  "seen_so_far": {"A": 3, "F": 1},
  "machines": [
    {"id": 0, "a": 0.05, "q": 0.05, "spd": 0.81, "free_at": 11.2},
    {"id": 1, "a": 0.24, "q": 0.24, "spd": 1.49, "free_at": 9.6}
  ]
}
```

- `job`: the job to place now. `size` is its base work amount; `weight` its
  importance.
- `seen_so_far`: counts of past job types, so a stateless program can guess
  the stream's composition.
- `machines`: each machine's fixed `a`/`q` sensitivities and its CURRENT
  `spd` (speed multiplier) and `free_at` (when it becomes free).

### Output (stdout)

```json
{ "assign": 1 }
```

An integer machine id in `[0, n_machines)` (a JSON number that is
integer-valued, such as `1` or `1.0`, is accepted either way). A missing/
out-of-range/non-numeric/fractional id, a crash, a timeout, or non-JSON
output at **any** job in the stream voids the **entire instance** (score
`0.0` for it).

## Objective & scoring (deterministic)

You **minimize** total weighted completion time: `sum(weight * completion_time)`
over the stream, from the REAL speed trajectory your assignments induce (a
job on a machine with `free_at = F` and speed `spd` takes `size / spd`,
completes at `F + size / spd`; the machine's speed then updates per the
job's type before its next job).

The evaluator normalizes each instance with an affine anchor:

- `weak` = what a fixed **round-robin over machines** recipe achieves
  (simulated directly by the evaluator).
- `lb` = a **lower bound**: replace every job's duration with its best-case
  value (`size / 6.0`, the fastest any machine could ever run) and pool the
  `M` machines into one preemptive machine of combined capacity `M` (no real
  machine can borrow an idle sibling's capacity), solved exactly by Smith's
  ratio rule. `lb` never exceeds anything achievable, and is usually
  unreachable in practice, leaving headroom above a strong policy.

```
r = clamp( 0.1 + 0.9 * (weak - your_total) / max(weak - lb, 1e-6), 0, 1 )
```

Reproducing round-robin scores ~0.1; doing worse scores below 0.1; every
unit of weighted completion time you shave off moves you toward 1.0. Your
final score is the mean of `r` over a fixed family of 10 job streams -- some
skew heavily toward cheap abrasive filler punctuated by rare high-value
finishing jobs, some alternate in short bursts, and one uses two identical
machines where no specialization is possible at all.

## Notes

- Fully deterministic and seeded: the objective is a pure function of your
  decisions, never wall-clock/hardware timing (a per-job time limit is only
  a safety valve against a hung program). Your program sees only the
  current job's public state, in an isolated sandboxed subprocess; the
  future stream, baseline, and bound live only in the evaluator.
- Reacting only to current queue length (shortest-queue dispatch) spreads
  wear evenly across every machine instead of concentrating it -- no machine
  ever becomes genuinely fast, and cheap filler dulls machines that could
  otherwise have been protected for valuable work.
