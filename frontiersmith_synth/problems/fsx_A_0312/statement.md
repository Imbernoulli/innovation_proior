# Belt Rush: Hauler Stowage for a Mining Shift

## Story

A robotic rig cracks an asteroid. Over a single mining shift the rig spits out a
stream of ore **fragments**, each with an integer **mass** (arbitrary units), in the
order they are ejected. Every cargo **hauler** can carry at most **capacity** `C`
units of mass. You must stow **every** fragment into some hauler without ever
exceeding `C`, using as **few haulers** as possible — each dispatched hauler burns
fuel, so fewer is better.

This is 1-D bin packing skinned as an asteroid-mining stowage contest.

The rig ships with a dumb on-board autoloader (the **fixed streaming simulator**): it
keeps one hauler docked, drops each arriving fragment into it, and the instant a
fragment does not fit it seals that hauler, dispatches it, and docks a fresh one
(the classic **Next-Fit** online rule). Because it can never back-fill a sealed
hauler it wastes capacity — your job is to beat it with a smarter stowage plan.

## You write a program

Your program is a standalone process. It reads **one** JSON object (the public
instance) from **stdin** and writes **one** JSON object (your plan) to **stdout**.

### Input (stdin)

```json
{
  "name": "shift101",
  "capacity": 100,
  "masses": [37, 12, 55, 8, 44, ...]
}
```

- `capacity`: integer `C > 0`, the mass each hauler can carry.
- `masses`: list of `n` integers, each `1 <= m_i <= C`, in **arrival order**.

### Output (stdout)

```json
{ "assign": [0, 0, 1, 0, 2, ...] }
```

- `assign` must have **exactly `n`** integer entries.
- `assign[i]` is the index of the hauler that fragment `i` is stowed in.
- Each index must satisfy `0 <= assign[i] < n` (you may open at most `n` haulers;
  indices need not be contiguous — only the **count of distinct haulers used** is
  charged).

## Validity

A plan is **valid** iff `assign` is a list of exactly `n` integers, every entry is a
non-boolean integer in `[0, n)`, and for **every** hauler the total mass of the
fragments assigned to it is `<= C`. A wrong length, an out-of-range index, an
over-capacity hauler, a non-integer entry, a crash, a timeout, or non-JSON output
makes that instance score **0.0**.

## Objective & scoring (deterministic)

You **minimize** the number of distinct haulers used. Per instance the evaluator
computes:

- `nf` = haulers used by the fixed **Next-Fit** streaming autoloader (weak baseline).
- `lb` = `ceil(sum(masses) / C)`, the **L1 lower bound** — no feasible plan can use
  fewer haulers than this, and it is usually **not achievable** because of packing
  waste, which leaves headroom so even a strong stower stays below 1.0.
- `cnd` = distinct haulers your valid plan uses (always `cnd >= lb`).

The normalized per-instance score, with an affine anchor (reproduce Next-Fit → 0.1,
reach the L1 bound → 1.0):

```
r = clamp( 0.1 + 0.9 * (nf - cnd) / max(nf - lb, 1),  0, 1 )
```

Reproducing the streaming autoloader scores ~0.1; using more haulers scores below
0.1; every hauler you save versus Next-Fit moves you toward 1.0, capped there. Your
final score is the mean of `r` over a fixed family of 12 shifts (uniform, skewed, and
Weibull-like mass distributions, plus larger held-out shifts).

## Notes

- Scoring is fully deterministic and seeded — no wall-clock or hardware timing.
- Your program only ever sees the public instance and runs in an isolated sandboxed
  subprocess; the baseline, the lower bound, and all validation happen in the
  evaluator process.
- Bin packing is NP-hard: there is no easy optimum, and First-Fit, Best-Fit,
  decreasing-order variants, and local search are all viable strategies with
  different trade-offs.
