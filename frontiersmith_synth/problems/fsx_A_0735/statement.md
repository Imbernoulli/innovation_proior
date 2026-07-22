# Roots on the Gamble: Carbon Allocation over Hidden Nutrient Patches

## Story

A young plant grows `K` root **tips** starting at the soil surface (depth 0).
Each tip follows its own fixed vertical shaft down to a maximum depth `L`.
The soil along every shaft has a hidden, **patchy** nutrient concentration
profile: mostly a low, noisy background, with a few localized rich patches
scattered at various depths. Over `T` developmental steps the plant has a
fixed **carbon budget** `B` to spend each step, split however it likes across
its currently active tips. Spending `c` carbon on a tip extends it by `c`
depth units and harvests the nutrient integrated over the newly grown
stretch, minus a fixed **construction cost** per unit length spent.

Crucially, a tip can only ever **sense the concentration at its own current
position** — never deeper on itself, never on any other tip. You must invest
carbon to reveal what lies ahead. Because the best-looking patch right now is
only known *locally*, always committing everything to today's best tip can
permanently strand you on a small, shallow patch while a far larger one sits
undiscovered deeper on a tip that currently looks unpromising.

## You write a program, called once per developmental step

Your program is a standalone process, invoked **fresh** (no memory between
calls) once per step. It reads **one** JSON object from **stdin** and writes
**one** JSON object to **stdout**.

### Input (stdin)

```json
{
  "step": 0, "steps_total": 9, "budget_step": 9.0, "ccost": 1.0, "L": 40,
  "n_tips": 4,
  "tips": [ {"id": 0, "pos": 0.0, "sensed": 0.83}, {"id": 1, "pos": 0.0, "sensed": 1.02}, ... ],
  "history": [ {"sensed": {"0": 0.83}, "alloc": {"0": 4.5, "1": 4.5}}, ... ]
}
```

- `tips`: only the currently **active** tips (`pos < L`). `sensed` is the
  concentration at the tip's *current* position — no lookahead.
- `history`: one record per **past** step, in arrival order, so a stateless
  program can reconstruct anything it wants to remember (visit counts,
  running averages, etc.).

### Output (stdout)

```json
{ "alloc": { "0": 3.0, "1": 6.0 } }
```

- One non-negative, finite amount of carbon per active tip id you want to
  fund this step (unlisted ids default to 0; unrecognized ids are ignored).
- The sum over active tips must be `<= budget_step`.
- A tip's position advances by `min(amount, L - pos)` — funding it past `L`
  simply isn't credited beyond the end.

**Validity.** A negative, non-finite, or non-numeric amount, a total over
`budget_step`, a crash, a timeout, or non-JSON output at **any** step voids
the **entire instance** (score `0.0` for it).

## Objective & scoring (deterministic)

You **maximize** total net carbon return: nutrient harvested minus
construction cost, summed over all steps and tips, using the real (hidden)
profile. The evaluator normalizes each instance with an affine anchor:

- `weak` = what a fixed "spread the budget evenly over every active tip,
  every step" recipe achieves (simulated directly by the evaluator).
- `ub` = an **omniscient offline upper bound**: knowing every tip's full
  profile in advance and ignoring the per-step / partial-information
  constraints, the best possible total carbon return, found by an exact
  small knapsack over "how much to sink into each tip" (a strict relaxation,
  so `ub` is never below anything achievable online, and it is usually not
  reachable in practice — this leaves headroom above a strong policy).

```
r = clamp( 0.1 + 0.9 * (your_net_return - weak) / max(ub - weak, 1e-6), 0, 1 )
```

Reproducing the evenly-spread recipe scores ≈ 0.1; doing worse scores below
0.1; every unit of net return you gain over it moves you toward 1.0. Your
final score is the mean of `r` over a fixed family of 10 developmental beds
(uniform-noise backgrounds with planted patches of varying depth, width, and
value — some beds hide a substantially bigger patch deep on a tip that
currently looks worse than a shallower, smaller patch elsewhere).

## Notes

- Scoring is fully deterministic and seeded — no wall-clock or hardware
  timing anywhere in the objective.
- Your program only ever sees the current step's public state and runs in an
  isolated sandboxed subprocess; the hidden profiles, the weak recipe, and
  the offline oracle bound are all computed only in the evaluator process.
- There is no free lunch: probing every tip costs carbon even when a tip
  turns out to be poor, and committing too early can miss a bigger patch.
  Uniform spreading, full commitment to the current best, and adaptive
  portfolios that keep a minimum probe on every tip while tilting the rest
  toward the best evidence so far are all viable strategies with very
  different payoffs.
