# Titrate the Unknown: Closed-Loop Equivalence-Point Control

A sealed vessel holds an unknown sample (a mixture of 1-3 weak-acid
"species") in water. You control a burette of strong-base titrant. You may
only **add** titrant — the cumulative volume `V` is monotone non-decreasing
and additions can **never be undone**, exactly like a real titration. After
every addition you may read the vessel's pH. Your goal: reach a given
**target pH** using as little titrant, and as few readings, as possible —
without overshooting past it, because once you pass it you cannot go back.

The sample's true pH curve is smooth and strictly increasing in `V`, built
from independent neutralization steps, one per species: each species
contributes a rise around its own equivalence volume. Species differ in
**buffer capacity**: some produce a wide, gently-buffered stretch (pH moves
little per mL added); others produce a narrow, steep jump (pH moves a lot
per mL). Where these regions sit, and how narrow the jumps are, is **not
disclosed** — you only ever see `(V, pH)` readings you have taken.

## Protocol (interactive, one round-trip per program run)

Your program is invoked **once per round**, up to `max_rounds` times for
each of the 10 fixed instances. It is **stateless** across calls — you must
re-derive any adaptive logic purely from the `history` you are handed each
time; nothing persists between invocations.

**stdin (one JSON object):**
```json
{"phase": "step", "round": 3, "max_rounds": 18, "rounds_left": 15,
 "V": 8.19, "pH": 3.52,
 "history": [[0.0, 2.46], [0.42, 2.50], [2.90, 2.77], [8.19, 3.52]],
 "V_max": 21.07, "target_pH": 6.54, "max_add": 7.37}
```
`history` lists every `[V, pH]` reading so far, including the current one
(same as the top-level `V`/`pH`). `V_max` is a hard cap on total titrant.
`max_add` is the burette's single-dispense safety limit (always `0.35*V_max`,
given explicitly each round): no single addition may exceed it, so the
target — which always needs more titrant than one dispense can deliver —
can never be reached without at least a few genuine, feedback-driven
rounds, however confident you are about where it sits.

**stdout (one JSON object):**
```json
{"add": 1.35}
```
`add` is the volume (mL) of titrant to add before the next reading. Sending
`add <= 0` means **stop now**: the episode ends immediately at the current
`(V, pH)`. If you never stop, the episode auto-ends after `max_rounds`
readings, using whatever `V` you last reached.

## Feasibility (violating this scores `0` on that instance)

`add` must be a finite real number. If `V + add` would exceed `V_max`, or if
`add` exceeds `max_add`, your answer is infeasible for that instance.

## Scoring (minimize; deterministic, no wall-time)

Let `V_final, pH_final` be the state when you stop (or the budget runs out),
and `steps` the number of additions actually made. With `V*` the (hidden)
true volume at which `pH(V*) = target_pH`:

```
pH_err  = |pH_final - target_pH|
excess  = max(0, V_final - V*) / V*
cost    = pH_err + 0.12*excess + 0.15*(steps / max_rounds)
```

The evaluator also computes `cost_ref` from its own reference construction:
a **blind fixed-size increment every round**, ignoring pH entirely except to
stop the first time it crosses the target. Your score on that instance is

```
quality = clip(1 - cost/cost_ref, 0, 1)
r = 0.10 + 0.82 * quality        (so r in [0.10, 0.92])
```

matching the reference exactly maps to `0.10`; the `0.82` span leaves
headroom above any reference solution. An infeasible, malformed, crashed, or
timed-out round scores `0.0` for that whole instance. The reported `Ratio`
is the mean of `r` over 10 deterministic, seeded instances (single-species
gentle curves plus multi-species curves that hide a steep, narrow
equivalence jump behind — or in front of — a gentle plateau). `Vector`
lists the 10 per-instance scores.

Your program runs in an **isolated subprocess**, sandboxed and re-invoked
fresh each round; it only ever sees the JSON shown above.
