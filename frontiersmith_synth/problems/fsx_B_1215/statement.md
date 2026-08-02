# Ground Swelling

A volcano observatory keeps a catalogue of past **unrest episodes**: bouts of
ground deformation and seismicity that either culminated in an **eruption**, or
ended as a **failed intrusion** (magma pushed upward but stalled before
reaching the surface). Failed intrusions are the more common outcome. For each
past episode the catalogue logs three summary statistics:

- `ACC`  — peak **deformation acceleration** during the episode (how fast the
  ground uplift rate itself was speeding up).
- `INFL` — the cumulative **magma-chamber inflation index**, a proxy for how
  much magma actually accumulated beneath the volcano.
- `SEIS` — the background **seismic energy-release rate** during the episode.

...and whether it erupted: `erupted = 1` (eruption) or `0` (failed intrusion).
All three statistics tend to rise together whenever unrest intensifies — an
accelerating episode is a *seismically and geodetically loud* episode, whether
or not it ends in eruption. Your job: recover a closed-form expression for
`P(erupt)` from this catalogue, so it can be trusted on a **markedly more
intense bout of unrest now underway** — a regime the catalogue never reached.

## Input (stdin)
- Line 1: two integers `n` and a case id.
- Next `n` lines: `ACC INFL SEIS erupted`, one past episode each (`erupted` is
  `0` or `1`).

## Output (stdout)
One line: a closed-form Python expression for `P(erupt)` in variables `ACC`,
`INFL`, `SEIS`. Allowed: `+ - * / **`, unary `-`, numeric constants, and the
functions `sqrt log exp sig tanh absv`. Example (illustrative **form only —
NOT the hidden law**): `sig(0.4*ACC - 0.3*SEIS + 0.1)`. No other names are
accepted. The grader clips your prediction to `[0,1]` before scoring.

## Scoring (deterministic, maximization)
Your expression is evaluated on a **held-out set of episodes from the more
intense ongoing bout**, regenerated deterministically inside the grader from
the same case id — you never see it. Let `p_i` be your (clipped) prediction
and `y_i in {0,1}` the actual realized outcome at held-out episode `i`:

```
metric   = mean_i (p_i - y_i)^2                        # Brier score
O        = metric * (1 + LAMBDA * nodes)               # nodes = expr size
baseline = the same metric for predicting the constant training eruption rate
Ratio    = min(1000, 100 * baseline / O) / 1000
```

Lower held-out Brier score gives a higher `Ratio` (capped at `1.0`). Predicting
the constant training rate scores about `0.1`. `LAMBDA` is a small parsimony
weight. Held-out outcomes are single Bernoulli draws — irreducible even for a
perfectly recovered law — so no correct law reaches `Ratio = 1.0`.

## Why the obvious rule is a trap
A practitioner who scans the catalogue for **what accelerating episodes
usually looked like right before they erupted** — say, the lowest `ACC` and
`SEIS` ever seen ahead of a past eruption — and raises a high alert probability
whenever a new episode clears those levels, builds a rule from the *eruptive*
rows alone. It never checks how often the far more numerous *failed*
intrusions **also** cleared the same levels, because `ACC` and `SEIS` respond
to unrest intensity in general, not specifically to whether magma will reach
the surface. That rule looks plausible on the training catalogue, but it
structurally cannot express the actual determinant of eruption (how much
magma has accumulated) — and it over-forecasts hardest exactly where the held
episodes are loudest: the more intense ongoing bout, where acceleration and
seismicity are elevated across the board regardless of outcome. Only weighing
**both** outcome classes — asking which signal's distribution shifts between
eruptions and failed intrusions once the (low) base rate of eruption is
accounted for — finds that the chamber-inflation index, not the acceleration,
is what actually separates them, and recovers a law that keeps working once
the ground gets louder.

## Constraints
- Time limit 5 s, memory 512 MB; `n` ranges from 80 to 350 across cases.
- Held-out Bernoulli noise leaves irreducible error, so even a correctly-shaped
  law does not reach `Ratio = 1.0` — there is room above the reference
  solutions.
