# Ensemble Cue Policy Through a Rubato

## Story

A small chamber ensemble is reading through a passage of `T` beats. You are
the conductor: at every beat you broadcast one tempo-multiplier **cue**
(around `1.0` = the base tempo) to the whole ensemble at once. Each player
only responds to your cue after their own **reaction latency** (a few beats,
since the signal has to reach their eyes/ears and they have to physically
react) and then closes part of the gap to it every beat according to their
own **inertia**. Both latency and inertia are known from rehearsal.

The passage has a designated **phrase carrier** at every beat — the score
tells you which player currently has the melodic line and is "pulling" the
tempo expressively (a `role_weight` per beat, over players, summing to 1,
strongly concentrated on whoever carries the phrase right now, with a smooth
handoff when the phrase moves to a different player). Other players mostly
keep their own inner pulse and only loosely sense the expressive push and
pull.

You do not get to adjust your cues live during the performance — you submit
your **entire cue schedule** for the whole passage up front, exactly like a
conductor who has studied the whole score before rehearsal.

## Public instance (stdin)

```json
{
  "n_players": int,
  "T": int,
  "role_weight": [[float, ...]],   // T x n, phrase-carrier weight per beat (score annotation, exact)
  "latency":     [int, ...],       // length n, each player's cue reaction delay in beats
  "inertia":     [float, ...],     // length n, each player's per-beat closing fraction (0,1]
  "observed":    [[float, ...]],   // T x n, each player's own noisy first read-through tempo,
                                    // with NO conductor cueing at all
  "seed": int
}
```

`observed[t][i]` is player `i`'s own inclination at beat `t` before any cue:
noisy, and only loosely related to the true phrase shape unless player `i`
is that beat's phrase carrier.

## Answer (stdout)

A JSON list of `T` finite floats in `[0.1, 4.0]`: `cue[t]` is the tempo
multiplier you broadcast at beat `t`. (A JSON object `{"cue": [...]}` is
also accepted.) Wrong length, a non-finite value, a value outside
`[0.1, 4.0]`, a crash, a timeout, or no output scores the instance **0**.

## Scoring

The evaluator re-simulates each player's actual tempo trajectory under your
cue schedule using their real latency and inertia (`v[0]=1.0`; each beat,
every player closes `inertia` of the gap toward the cue they received
`latency` beats ago). From that it computes:

- **tracking error** — RMSE between the phrase-weighted ensemble tempo
  (`role_weight`-weighted average of players' realized tempi at each beat)
  and the hidden true phrase tempo curve;
- **tightness error** — the phrase-weighted spread of players' tempi around
  that ensemble tempo at each beat (do the players actually land together,
  weighted by who currently matters most).

These combine into one error (tracking weighted more heavily than
tightness). The evaluator also computes this same error for its own
internal baseline policy — a flat cue of `1.0` for the whole passage, i.e.
never moving at all — and reports

```
r = clamp( 0.1 + 0.9 * (err_base - err_cand) / err_base, 0, 1 )
```

per instance, so reproducing the flat baseline maps to `~0.1` and driving
the error toward zero maps toward `1.0`.

```
Ratio:  <mean of per-instance r, in [0,1]>
Vector: [r_1, r_2, ..., r_10]
```

## Objective

**Maximize `Ratio`.** Instances include metronomic warm-ups (the true tempo
barely moves) and real rubato passages (genuine tempo swells). There is no
easy optimum: a policy that just reacts to what it currently hears, without
distinguishing who is carrying the phrase and without accounting for how
long its cue takes to reach each player, tends to arrive both diluted and
late through a swell. A policy that reads the whole passage in advance,
trusts the structural markings about who is carrying the phrase at each
moment, and cues ahead of the ensemble's own reaction time does
substantially better across the whole family.
