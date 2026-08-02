# Almost-No-Memory Relay Courier

A tiny courier robot patrols one-cell-wide corridor mazes. It has almost no
memory: each tick it senses only the landmark token painted on the floor
tile it is currently standing on, plus a small integer **mode** register
that it controls itself. Your job is not to solve one maze — it is to
**write the robot's controller once**, as a compact reactive state machine,
and have it work on mazes it has never been shown.

## World

- A maze is an `H x W` grid of tokens. `"#"` is wall (never enterable).
- Floor tiles carry one of four colors `R`, `G`, `B`, `Y`. A color's meaning
  as a compass direction depends on the robot's **current mode** — a fixed,
  globally-known code, given to you verbatim in the input:
  `phase0_code` (mode 0) and `phase1_code` (mode 1), e.g.
  `{"R":"N","G":"E","B":"S","Y":"W"}`. The SAME four colors mean different
  directions in the two modes.
- One tile per maze is the **relay beacon**, token `"K_<color>"`. Its
  color is always meant to be read with `phase1_code` (it tells you where
  to go immediately *after* recalibrating) — but recalibration itself is
  not automatic: it happens only because *you* write a rule for that
  `(state, "K_<color>")` pair whose `next` is the mode that reads
  `phase1_code`. From then on, every rule you wrote for that mode governs
  how colors are read; a controller that never transitions never
  recalibrates.
- One tile is the **goal**, token `"X"`; reaching it ends the run.
- Moving into a wall or off the grid does nothing (the robot stays put) but
  still consumes one of the run's `max_steps` ticks.

## Candidate program contract

Standalone program, stdin -> stdout, run once in an isolated subprocess.

**stdin** (public instance): `state_budget`, `actions` (`["N","S","E","W","WAIT"]`),
`phase0_code`, `phase1_code`, `max_steps`, `n_total_grids` (10),
`n_holdout` (7), and `visible_grids`: 3 mazes given **in full**
(`width`, `height`, `start`, `grid`). The other 7 mazes are **never shown
to you** — they are generated the same way (same color codes, same
mechanics, larger on average) and are scored only by the evaluator.

**stdout**: ONE controller —
```json
{"start_state": 0,
 "rules": [{"state": 0, "see": "R", "action": "N", "next": 0}, ...]}
```
`rules` is a Mealy machine: `(state, sensed token) -> (action, next state)`.
`state`/`next` must be integers in `[0, state_budget)`; `action` must be one
of the allowed actions. A `(state, token)` pair with no matching rule
defaults to `WAIT`, same state. Any structurally malformed answer (wrong
types, an out-of-range state, an unknown action, more than 600 rules) scores
the whole instance `0.0`.

## Scoring (deterministic)

The evaluator **simulates your one controller** (never your code — just the
rule table you returned) against all 10 mazes, starting at `start_state`.
Per maze, let `g`:
- `0.0` if the beacon is never reached,
- `0.35` if the beacon is reached but not the goal,
- `0.60 + 0.40 * clamp(opt_steps / steps_used, 0, 1)` if the goal is
  reached, where `opt_steps` is that maze's true shortest-path length.

Let `rule_count = len(rules)` and `REF_SIZE = 14`:
`size_factor = 1.0` if `rule_count <= REF_SIZE`, else
`max(0.15, REF_SIZE / rule_count)` — bigger controllers score less, all
else equal.

Each maze's reported value is `v = 0.05 + 0.80 * g * size_factor`
(clamped to `[0, 1]`). **Ratio** is the mean of the 10 `v`'s; **Vector**
lists them in generation order (3 visible, then 7 held-out).

## Why the obvious move fails

The 3 visible mazes are shown in full, so the path of least resistance is
to shortest-path one of them on raw coordinates and hardcode that exact
move sequence as a chain of one-shot states. It reproduces that ONE maze
almost perfectly — and burns roughly one state per step doing it. On any
other maze the token seen at a given state index is essentially never what
the memorized chain expects, so it falls back to `WAIT` and stalls, usually
before even reaching the beacon.

## Suggested strategies

1. **Null controller**: no rules; never moves.
2. **Memorize the visible maze**: BFS one visible grid, hardcode the path.
3. **Decode the color code**: a 2-mode table (~12 rules) reading
   `phase0_code`/`phase1_code`, switching mode at the beacon — identical
   performance on visible and held-out mazes alike.
4. Refine rule count / redundancy against the size penalty once the core
   reactive table works.
