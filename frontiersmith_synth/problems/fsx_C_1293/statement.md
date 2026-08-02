# The Hill Isn't Worth Less Than the Kill

You command a small squad on a grid against a fixed, fully-disclosed enemy
AI. The battle runs a fixed number of **turns**. Two designated tiles are
the **objective** — every turn-end a living squad member stands on one, you
score a point. That's it: the whole battle's score is the objective points
you accumulate, plus a small credit for units you keep alive and enemies
you kill. Nothing else counts.

Every tile has a **defense bonus** (0, 1 or 2). It cuts incoming damage to
whoever stands there, dealt by whoever attacks from there — so a tile is
simultaneously a shield for you and, if you take it, a shield you have
denied the enemy for however many turns remain. The two objective tiles
always carry the maximum defense bonus.

**Zone of control (ZoC):** every living unit (either side) projects its
control onto its 4 orthogonal neighbours. A move may step *into* such a
tile — that is always a legal place to stop — but may never continue
*past* it. You cannot dash around or through a guarded doorway; you either
fight there or route around entirely.

**Focus fire vs. spread:** every attack order given this turn is pooled per
target and applied all at once, only after every unit has moved. A target
that dies this turn contributes nothing to the enemy's counter-attack and
loses whatever tile it was denying you; a target merely wounded keeps both.

## Candidate program contract

Standalone program, invoked **once per turn** (many times per battle — a
fresh, isolated subprocess call each time, no memory across calls).

```python
import sys, json
inst = json.load(sys.stdin)
# ... decide this turn's moves/attacks ...
print(json.dumps({"orders": [...], "state": ...}))
```

### Public input for turn `t` (stdin)

```json
{
  "turn": 3, "total_turns": 9, "W": 10, "H": 6,
  "terrain": [[0,0,2,...], ...],
  "objective_tiles": [[3,2],[3,3]],
  "friendly": [{"id":"f0","x":3,"y":2,"hp":7,"atk":4,"move":3}, ...],
  "enemy":    [{"id":"e1","x":6,"y":3,"hp":10,"atk":4,"move":3}, ...],
  "state": <whatever you returned last turn, or null on turn 0>
}
```

`terrain[y][x]` is the defense bonus, or `-1` for an impassable wall.
Only currently-*living* units appear. `state`: an opaque JSON value you
control (max ~40000 chars) — nothing persists between calls, so any memory
you want must ride here.

### Answer (stdout)

```json
{"orders": [{"unit_id":"f0","move_to":[4,2],"attack":"e1"}, ...], "state": ...}
```

Exactly one entry per living friendly unit. `move_to`: `null` to stay, or a
destination tile (no path needed) legal iff reachable within your move
points stepping only through in-bounds, non-wall, unoccupied tiles, where
stepping into a ZoC tile is always allowed but must be the last step of the
move. `attack`: `null`, or an enemy id adjacent (Manhattan distance 1) to
your unit's position *after* its move. A malformed top-level answer (not
the right shape, wrong/duplicate/missing unit ids, oversized `state`) makes
the **whole battle score 0**. A single infeasible order (bad destination,
out-of-range attack) is simply ignored for that unit — it stays and does
not attack — the rest of the turn still counts.

## Fixed enemy AI (deterministic, disclosed in full)

Each surviving enemy, in ascending id order: if not already adjacent to a
living friendly unit, it moves (same ZoC rule) to the reachable tile
minimizing Manhattan distance to the *current* nearest living friendly unit
(ties: lower id). After all enemy moves, every enemy now adjacent to a
living friendly unit adds pooled damage to the reachable friendly unit with
the **lowest current HP** (ties: lower id) — the enemy always finishes the
weakest target it can reach, exactly like a damage-greedy attacker would.

## Scoring

For 10 fixed battles, the grader also computes `baseline` = the score of
"never move, never attack" (always feasible). Per-battle
`r = min(1, 0.1 * score / baseline)`. **Ratio** is the mean over all 10
battles; **Vector** lists the per-battle scores.

## Constraints

Grid up to 11×8, squads up to 6 units, up to 11 turns, time limit 5s per
turn call, memory 512MB. Everything is deterministic — no randomness, no
wall clock, no network.
