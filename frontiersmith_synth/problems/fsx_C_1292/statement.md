# Dungeon Descent: Consumables Buy Depth

## Setting

You are planning a single expedition into a deterministic dungeon of `F` floors.
You start at depth `0` (the entrance) with `hp_start` HP (max `hp_max`), a stock
of **potions** and **wards**, and a turn budget `T`. You submit your **entire
plan** as an ordered list of actions; it is replayed turn by turn against the
dungeon's (fully known) floor data:

- **`{"action": "descend", "use_ward": true|false}`** — attempt floor `d -> d+1`.
  You take `hazard[d]` damage, UNLESS `use_ward` is `true` (and you have a ward),
  in which case the ward **fully negates** that hit and is consumed. If your HP
  drops to `0` or below, you **die immediately**: the expedition ends and
  **all** its reward is lost — no matter how deep you had gotten. This is the
  whole risk: there is no partial credit for an unsurvived run.
- **`{"action": "drink_potion"}`** — consume one potion, healing `potion_heal`
  HP (capped at `hp_max`).
- **`{"action": "grind"}`** — stay at your current depth: heal `grind_regen` HP
  (capped), and if this is the **first** grind ever performed at this depth,
  gain `grind_loot[depth] = [potion_gain, ward_gain]`. Grinding a floor you've
  already cleared yields **no more loot** (only the HP regen) — loot dries up.

Each action consumes one turn. The plan stops being applied once you die, once
`T` turns have been used, or once the list is exhausted (an implicit "head
back safely" with whatever you're currently banking).

## Objective

Let `depth` be the deepest floor you are standing on when the run ends **alive**.
Your banked value is `reward[depth]` (a table given in the input, `reward[0] = 0`,
strictly increasing and **convex** in `d` — each additional floor is worth
disproportionately more than the last). If you die, banked value is `0`.

Score is affinely anchored against the evaluator's own "always descend, never
use an item" reference run (`base`, computed the same deterministic way from
the same input) and a fixed reward-cap (`cap`, strictly above anything reachable
in this dungeon, so no run can ever saturate the score):

```
r = clamp( 0.1 + 0.9 * (banked - base) / max(cap - base, 1), 0, 1 )
```

Racing to maximum depth as fast as possible maximizes reward *per turn* — until
you meet a floor your current HP simply cannot survive, and you bank **nothing**.
Playing it completely safe (grinding, rarely descending) never dies, but never
reaches a depth worth much either. The prize is in reading your own consumable
stock against the hazard curve ahead: wards and potions are what convert into
*survivable* depth.

## Candidate contract (isolated stdin -> stdout program)

Read ONE JSON public instance from stdin, print ONE JSON list of actions (as
above) to stdout. Your program runs in an isolated subprocess.

### Public instance (stdin)

```json
{
  "F": 26, "T": 30, "hp_max": 100, "hp_start": 100,
  "potions_start": 3, "wards_start": 1,
  "potion_heal": 34, "grind_regen": 16,
  "hazard": [7, 7, 8, ...],
  "reward": [0, 6, 17, 31, ...],
  "grind_loot": [[1,0], [1,0], [0,0], ...],
  "seed": 910002
}
```
`hazard` has length `F` (`hazard[d]` = damage descending floor `d -> d+1`).
`reward` has length `F+1` (`reward[d]` = banked value at depth `d`).
`grind_loot` has length `F`.

### Answer (stdout)

A JSON array of action objects as defined above, applied in order. Any
structurally or semantically invalid action anywhere in the list (unknown
`action` name, `drink_potion` with zero potions, `use_ward: true` with zero
wards left, a non-boolean `use_ward`, or a payload that isn't a JSON list)
**rejects the entire run**: score `0` for that instance.

## Notes

- **Deterministic**: all 10 instances are fixed and seeded; the same
  submission always gets the same `Ratio`/`Vector`.
- Several instances plant a hazard spike large enough to kill you **even at
  full HP** — surviving it needs a ward used on that specific floor, not more
  healing. Others run the grind loot dry after a few floors, so consumables
  must be front-loaded before you need them, not farmed reactively once low.
- Multiple strategies are viable: reckless descent (fast but fatal), pure
  grinding (safe but worthless), a reactive HP-threshold policy (better, but
  blind to hazard *magnitude* and to attrition), or a policy that reads the
  hazard/loot tables and its own inventory to decide how deep it can safely
  go before banking.
