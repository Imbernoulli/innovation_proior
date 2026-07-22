# Wound-Ring: A Regenerating Cellular Organism

## Story
A creature's tissue lives on a `H x W` grid. Every cell is a finite-state machine
with a **von Neumann neighborhood**: itself plus North/South/East/West. Some cells
are permanent **walls** (skeleton/environment) that never change; off-grid neighbors
also count as wall. The creature starts from a **single seed cell** and must grow,
under its own **local update rule** applied **asynchronously** (cells update one at a
time in a scrambled-but-deterministic order each tick, each update immediately
visible to the rest of that tick), into a specified target body plan and hold it.
The evaluator then cuts a rectangular **wound** out of the settled body (resets those
cells to empty) and keeps applying the **same** rule — regeneration is whatever
happens next.

## Task
Write a **standalone program**: read ONE JSON instance from `stdin`, write ONE JSON
answer to `stdout` — your local update rule.

### Public instance (stdin)
```json
{ "name":"corridor1", "H":15, "W":15,
  "seed":[7,7], "seed_state":1, "K":9, "wall_state":8,
  "walls":[[r,c], ...],
  "target":[[int]*15]*15,
  "n_growth_ticks":26, "n_repair_ticks":20, "max_table_entries":20000 }
```
States are integers `0..K-1`: `0` = empty, `1..K-2` = live tissue "ring" values,
`K-1` = wall (immutable, and used for any off-grid neighbor too). `target[r][c]` is
the state cell `(r,c)` should hold once fully grown/healed.

### Answer (stdout) — your rule
```json
{ "table": { "c,n,s,e,w": next_state, ... }, "default": "stay" }
```
Each key is a literal string of 5 comma-joined ints in `[0,K-1]`: `c`=self,
`n,s,e,w`=neighbor states (North,South,East,West; off-grid = `wall_state`). Each
value is the next state, an int in `[0,K-1]`. `default` is applied to any 5-tuple
absent from `table`: either the string `"stay"` (next = current self state) or a
fixed int in `[0,K-1]`. `table` may have at most `max_table_entries` entries.

### Validity
Any violation — non-dict `table`, malformed/out-of-range key or value, table too
large, a crash, a timeout, or non-JSON — scores **0.0** on that instance.

## Transition (per tick, deterministic, ASYNCHRONOUS)
Wall cells never change. Every OTHER cell is visited once this tick, in a scrambled
order seeded by the instance (not by your rule). Visiting a cell: look up its
CURRENT 5-tuple `(self,N,S,E,W)` in your table (or `default` if absent) and write the
next state immediately — later cells in the same tick already see this update.

## Episode & scoring (deterministic)
1. **Grow**: from an empty grid (only the seed cell alive, at `seed_state`, plus the
   fixed walls), run `n_growth_ticks` ticks. Growth fidelity = fraction of cells in a
   padded bounding box of the body (excluding walls) matching `target`.
2. **Wound & repair**: from the grown grid, the evaluator cuts several **hidden**
   rectangular wounds (their location and size are never revealed to you — your rule
   must generalize, not special-case a known cut), resets those cells to empty, and
   runs `n_repair_ticks` more ticks with your SAME rule. Repair fidelity per wound =
   fraction of the wound's own cells matching `target` afterward.
3. Per instance: `obj = 0.3*growth_fidelity + 0.7*mean(repair_fidelity)`.
4. `r = clamp(0.1 + 0.8*(obj - obj_base)/(1 - obj_base), 0, 1)`, where `obj_base` is
   the do-nothing rule's objective on the same instance (≈0, so do-nothing ≈ 0.1).
   The `0.8` coefficient caps the reachable score at `0.9`, leaving headroom.
Final score = mean `r` over **10** fixed seeded instances: open disks, obstacle
mazes, branching multi-limb bodies, edge-clipped bodies, and a larger held-out maze.

## Why it is open-ended
Your rule never sees its own coordinates — only local states. Copying a neighbor's
state and freezing often grows the shape once (a single outward wave usually offers
only the truly-nearest neighbor when a cell first turns on), but a reopened interior
wound offers several already-settled neighbors of *different* distances at once, in
patterns the one-shot trace never saw — a frozen rule heals wrong or not at all.
Real regeneration needs the rule to maintain, and *re-derive* under damage, an
internally reconstructable positional coordinate (distance from the seed through
live tissue), not just a static shape. Obstacles turn that coordinate into a
maze/branch structure, and different wound shapes/positions stress it differently:
no single recipe is best everywhere.

## Isolation
Your program runs in a fresh sandboxed subprocess and only ever sees the public
instance above. The wound locations, the tick simulator, and the do-nothing
reference all live only in the evaluator process.
