# Sokoban-Style Box Pushing

## Research question

You control a single agent on an `H x W` grid with classic **push-only** Sokoban mechanics. On each
move the agent steps one cell up / down / left / right; if the destination cell holds a box, the box
is **pushed** one further cell in the same direction — which is legal only when that further cell is
empty floor or a target (never a wall, never another box; the agent cannot pull and cannot push two
boxes at once). The agent has a hard budget of `S` moves. The task is to emit a move string that
**maximizes the number of boxes resting on target cells** when the sequence is replayed.

Sokoban planning is PSPACE-complete in general, and this scored variant (maximize boxes parked under
a move budget) has no exact answer to read off: the quality of a move string is judged by a
continuous score, and a single illegal move floors that score to **0**. The lever is the heuristic
that decides *which* boxes to park, *on which* targets, and *in what order*.

## Input / output contract

- Input (stdin):
  - first line: `H W S` — grid height `H` and width `W` (`8 <= H, W <= 14`) and the move budget `S`
    (a few times `H*W`);
  - then `H` lines, each a string of exactly `W` characters over the alphabet:
    - `#` wall (impassable; the outer border is always wall),
    - `.` empty floor,
    - `@` the single agent on empty floor,
    - `+` the agent standing on a target cell,
    - `$` a box on empty floor,
    - `*` a box already resting on a target cell,
    - `o` an unoccupied target cell.
  - There are exactly `B` boxes and `B` targets (`4 <= B <= 9`); some boxes may start on a target.
- Output (stdout):
  - a single **move string** — a sequence of characters in `{U, D, L, R}` (`U`=up `y-1`, `D`=down
    `y+1`, `L`=left `x-1`, `R`=right `x+1`). Whitespace and newlines between the characters are
    ignored. The **empty string is allowed** and means "do nothing" (the agent never moves).
- Time limit: about 2 seconds. Memory: 256 MB.

Example shape: for a `5x5` board with the agent at one corner and a single box one cell away from one
target, a valid output is a short string like `RRU` that walks behind the box and pushes it onto the
target. An output that ever steps the agent into a wall, or pushes a box into a wall or onto another
box, is rejected outright (score 0) — it is **not** truncated at the bad move.

## Background

Move-level search is hopeless here: branching factor 4 and depth up to `S` (hundreds) make breadth-
or depth-first search over raw moves blow up immediately, and most move sequences are illegal or
useless. The established way to plan in Sokoban is to lift the search to **macro-moves**: instead of
"the agent steps left", the unit of search is "**push box X all the way to cell Y**". Two reference
points frame the design.

- **Greedy nearest-box-to-nearest-target pusher.** Repeatedly pick the closest loose box / free
  target pair (Manhattan), then push that box toward that target one axis-aligned step at a time,
  walking the agent behind the box (via a floor BFS that treats boxes as obstacles) before each
  push, committing a push only if it is legal and affordable. This is fast and always feasible, but
  it is myopic: it ignores the *order* in which boxes are parked, lets one box block another's only
  corridor, and happily pushes a box into a corner it can never escape. This is the deterministic
  baseline the scorer normalizes against.
- **Macro-move beam search with deadlock pruning.** Precompute, per box, its **push-reachable set**
  — every cell the box can be driven to and at what cost — with a single-box **push BFS** over
  states `(box cell, the floor-component the agent can reach given that box blocks the grid)`. From
  that BFS any reachable destination yields a concrete, rule-honoring move string (agent walks + the
  push step). On top of the macros, run a **beam search over the order** of "park box X on target Y"
  macros, each beam state carrying the full board, scored by `(boxes parked, remaining budget)`. A
  static **deadlock detector** (a box pushed into a non-target corner, or into a wall-hugging pocket
  with no target) prunes hopeless box placements both inside the push BFS and when expanding the
  beam. The macro abstraction collapses the search depth from `S` moves to `B` parks, and the
  deadlock test keeps the beam from wasting itself on un-recoverable boards. This is the lever.

## Evaluation settings

A solution is first checked for **feasibility**; any violation floors the score to **0**:

1. the move string contains only characters in `{U, D, L, R}` (whitespace ignored);
2. its length is `<= S`;
3. every move in the replay is **legal** — the agent never steps into a wall, and every push drives
   a box into a cell that is inside the grid, not a wall, and not already occupied by another box.

An illegal move rejects the **whole** solution (score 0); it is not truncated. For a feasible
solution the scorer replays the moves from the board's starting layers and counts `parked` = the
number of boxes resting on target cells at the end. The score normalizes against the deterministic
**greedy nearest-box-to-nearest-target pusher** the scorer recomputes itself, which parks `base`
boxes on the same instance:

```
score = round(1_000_000 * (1 + parked) / (1 + base))     (0 if INFEASIBLE)
```

The `1 +` offsets keep the ratio well-defined when `base == 0` and reward the very first extra box.
The greedy baseline scores about `1_000_000`; parking strictly more boxes than greedy scores strictly
more, and an infeasible output scores `0`.

**Instances** are generated deterministically from an integer seed. The grid is `H x W` with
`H, W in [8, 14]` and a solid wall border; a sparse scatter of interior walls (a few percent of
interior cells, kept low so boxes are generally pushable and deadlocks are avoidable but not free);
`B in [4, 9]` boxes with exactly `B` targets placed on distinct floor cells, and the agent on a
remaining floor cell. The move budget `S = H*W * U[3,5]` is generous but finite, so the binding
constraint is the **order** in which boxes are parked and how far the agent must walk to reposition
behind each box, not raw budget starvation — exactly the regime where a greedy pusher leaves boxes
stranded and the macro-move beam pays off.

## Code framework

A single self-contained C++17 program that reads the instance from stdin and writes a feasible move
string to stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int H, W, S;

int main() {
    if (!(cin >> H >> W >> S)) return 0;
    cin.ignore();
    vector<string> g(H);
    for (int i = 0; i < H; ++i) getline(cin, g[i]);

    // Parse layers: walls, targets, boxes, agent start, from the char alphabet
    // ('#','.','o','@','+','$','*'). The EMPTY move string is always feasible
    // (the agent never moves, so no move can be illegal) -- our guaranteed
    // fallback answer.
    //
    // TODO heuristic: precompute per box its PUSH-REACHABLE set via a single-box
    // push BFS over states (box cell, agent-reachable component), prune
    // statically-dead placements (corner / wall-pocket deadlocks), then run a
    // BEAM SEARCH over the ORDER of "park box X on target Y" macros, scoring beam
    // states by (boxes parked, remaining budget). Reconstruct each macro into a
    // concrete, rule-honoring move string (agent walks + push steps). Keep the
    // best feasible move string seen and print it.

    string moves = "";            // empty == do nothing == always feasible
    cout << moves << "\n";
    return 0;
}
```
