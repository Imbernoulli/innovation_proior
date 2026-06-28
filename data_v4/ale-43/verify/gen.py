#!/usr/bin/env python3
"""Instance generator for "Sokoban-Style Box Pushing" (ALE-Bench heuristic optimization).

Usage:
    python3 gen.py <seed>

Writes one instance to stdout in the format:

    H W S
    row_0
    row_1
    ...
    row_{H-1}

where each row is a string of exactly W characters drawn from:
    '#' wall            (impassable; the outer border is always wall)
    '.' empty floor
    '@' the single agent, on empty floor
    '+' the agent standing on a target cell
    '$' a box, on empty floor
    '*' a box already resting on a target cell
    'o' an (unoccupied) target cell

Meaning: a single agent moves on the grid with classic PUSH-ONLY Sokoban
mechanics. In one move the agent steps one cell in {U,D,L,R}; if the destination
holds a box, the box is PUSHED one further cell in the same direction, which is
legal only when that further cell is empty floor or a target (never a wall, never
another box -- the agent cannot pull, and cannot push two boxes at once). The
agent has a budget of S moves. The objective is to MAXIMIZE the number of boxes
resting on target cells after the move sequence is replayed (see score.py /
context.md for the exact rule and the feasibility -> 0 floor).

Instance regime (deterministic from the seed):
  * Grid H x W with H, W in [8, 14]; the outer border is solid wall.
  * A scatter of interior walls (a few percent of interior cells), kept sparse so
    that boxes are generally pushable and deadlocks are avoidable but not free.
  * B boxes and exactly B targets (B in [4, 9]); a few boxes may start already on
    a target. Agent placed on a remaining empty floor cell.
  * The move budget S is generous but finite (a multiple of H*W), so the binding
    constraint is the ORDER in which boxes are parked and how far the agent has to
    walk to reposition behind each box, not raw budget starvation.
The generator only guarantees a syntactically valid board (single agent, equal
boxes and targets, solid border). It does NOT guarantee every box can be parked;
the score is continuous in how many the solver manages to park.
"""
import sys
import random


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <seed>\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    rng = random.Random(0x43B0_0000 ^ (seed * 2654435761 & 0xFFFFFFFF))

    H = rng.randint(8, 14)
    W = rng.randint(8, 14)
    B = rng.randint(4, 9)

    # grid[y][x]: '#','.','o' (target). agent / boxes tracked separately.
    grid = [['.' for _ in range(W)] for _ in range(H)]
    for x in range(W):
        grid[0][x] = '#'
        grid[H - 1][x] = '#'
    for y in range(H):
        grid[y][0] = '#'
        grid[y][W - 1] = '#'

    interior = [(y, x) for y in range(1, H - 1) for x in range(1, W - 1)]

    # sparse interior walls (a few percent), but never so many we choke the board.
    n_int = len(interior)
    n_wall = rng.randint(0, max(0, n_int // 12))
    rng.shuffle(interior)
    walls = set()
    for i in range(n_wall):
        y, x = interior[i]
        grid[y][x] = '#'
        walls.add((y, x))

    free = [(y, x) for (y, x) in interior if (y, x) not in walls]
    rng.shuffle(free)

    # we need B box cells, B target cells, 1 agent cell -- all distinct floor cells.
    need = 2 * B + 1
    if len(free) < need:
        # board too cramped for this B: shrink B to fit.
        B = max(1, (len(free) - 1) // 2)
        need = 2 * B + 1

    box_cells = [free[i] for i in range(B)]
    target_cells = [free[B + i] for i in range(B)]
    agent_cell = free[2 * B]

    boxset = set(box_cells)
    targetset = set(target_cells)
    # mark targets on the grid
    for (y, x) in target_cells:
        grid[y][x] = 'o'

    # render: combine layers into the single-character alphabet.
    out_rows = []
    for y in range(H):
        row = []
        for x in range(W):
            is_box = (y, x) in boxset
            is_tar = (y, x) in targetset
            is_ag = (y, x) == agent_cell
            base = grid[y][x]
            if base == '#':
                row.append('#')
            elif is_ag and is_tar:
                row.append('+')
            elif is_ag:
                row.append('@')
            elif is_box and is_tar:
                row.append('*')
            elif is_box:
                row.append('$')
            elif is_tar:
                row.append('o')
            else:
                row.append('.')
        out_rows.append(''.join(row))

    # generous but finite move budget: a multiple of grid area.
    S = (H * W) * rng.randint(3, 5)

    out = [f"{H} {W} {S}"]
    out.extend(out_rows)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
