#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0938 -- "Wavefront Filling: Self-Reconfiguring Modular Swarms"
(family: programmable-matter-selfrecon; format B, quality-metric; theme: a robot swarm that
reshapes itself).

THEME.  `N` identical unit modules sit on a walled 2D lattice at a START configuration and
must reshape into a TARGET configuration of `N` cells (a lattice-embedded programmable-matter
problem, e.g. Kilobot / modular-robot self-reconfiguration).  Time proceeds in `T`
SYNCHRONOUS discrete rounds.  In each round every module looks at its own 4-neighbourhood
(N,E,S,W) and picks ONE action (STAY or move one step) using a single LOCAL RULE TABLE --
one shared table applied to every module and every round OF THAT INSTANCE'S run (no module
IDs, no global controller, no per-round replanning; a run may of course read the public
instance to construct a good table, exactly like any other format-B answer). Moves are
resolved in parallel: a move into a cell occupied at the START of the round is always
blocked (no swaps). If 2+ modules request the SAME currently-empty cell in the same round,
the request from the module at the numerically SMALLEST (row,col) wins -- a fixed,
content-independent arbitration rule (part of the environment's physics, not information
handed to any candidate); the other claimants stay and re-evaluate next round once the
contested cell's occupancy has changed. See `simulate()` below for the exact mechanics.

TWO STATIC BROADCAST FIELDS (mechanism: gradient-coordination).  The evaluator precomputes,
once per instance, two scalar fields over every free cell (BFS shortest-path distance,
respecting walls):
  A(cell) = distance to the NEAREST target cell           ("where do I still need to go")
  G(cell) = distance to the NEAREST seed cell              ("how deep into the structure am I")
`seeds` is a short public list of cells (almost always inside/at the far end of the target
shape) -- a single broadcast signal every module can read at its own cell and its neighbours',
exactly like the spec's "broadcast seed gradient". Neither field depends on module positions.

LOCAL-RULE-TABLE (mechanism: local-rule-table).  For each of its 4 neighbours (order
N,E,S,W; N=(-1,0), E=(0,+1), S=(+1,0), W=(0,-1)) a module computes ONE digit 0-8:
    0            = wall or off-grid (can never move there)
    1..4 (OCC)   = neighbour is a FREE-of-walls cell but currently occupied by another
                   module this round: 1=neither field improves by moving there, 2=only G
                   improves, 3=only A improves, 4=both improve
                   (digit = 1 + 2*impA + impG)
    5..8 (FREE)  = neighbour is empty and may be moved into: 5=neither improves, 6=only G
                   improves, 7=only A improves, 8=both improve (digit = 5 + 2*impA + impG)
  "improves" means the field is STRICTLY LOWER at the neighbour than at the module's own
  cell.  The 4 digits concatenated in N,E,S,W order form a 4-character key; your program
  outputs a table mapping every key it cares about to an action in {STAY,N,E,S,W}; any key
  you omit uses your declared "default" action (STAY if you omit that too).

SELF-RECONFIGURATION (mechanism 3).  The evaluator applies your table for T rounds and
scores how much of the TARGET shape is finally occupied.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON public instance:
    {"name":str,"H":int,"W":int,"N":int,"T":int,
     "walls":[[r,c]...], "start":[[r,c]... N cells], "target":[[r,c]... N cells],
     "seeds":[[r,c]...]}
  stdout: ONE JSON object: {"table": {"XXXX": "ACT", ...}, "default": "STAY"}
    keys are exactly 4 characters, each in '0'-'8'; ACT in {"STAY","N","E","S","W"}.

SCORING (deterministic; no wall-time).  base = |start ∩ target| / N (the "do nothing"
overlap).  obj = |final ∩ target| / N after simulating T rounds with the submitted table.
    r = clamp( 0.1 + 0.9 * (obj - base) / max(CEIL - base, eps), 0, 1 )     CEIL = 1.15
Doing nothing scores exactly 0.1 on every instance. The CEIL constant means even a PERFECT
reconfiguration (obj == 1) does not saturate r at 1.0 (r tops out at 0.1+0.9/1.15 ~= 0.883
when base == 0) -- headroom stays open above every reference solution. The final score is
the mean of r over 10 fixed seeded instances (open reconfigurations plus single-file "tube"
traps where the target region is only 1 cell wide, so a module that stops moving the
instant it reaches ANY target cell -- which is what purely local greedy descent on A does,
since A cannot improve below its own minimum of 0 -- permanently blocks every module
behind it).

ISOLATION.  The candidate is untrusted and runs in a FRESH bwrap-SANDBOXED SUBPROCESS via
`isorun.run_candidate`; it only ever sees the public instance above.

CLI:  python3 evaluator.py <solution.py>
"""
import sys, json
from collections import deque
import isorun

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]   # N, E, S, W
ACT_DELTA = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
VALID_ACTS = {"STAY", "N", "E", "S", "W"}
INF = 1 << 20
CEIL = 1.15   # headroom constant: even a PERFECT reconfiguration (obj==1.0) must not
              # saturate the reported ratio at 1.0, leaving room above any reference
              # solution for a stronger policy.


# ----------------------------- geometry helpers -----------------------------
def rect_block(r0, c0, rows, cols, n):
    out = []
    for r in range(r0, r0 + rows):
        for c in range(c0, c0 + cols):
            out.append((r, c))
            if len(out) == n:
                return out
    raise ValueError("rect too small")


def bfs_multi_source(free, sources, H, W):
    dist = [[INF] * W for _ in range(H)]
    q = deque()
    for (r, c) in sources:
        if free[r][c] and dist[r][c] > 0:
            dist[r][c] = 0
            q.append((r, c))
    while q:
        r, c = q.popleft()
        d0 = dist[r][c]
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and free[nr][nc] and dist[nr][nc] > d0 + 1:
                dist[nr][nc] = d0 + 1
                q.append((nr, nc))
    return dist


# ----------------------------- instance construction ------------------------
def _grid_from_free(H, W, free_cells):
    free_cells = set(free_cells)
    free = [[(r, c) in free_cells for c in range(W)] for r in range(H)]
    walls = [[r, c] for r in range(H) for c in range(W) if not free[r][c]]
    return free, walls


def _open_instance(name, H, W, N, start, target, extra_walls, seed_cell, T):
    free_cells = [(r, c) for r in range(H) for c in range(W)]
    free, walls = _grid_from_free(H, W, free_cells)
    for (r, c) in extra_walls:
        free[r][c] = False
        walls.append([r, c])
    assert len(start) == N and len(set(start)) == N
    assert len(target) == N and len(set(target)) == N
    for (r, c) in start + target:
        assert free[r][c], (name, r, c)
    return {"name": name, "H": H, "W": W, "N": N, "T": T,
            "walls": walls, "start": [list(x) for x in start],
            "target": [list(x) for x in target], "seeds": [list(seed_cell)]}


def _tube_instance(name, room_h, room_w, tube_row, tube_len, T, bend_at=None):
    """room occupies rows 0..room_h-1, cols 0..room_w-1 (fully open, no internal walls).
    A single 1-cell-wide tube exits the room to the east at row `tube_row`. If `bend_at`
    is given (h_len) the tube runs `h_len` cells east then turns south for the remaining
    `tube_len - h_len` cells. Everything outside room+tube in the bounding box is a wall,
    so the tube is the ONLY route from the room to its own far end."""
    room_cells = [(r, c) for r in range(room_h) for c in range(room_w)]
    if bend_at is None:
        tube_cells = [(tube_row, room_w + k) for k in range(tube_len)]
        H = room_h
        W = room_w + tube_len
    else:
        h_len = bend_at
        v_len = tube_len - h_len
        horiz = [(tube_row, room_w + k) for k in range(h_len)]
        corner_c = room_w + h_len - 1
        vert = [(tube_row + 1 + k, corner_c) for k in range(v_len)]
        tube_cells = horiz + vert
        H = max(room_h, tube_row + 1 + v_len)
        W = room_w + h_len
    N = tube_len
    start = rect_block(0, 0, room_h, room_w, N)
    free, walls = _grid_from_free(H, W, room_cells + tube_cells)
    seed_cell = tube_cells[-1]
    return {"name": name, "H": H, "W": W, "N": N, "T": T,
            "walls": walls, "start": [list(x) for x in start],
            "target": [list(x) for x in tube_cells], "seeds": [list(seed_cell)]}


def _two_tube_instance(name, room_h, room_w, tube_row_a, tube_row_b, tube_len, T):
    room_cells = [(r, c) for r in range(room_h) for c in range(room_w)]
    tube_a = [(tube_row_a, room_w + k) for k in range(tube_len)]
    tube_b = [(tube_row_b, room_w + k) for k in range(tube_len)]
    H = room_h
    W = room_w + tube_len
    N = 2 * tube_len
    start = rect_block(0, 0, room_h, room_w, N)
    free, walls = _grid_from_free(H, W, room_cells + tube_a + tube_b)
    target = tube_a + tube_b
    seeds = [list(tube_a[-1]), list(tube_b[-1])]
    return {"name": name, "H": H, "W": W, "N": N, "T": T,
            "walls": walls, "start": [list(x) for x in start],
            "target": [list(x) for x in target], "seeds": seeds}


def build_instances():
    out = []
    # ---- open / diffuse: N independent single-file COLUMNS, each module's own
    # nearest target cell sits in its own column/lane, so columns never interact and
    # greedy's per-module descent has nothing to jam on. ----
    out.append(_open_instance(
        "easy_open_1", 12, 8, 8,
        [(0, c) for c in range(8)], [(9, c) for c in range(8)],
        [], (9, 7), 16))
    out.append(_open_instance(
        "easy_open_2", 12, 14, 10,
        [(0, c) for c in range(10)], [(7, c + 2) for c in range(10)],
        [(9, 5), (10, 6), (8, 9)], (7, 11), 20))
    out.append(_open_instance(
        "easy_open_3", 9, 14, 12,
        [(0, c) for c in range(12)], [(6, c) for c in range(12)],
        [], (6, 11), 16))
    out.append(_open_instance(
        "easy_diag_shift", 10, 14, 9,
        [(0, c) for c in range(9)], [(4, c + 3) for c in range(9)],
        [], (4, 11), 16))
    # ---- single-file "tube" traps: >=3 planted here (this block has 4) ----
    out.append(_tube_instance("trap_straight_short", 4, 4, 2, 6, 30))
    out.append(_tube_instance("trap_straight_med", 5, 5, 2, 9, 38))
    out.append(_tube_instance("trap_straight_long", 5, 5, 2, 12, 44))
    out.append(_tube_instance("trap_bent", 4, 4, 1, 10, 38, bend_at=5))
    # ---- held-out / harder generalization instances ----
    out.append(_two_tube_instance("heldout_two_tubes", 6, 4, 1, 4, 5, 42))
    out.append(_tube_instance("heldout_long_bent_large", 6, 6, 2, 14, 40, bend_at=7))
    return out


# ----------------------------- simulation ------------------------------------
def local_key(r, c, H, W, free, occ, A, G):
    digits = []
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < H and 0 <= nc < W) or not free[nr][nc]:
            digits.append(0)
            continue
        occupied = (nr, nc) in occ
        impA = 1 if A[nr][nc] < A[r][c] else 0
        impG = 1 if G[nr][nc] < G[r][c] else 0
        base = 1 if occupied else 5
        digits.append(base + 2 * impA + impG)
    return "".join(str(d) for d in digits)


def simulate(inst, free, A, G, table, default):
    """Synchronous round update. A move into a cell OCCUPIED at the start of the round
    is always blocked (no swaps). If 2+ modules request the SAME currently-empty cell in
    the same round, the request from the module at the numerically SMALLEST (row,col)
    wins (a fixed, content-independent arbitration rule -- part of the environment's
    physics, not extra information handed to any candidate); the other claimants simply
    stay this round and re-evaluate next round against the now-changed occupancy (the
    contested cell they lost is occupied next round, so their local pattern changes and
    they naturally reconsider). This breaks transient contention within a couple of
    rounds while leaving the PERSISTENT self-block inside a single-file target region
    (an already-arrived module that never again sees a strictly-improving move) fully
    intact as a permanent physical blockage."""
    H, W = inst["H"], inst["W"]
    positions = [tuple(p) for p in inst["start"]]
    for _ in range(inst["T"]):
        occ = set(positions)
        requests = {}
        new_positions = list(positions)
        for i, (r, c) in enumerate(positions):
            key = local_key(r, c, H, W, free, occ, A, G)
            act = table.get(key, default)
            if act == "STAY":
                continue
            dr, dc = ACT_DELTA[act]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W) or not free[nr][nc] or (nr, nc) in occ:
                continue
            requests.setdefault((nr, nc), []).append(i)
        for cell, claimants in requests.items():
            winner = min(claimants, key=lambda i: positions[i])
            new_positions[winner] = cell
        positions = new_positions
    return positions


def _parse_answer(answer, H, W):
    """Return (table, default) or None if malformed."""
    if not isinstance(answer, dict):
        return None
    table_raw = answer.get("table")
    if not isinstance(table_raw, dict) or len(table_raw) > 7000:
        return None
    default = answer.get("default", "STAY")
    if not isinstance(default, str) or default not in VALID_ACTS:
        return None
    table = {}
    for k, v in table_raw.items():
        if not isinstance(k, str) or len(k) != 4 or any(ch not in "012345678" for ch in k):
            return None
        if not isinstance(v, str) or v not in VALID_ACTS:
            return None
        table[k] = v
    return table, default


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = build_instances()

    vec = []
    for inst in instances:
        H, W, N = inst["H"], inst["W"], inst["N"]
        free_cells = [(r, c) for r in range(H) for c in range(W)]
        wall_set = {tuple(w) for w in inst["walls"]}
        free = [[(r, c) not in wall_set for c in range(W)] for r in range(H)]
        target_cells = [tuple(t) for t in inst["target"]]
        seed_cells = [tuple(s) for s in inst["seeds"]]
        A = bfs_multi_source(free, target_cells, H, W)
        G = bfs_multi_source(free, seed_cells, H, W)

        base = len(set(tuple(s) for s in inst["start"]) & set(target_cells)) / N

        public = {"name": inst["name"], "H": H, "W": W, "N": N, "T": inst["T"],
                  "walls": [list(w) for w in inst["walls"]],
                  "start": [list(x) for x in inst["start"]],
                  "target": [list(x) for x in inst["target"]],
                  "seeds": [list(x) for x in inst["seeds"]]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            parsed = _parse_answer(ans, H, W)
            if parsed is None:
                vec.append(0.0)
                continue
            table, default = parsed
            final_positions = simulate(inst, free, A, G, table, default)
            if len(set(final_positions)) != N:
                vec.append(0.0)   # invariant violated -> reject defensively
                continue
            matched = len(set(final_positions) & set(target_cells))
            obj = matched / N
        except Exception:
            vec.append(0.0)
            continue

        denom = max(CEIL - base, 1e-9)
        r = 0.1 + 0.9 * (obj - base) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        r = max(0.0, min(1.0, r))
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
