#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_S_0691 -- "Wound-Ring: A Regenerating Cellular Organism"
(family: regenerating-ca-organism; format B, quality-metric; theme: a creature that
regrows what you cut off).

THEME.  A creature's tissue lives on a small grid.  Every cell is a finite-state
automaton with a von Neumann neighborhood (self, North, South, East, West).  The
creature starts as a SINGLE seed cell and must, under its OWN local update rule
applied ASYNCHRONOUSLY (mechanism 1: cells update one at a time, in a scrambled but
deterministic per-tick order, each update immediately visible to the rest of that
tick's sweep -- Gauss-Seidel style, not a synchronous snapshot-swap), grow into a
specific target body plan and hold it steady.  The evaluator then cuts a rectangular
wound out of the settled body (resets those cells to empty) and continues applying
the SAME rule: mechanism 3 (self-repair) is whether the wound regrows correctly.

INNOVATION HOOK / mechanism 2 (positional information).  A cell's local rule NEVER
sees its own (row, col) -- only the five states in its neighborhood.  The target body
plan is defined by true shortest-path ("BFS") distance from the seed through the
live tissue (obstacles/walls block paths, so the body can be a maze-like, branching
organism, not just a disk).  distance IS the positional coordinate the organism must
maintain: a cell's correct identity ("ring" state) equals 1 + the distance of its
nearest already-correct neighbor.  A rule that satisfies this INVARIANT everywhere is
self-stabilizing (Bellman-Ford-style relaxation: order-independent, converges under
ANY asynchronous sweep) and heals ANY wound, because surviving tissue around a hole
still carries correct distance values and the invariant re-derives the missing ones
from local context alone -- true positional information, reconstructed, not just
copied.  A rule that instead "copies-and-freezes" (memorizes per-pattern transitions
from a single forward growth trace, then never revisits an already-alive cell) can
often *grow* the shape correctly the first time (a monotone unidirectional wave from
a single source usually offers only the truly-nearest neighbor when a cell first
activates) -- but on repair, a reopened wound's surviving boundary offers MULTIPLE
neighbors of different distances simultaneously, in an order that does not match the
original growth trace, so the frozen/memorized rule mis-derives or fully misses the
correct distance and the wound heals wrong or not at all.  This is the seed's
"copy-and-freeze plateaus low" trap, made concrete.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
    {"name": str, "H": int, "W": int, "seed": [r, c], "seed_state": int,
     "K": int, "wall_state": int, "walls": [[r,c], ...],
     "target": [[int]*W]*H,                 # desired FINAL state per cell
     "n_growth_ticks": int, "n_repair_ticks": int, "max_table_entries": int}
  stdout: ONE JSON object -- the candidate's LOCAL UPDATE RULE:
    {"table": {"c,n,s,e,w": next_state, ...}, "default": "stay" | int}
    Keys are literal strings "c,n,s,e,w" (5 comma-joined ints in [0,K-1]; c=self,
    n/s/e/w = North/South/East/West neighbor state, off-grid neighbors = wall_state).
    "default" is applied to any 5-tuple absent from "table": either the literal
    string "stay" (next state = current self state) or a fixed int in [0,K-1].
    VALID iff: table is a dict of at most max_table_entries entries, every key parses
    to exactly 5 ints each in [0,K-1], every value is an int in [0,K-1]; default is
    "stay" or an int in [0,K-1].  Any violation, non-JSON, crash, or timeout -> 0.0.

TRANSITION (deterministic, ASYNCHRONOUS).  Wall cells never change.  Each tick visits
every non-wall cell in a scrambled-but-seeded order; the cell's CURRENT 5-tuple
(self + 4 neighbors, off-grid = wall_state) is looked up in the table (falling back
to "default") and the cell is updated IN PLACE immediately (visible to the rest of
this tick's sweep -- this is what "asynchronous" buys a well-designed rule: multi-hop
relaxation within a single tick).

EPISODE.  (1) GROW: start from a grid of all-empty except the seed cell (seed_state)
and the fixed walls; run n_growth_ticks ticks; measure growth fidelity against target
over a padded bounding box of the body (excluding wall cells).  (2) REPAIR: from the
grown grid, for each of several HIDDEN rectangular wounds (excision rectangles whose
location/size the candidate never sees -- forces a general rule, not one special-cased
to a known cut), reset those cells to empty, run n_repair_ticks ticks, and measure
repair fidelity restricted to the wound's own cells.  Per-instance objective:
    obj = 0.3 * growth_fidelity + 0.7 * mean(repair_fidelity over wounds)
Score: r = clamp(0.1 + 0.8 * (obj - obj_base) / (1 - obj_base), 0, 1), where obj_base
is the "do nothing" rule's (empty table, default "stay") objective on the SAME
instance -- do-nothing scores ~0.1, and the 0.8 coefficient caps the theoretical
ceiling at r=0.9 < 1, leaving headroom even for a hypothetically perfect rule.  Final
score = mean r over 10 fixed seeded instances (open disks, obstacle mazes, branching
multi-limb bodies, edge-clipped bodies, and a larger held-out maze).

ISOLATION.  The candidate is untrusted and runs in a FRESH sandboxed subprocess via
`isorun.run_candidate`; it only ever sees the PUBLIC instance above (the wound
rectangles and the growth/repair simulator live only in this parent process).

CLI: python3 evaluator.py <solution.py>
"""
import sys, json
from collections import deque
import isorun

# ----------------------------- constants -----------------------------------
H = 15
W = 15
K = 9                 # states 0..8
WALL_STATE = K - 1     # 8
SEED_STATE = 1
GMAX = K - 2           # max ring value = 7 -> max distance = 6
N_GROWTH_TICKS = 26
N_REPAIR_TICKS = 20
MAX_TABLE_ENTRIES = 20000
GROWTH_W = 0.3
REPAIR_W = 0.7


# ----------------------------- deterministic RNG ----------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt_int


def _sweep_order(rng_next, cells):
    order = list(cells)
    for i in range(len(order) - 1, 0, -1):
        j = rng_next(0, i)
        order[i], order[j] = order[j], order[i]
    return order


# ----------------------------- shape / target generation -------------------
def _bfs_target(seed, walls_set):
    dist = [[-1] * W for _ in range(H)]
    sr, sc = seed
    dist[sr][sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        d = dist[r][c]
        if d >= GMAX:
            continue
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in walls_set and dist[nr][nc] == -1:
                dist[nr][nc] = d + 1
                q.append((nr, nc))
    target = [[0] * W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            if (r, c) in walls_set:
                target[r][c] = WALL_STATE
            elif dist[r][c] != -1:
                target[r][c] = dist[r][c] + 1
    return target, dist


def _gen_walls(base_seed, seed_pos, n_segments, seg_len_range):
    for attempt in range(60):
        rng_next = _rng(base_seed * 1000003 + attempt * 97 + 7)
        walls = set()
        for _ in range(n_segments):
            r0 = rng_next(1, H - 2)
            c0 = rng_next(1, W - 2)
            length = rng_next(*seg_len_range)
            horiz = rng_next(0, 1) == 0
            for k in range(length):
                r = r0 + (0 if horiz else k)
                c = c0 + (k if horiz else 0)
                if 0 <= r < H and 0 <= c < W and (r, c) != tuple(seed_pos):
                    walls.add((r, c))
        target, dist = _bfs_target(seed_pos, walls)
        mask_size = sum(1 for r in range(H) for c in range(W) if 0 < target[r][c] < WALL_STATE)
        if mask_size >= 40:
            return target, dist, walls
    target, dist = _bfs_target(seed_pos, set())
    return target, dist, set()


def _choose_wounds(base_seed, seed_pos, target, dist, specs):
    rng_next = _rng(base_seed * 7919 + 13)
    mask_cells = [(r, c) for r in range(H) for c in range(W)
                  if 0 < target[r][c] < WALL_STATE and dist[r][c] >= 2]
    if not mask_cells:
        mask_cells = [tuple(seed_pos)]
    wounds = []
    for (hh, ww) in specs:
        best = None
        for _tr in range(300):
            cr, cc = mask_cells[rng_next(0, len(mask_cells) - 1)]
            r0 = max(0, min(H - hh, cr - hh // 2))
            c0 = max(0, min(W - ww, cc - ww // 2))
            cells = [(r, c) for r in range(r0, r0 + hh) for c in range(c0, c0 + ww)]
            alive = sum(1 for (r, c) in cells if 0 < target[r][c] < WALL_STATE)
            has_wall = any(target[r][c] == WALL_STATE for (r, c) in cells)
            if not has_wall and alive >= max(3, int(0.55 * hh * ww)):
                best = {"r0": r0, "c0": c0, "h": hh, "w": ww}
                break
            if best is None:
                best = {"r0": r0, "c0": c0, "h": hh, "w": ww}
        wounds.append(best)
    return wounds


SPECS = [
    # name, base_seed, seed_pos, n_segments, seg_len_range, wound_specs
    ("open1", 9001, (7, 7), 0, (2, 2), [(3, 3), (2, 3)]),
    ("open2", 9002, (5, 8), 0, (2, 2), [(3, 3), (3, 2)]),
    ("corridor1", 9003, (7, 7), 3, (3, 5), [(3, 3), (2, 2)]),
    ("corridor2", 9004, (6, 6), 4, (3, 6), [(4, 3), (2, 2)]),
    ("branchy1", 9005, (7, 7), 5, (2, 4), [(3, 3), (3, 3)]),
    ("branchy2", 9006, (7, 4), 6, (2, 5), [(3, 4), (2, 2)]),
    ("edge1", 9007, (3, 7), 2, (2, 4), [(3, 3), (2, 3)]),
    ("edge2", 9008, (7, 11), 3, (2, 4), [(3, 3), (3, 2)]),
    ("severed_limb", 9009, (7, 7), 7, (2, 5), [(2, 4), (4, 2)]),
    ("heldout_big", 9010, (7, 7), 8, (2, 6), [(4, 4), (3, 3)]),
]


def _build_instances():
    out = []
    for name, base_seed, seed_pos, n_seg, seg_len, wound_specs in SPECS:
        target, dist, walls = _gen_walls(base_seed, seed_pos, n_seg, seg_len)
        wounds = _choose_wounds(base_seed, seed_pos, target, dist, wound_specs)
        out.append({
            "name": name, "seed": list(seed_pos), "walls": sorted(walls),
            "target": target, "wounds": wounds, "sim_seed": base_seed * 2654435761 % (1 << 31),
        })
    return out


# ----------------------------- simulation -----------------------------------
def _init_grid(inst):
    g = [[0] * W for _ in range(H)]
    for (r, c) in inst["walls"]:
        g[r][c] = WALL_STATE
    sr, sc = inst["seed"]
    g[sr][sc] = SEED_STATE
    return g


def _run_ticks(grid, walls_set, table, default, ticks, sim_seed):
    rng_next = _rng(sim_seed)
    cells = [(r, c) for r in range(H) for c in range(W) if (r, c) not in walls_set]
    for t in range(ticks):
        order = _sweep_order(rng_next, cells)
        for (r, c) in order:
            self_s = grid[r][c]
            n = grid[r - 1][c] if r - 1 >= 0 else WALL_STATE
            s = grid[r + 1][c] if r + 1 < H else WALL_STATE
            e = grid[r][c + 1] if c + 1 < W else WALL_STATE
            w = grid[r][c - 1] if c - 1 >= 0 else WALL_STATE
            key = (self_s, n, s, e, w)
            if key in table:
                nv = table[key]
            else:
                nv = self_s if default == "stay" else default
            grid[r][c] = nv
    return grid


def _scored_region(target):
    cells = [(r, c) for r in range(H) for c in range(W) if 0 < target[r][c] < WALL_STATE]
    if not cells:
        return [(r, c) for r in range(H) for c in range(W) if target[r][c] != WALL_STATE]
    rmin = max(0, min(r for r, c in cells) - 2)
    rmax = min(H - 1, max(r for r, c in cells) + 2)
    cmin = max(0, min(c for r, c in cells) - 2)
    cmax = min(W - 1, max(c for r, c in cells) + 2)
    return [(r, c) for r in range(rmin, rmax + 1) for c in range(cmin, cmax + 1)
            if target[r][c] != WALL_STATE]


def _fidelity(grid, target, cells):
    if not cells:
        return 1.0
    ok = sum(1 for (r, c) in cells if grid[r][c] == target[r][c])
    return ok / len(cells)


def _episode_objective(inst, table, default):
    target = inst["target"]
    walls_set = set(tuple(x) for x in inst["walls"])
    grid = _init_grid(inst)
    grid = _run_ticks(grid, walls_set, table, default, N_GROWTH_TICKS, inst["sim_seed"])
    growth_region = _scored_region(target)
    growth_fid = _fidelity(grid, target, growth_region)

    repair_fids = []
    for wi, wd in enumerate(inst["wounds"]):
        gcopy = [row[:] for row in grid]
        rect = [(r, c) for r in range(wd["r0"], wd["r0"] + wd["h"])
                for c in range(wd["c0"], wd["c0"] + wd["w"]) if (r, c) not in walls_set]
        for (r, c) in rect:
            gcopy[r][c] = 0
        gcopy = _run_ticks(gcopy, walls_set, table, default, N_REPAIR_TICKS,
                            inst["sim_seed"] + 1000 * (wi + 1) + 31)
        repair_fids.append(_fidelity(gcopy, target, rect))
    repair_mean = sum(repair_fids) / len(repair_fids) if repair_fids else 0.0
    return GROWTH_W * growth_fid + REPAIR_W * repair_mean


# ----------------------------- answer validation -----------------------------
def _parse_table(answer, K_):
    if not isinstance(answer, dict):
        return None
    raw_table = answer.get("table")
    default = answer.get("default")
    if not isinstance(raw_table, dict) or len(raw_table) > MAX_TABLE_ENTRIES:
        return None
    if default != "stay":
        if isinstance(default, bool) or not isinstance(default, int):
            return None
        if not (0 <= default <= K_ - 1):
            return None
    table = {}
    for k, v in raw_table.items():
        if not isinstance(k, str):
            return None
        parts = k.split(",")
        if len(parts) != 5:
            return None
        try:
            ints = [int(p) for p in parts]
        except Exception:
            return None
        if any(not (0 <= iv <= K_ - 1) for iv in ints):
            return None
        if isinstance(v, bool) or not isinstance(v, int):
            return None
        if not (0 <= v <= K_ - 1):
            return None
        table[tuple(ints)] = v
    return table, default


def score(inst, answer):
    parsed = _parse_table(answer, K)
    if parsed is None:
        return False, 0.0
    table, default = parsed
    obj = _episode_objective(inst, table, default)
    if not (obj == obj) or obj in (float("inf"), float("-inf")):
        return False, 0.0
    return True, obj


# ----------------------------- scoring driver -------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        obj_base = _episode_objective(inst, {}, "stay")
        public = {
            "name": inst["name"], "H": H, "W": W,
            "seed": inst["seed"], "seed_state": SEED_STATE,
            "K": K, "wall_state": WALL_STATE,
            "walls": [list(x) for x in inst["walls"]],
            "target": inst["target"],
            "n_growth_ticks": N_GROWTH_TICKS, "n_repair_ticks": N_REPAIR_TICKS,
            "max_table_entries": MAX_TABLE_ENTRIES,
        }
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0)
            continue
        denom = 1.0 - obj_base
        if denom <= 1e-9:
            r = 0.1
        else:
            r = 0.1 + 0.8 * (obj - obj_base) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            r = 0.0
        r = max(0.0, min(1.0, r))
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
