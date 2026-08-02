#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_1297 -- "Almost-No-Memory Relay Courier"
(family: grid-robot-program; format B, quality-metric).

THEME.  A tiny courier robot patrols corridor-mazes.  It has essentially NO
memory: at every tick it can sense only the landmark token printed on the
FLOOR TILE it is currently standing on, and it carries only a small integer
"mode" register (its finite-state-controller state).  Floor tiles are painted
with one of four abstract colors (R/G/B/Y) whose meaning as a *compass
direction* depends on the robot's current mode -- a fixed, globally-known
color code (given below and echoed verbatim in the public instance).  Partway
along its route sits a RELAY beacon tile; the instant the robot steps onto it,
its onboard decoder recalibrates (mode flips), so the SAME four colors are
reinterpreted as different compass directions for the remainder of the run.

CANDIDATE CONTRACT (isolated stdin -> stdout program, ONE call total).
  stdin : ONE JSON "public instance": global config (action set, state
          budget, the two color->direction codes, max_steps) plus a handful
          of fully-visible "training" grids.  A held-out grid COUNT is given,
          but held-out grid layouts are NEVER shown to the candidate.
  stdout: ONE JSON "controller":
            {"start_state": int, "rules": [{"state":int,"see":str,
                                             "action":str,"next":int}, ...]}
          i.e. a small Mealy machine: (mode, sensed token) -> (action, mode').
          Missing (state, token) pairs default to WAIT (stay put) -- a
          candidate need only cover the tokens it actually cares about.

The evaluator (THIS process, never the candidate) then SIMULATES that ONE
submitted controller -- deterministically, no candidate code involved -- on
ALL 10 grids in the family (the visible ones AND 7 additional held-out grids
of the same kind the candidate never saw), and scores generalization.

TRAP.  Because the visible grids are given IN FULL, the path of least
resistance is: run a shortest-path search directly on the visible grid's
raw coordinates and hardcode the resulting action sequence as a chain of
one-shot states (state i, whatever i sees) -> (move i, i+1).  This
reproduces the visible grid almost perfectly (and needs only ~len(path)
distinct states) -- but that exact chain is meaningless on every OTHER
grid, where the tokens encountered at each state index differ, so it
stalls at WAIT almost immediately: near-zero score on 9 of the 10 grids
plus a large-state-count SIZE PENALTY (program-size-budget).  The insight
(innovation hook) is to decode the *given* color code into a compact,
token-triggered state machine (finite-state-controller + landmark
recognition) that needs only ~2 modes and a couple dozen rules and
therefore treats every grid -- visible or held-out -- identically.

SCORING (deterministic; per grid g in [0,1], see _grid_score):
    reached beacon              -> partial credit floor 0.35
    reached goal                -> 0.60 + 0.40 * clamp(opt_steps/steps_used)
    controller SIZE penalty (program-size-budget): size_factor =
        1.0                         if rule_count <= REF_SIZE
        max(SIZE_FLOOR, REF_SIZE / rule_count)   otherwise
    v = clamp(FLOOR + SCALE * g * size_factor, 0, 1)   (affine anchor, headroom)
  Ratio = mean(v) over the 10 grids.  A structurally malformed / out-of-budget
  / crashing controller scores the WHOLE Ratio 0.0 (no floor credit).

ISOLATION.  The single candidate call goes through isorun.run_candidate,
a FRESH OS-sandboxed subprocess that only ever sees the public instance;
grid layouts for held-out instances, the color-code inversion tables, and
the simulator itself never touch candidate-reachable memory.

CLI:  python3 evaluator.py <controller_writer.py>
Prints:
  Ratio: <mean v over 10 grids, in [0,1]>
  Vector: [v_1, ..., v_10]
"""
import sys, json
import isorun

# --------------------------------------------------------------------------
# deterministic RNG (splitmix-ish LCG, matches house style)
# --------------------------------------------------------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt_int


# --------------------------------------------------------------------------
# fixed global color codes (mode 0 = pre-relay, mode 1 = post-relay)
# --------------------------------------------------------------------------
PHASE0_CODE = {"R": "N", "G": "E", "B": "S", "Y": "W"}
PHASE1_CODE = {"R": "E", "G": "S", "B": "W", "Y": "N"}
INV_PHASE0 = {v: k for k, v in PHASE0_CODE.items()}
INV_PHASE1 = {v: k for k, v in PHASE1_CODE.items()}
DELTA = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1), "WAIT": (0, 0)}
DIRS = ["N", "S", "E", "W"]
ACTIONS = ["N", "S", "E", "W", "WAIT"]

STATE_BUDGET = 40
MAX_RULES = 600
REF_SIZE = 14
SIZE_FLOOR = 0.15
V_FLOOR = 0.05
V_SCALE = 0.80


# --------------------------------------------------------------------------
# maze / path generation -- self-avoiding random walk with clearance, then
# a couple of dead-end decoy spurs.  Fully deterministic given `seed`.
# --------------------------------------------------------------------------
def _shuffle(lst, ni):
    out = list(lst)
    for i in range(len(out) - 1, 0, -1):
        j = ni(0, i)
        out[i], out[j] = out[j], out[i]
    return out


def _neighbors_clear(cell, visited, exclude):
    """True if `cell` has no visited orthogonal neighbor other than `exclude`
    (keeps the corridor one-cell wide / non-self-adjacent)."""
    r, c = cell
    for dr, dc in DELTA.values():
        if (dr, dc) == (0, 0):
            continue
        n = (r + dr, c + dc)
        if n != exclude and n in visited:
            return False
    return True


def _gen_path(seed, target_len):
    ni = _rng(seed)
    pos = (0, 0)
    path = [pos]
    visited = {pos}
    last_dir = None
    guard = 0
    max_guard = target_len * 60
    while len(path) < target_len and guard < max_guard:
        guard += 1
        if last_dir is not None and ni(0, 99) < 68:
            order = [last_dir] + _shuffle([d for d in DIRS if d != last_dir], ni)
        else:
            order = _shuffle(DIRS, ni)
        moved = False
        for d in order:
            dr, dc = DELTA[d]
            np_ = (pos[0] + dr, pos[1] + dc)
            if np_ in visited:
                continue
            if not _neighbors_clear(np_, visited, pos):
                continue
            pos = np_
            path.append(pos)
            visited.add(pos)
            last_dir = d
            moved = True
            break
        if not moved:
            if len(path) > 3:
                popped = path.pop()
                visited.discard(popped)
                pos = path[-1]
                last_dir = None
            else:
                break
    return path, visited


def _dir_between(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    for d, (ddr, ddc) in DELTA.items():
        if (ddr, ddc) == (dr, dc):
            return d
    raise ValueError("non-adjacent cells")


def _add_spurs(path, visited, seed, n_spurs, spur_len):
    """Dead-end decoy branches off interior path cells (never on the true
    route, colored arbitrarily -- a controller that ever strays here just
    wastes steps, it is never the *correct* thing to follow)."""
    ni = _rng(seed)
    spurs = []  # list of list[(r,c)]
    n = len(path)
    if n < 8:
        return spurs
    candidates = list(range(2, n - 2))
    tries = 0
    picked = set()
    while len(spurs) < n_spurs and tries < n_spurs * 25 and candidates:
        tries += 1
        idx = candidates[ni(0, len(candidates) - 1)]
        if idx in picked:
            continue
        picked.add(idx)
        base = path[idx]
        order = _shuffle(DIRS, ni)
        for d in order:
            dr, dc = DELTA[d]
            cur = base
            branch = []
            ok = True
            for _ in range(spur_len):
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt in visited or not _neighbors_clear(nxt, visited, cur):
                    ok = False
                    break
                branch.append(nxt)
                visited.add(nxt)
                cur = nxt
            if ok and branch:
                spurs.append(branch)
                break
            else:
                for cell in branch:
                    visited.discard(cell)
    return spurs


def _build_grid(seed, target_len, n_spurs, spur_len, pad=2):
    path, visited = _gen_path(seed, target_len)
    if len(path) < max(8, target_len // 2):
        # regenerate with a bumped seed until we get a long-enough corridor
        return _build_grid(seed * 2654435761 + 1, target_len, n_spurs, spur_len, pad)
    spurs = _add_spurs(path, visited, seed + 999983, n_spurs, spur_len)

    key_idx = max(1, min(len(path) - 2, round(0.42 * (len(path) - 1))))

    rs = [p[0] for p in path] + [c[0] for s in spurs for c in s]
    cs = [p[1] for p in path] + [c[1] for s in spurs for c in s]
    min_r, max_r = min(rs), max(rs)
    min_c, max_c = min(cs), max(cs)
    H = (max_r - min_r) + 1 + 2 * pad
    W = (max_c - min_c) + 1 + 2 * pad
    off_r, off_c = pad - min_r, pad - min_c
    grid = [["#"] * W for _ in range(H)]

    def shift(cell):
        return (cell[0] + off_r, cell[1] + off_c)

    for i, cell in enumerate(path):
        r, c = shift(cell)
        if i == len(path) - 1:
            grid[r][c] = "X"
        elif i == key_idx:
            out_dir = _dir_between(path[i], path[i + 1])
            grid[r][c] = "K_" + INV_PHASE1[out_dir]
        elif i < key_idx:
            out_dir = _dir_between(path[i], path[i + 1])
            grid[r][c] = INV_PHASE0[out_dir]
        else:
            out_dir = _dir_between(path[i], path[i + 1])
            grid[r][c] = INV_PHASE1[out_dir]

    ci = _rng(seed + 424242)
    palette = ["R", "G", "B", "Y"]
    for branch in spurs:
        for cell in branch:
            r, c = shift(cell)
            grid[r][c] = palette[ci(0, 3)]

    start = shift(path[0])
    return {
        "width": W, "height": H,
        "start": [start[0], start[1]],
        "grid": grid,
        "opt_steps": len(path) - 1,
        "key_idx": key_idx,
        "path_len": len(path),
    }


# --------------------------------------------------------------------------
# the 10-grid family: 3 visible ("training"), 7 held-out.  Sizes grow for the
# held-out set to also test scaling, not just blind transfer.
# --------------------------------------------------------------------------
_SPECS = [
    # (seed, target_len, n_spurs, spur_len, visible?)
    (2027, 22, 2, 3, True),
    (3301, 27, 2, 3, True),
    (4111, 30, 3, 3, True),
    (5501, 18, 1, 2, False),
    (6007, 24, 2, 3, False),
    (7207, 29, 3, 3, False),
    (8009, 33, 3, 4, False),
    (9103, 26, 2, 3, False),
    (10501, 36, 3, 4, False),
    (11701, 40, 4, 4, False),
]


def _build_instances():
    out = []
    for seed, tl, ns, sl, vis in _SPECS:
        g = _build_grid(seed, tl, ns, sl)
        g["visible"] = vis
        g["name"] = f"relay{seed}"
        out.append(g)
    return out


MAX_STEPS = 260


# --------------------------------------------------------------------------
# controller validation + simulation (pure evaluator-side computation; no
# candidate code executes here -- this only interprets DATA the candidate
# returned, so it is safe to run in-process)
# --------------------------------------------------------------------------
def _validate_controller(answer):
    if not isinstance(answer, dict):
        return None
    ss = answer.get("start_state")
    if isinstance(ss, bool) or not isinstance(ss, int):
        return None
    if not (0 <= ss < STATE_BUDGET):
        return None
    rules = answer.get("rules")
    if not isinstance(rules, list) or len(rules) > MAX_RULES:
        return None
    lookup = {}
    for rule in rules:
        if not isinstance(rule, dict):
            return None
        st, see, act, nxt = rule.get("state"), rule.get("see"), rule.get("action"), rule.get("next")
        if isinstance(st, bool) or not isinstance(st, int) or not (0 <= st < STATE_BUDGET):
            return None
        if isinstance(nxt, bool) or not isinstance(nxt, int) or not (0 <= nxt < STATE_BUDGET):
            return None
        if not isinstance(see, str) or len(see) > 32:
            return None
        if act not in ACTIONS:
            return None
        lookup[(st, see)] = (act, nxt)
    return {"start_state": ss, "lookup": lookup, "rule_count": len(rules)}


def _simulate(ctrl, inst):
    grid = inst["grid"]; H, W = inst["height"], inst["width"]
    r, c = inst["start"]
    state = ctrl["start_state"]
    lookup = ctrl["lookup"]
    beacon = False
    goal = False
    steps = 0
    for _ in range(MAX_STEPS + 1):
        tok = grid[r][c]
        if tok.startswith("K_"):
            beacon = True
        if tok == "X":
            goal = True
            break
        if steps >= MAX_STEPS:
            break
        act, nxt = lookup.get((state, tok), ("WAIT", state))
        state = nxt
        dr, dc = DELTA.get(act, (0, 0))
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] != "#":
            r, c = nr, nc
        steps += 1
    return beacon, goal, steps


def _grid_score(beacon, goal, steps, opt_steps):
    if not beacon:
        return 0.0
    if not goal:
        return 0.35
    eff = 1.0 if steps <= 0 else opt_steps / steps
    eff = max(0.0, min(1.0, eff))
    return 0.60 + 0.40 * eff


def _size_factor(rule_count):
    if rule_count <= REF_SIZE:
        return 1.0
    return max(SIZE_FLOOR, REF_SIZE / rule_count)


def _public_instance(instances):
    visible = []
    for inst in instances:
        if inst["visible"]:
            visible.append({
                "name": inst["name"], "width": inst["width"], "height": inst["height"],
                "start": inst["start"], "grid": inst["grid"],
            })
    n_holdout = sum(1 for inst in instances if not inst["visible"])
    return {
        "state_budget": STATE_BUDGET,
        "actions": ACTIONS,
        "phase0_code": PHASE0_CODE,
        "phase1_code": PHASE1_CODE,
        "max_steps": MAX_STEPS,
        "n_total_grids": len(instances),
        "n_holdout": n_holdout,
        "visible_grids": visible,
    }


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <controller_writer.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()
    public = _public_instance(instances)

    ans, st = isorun.run_candidate(cand, public, timeout=20)
    ctrl = _validate_controller(ans) if st == "OK" else None

    vec = []
    if ctrl is None:
        vec = [0.0] * len(instances)
    else:
        sf = _size_factor(ctrl["rule_count"])
        for inst in instances:
            beacon, goal, steps = _simulate(ctrl, inst)
            g = _grid_score(beacon, goal, steps, inst["opt_steps"])
            v = V_FLOOR + V_SCALE * g * sf
            if v < 0.0:
                v = 0.0
            elif v > 1.0:
                v = 1.0
            vec.append(v)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
