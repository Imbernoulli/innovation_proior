#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_S_0631 -- "Bonsai Ledger: Pruning Between Visitor Seasons"
(family: rotation-policy-forge; format B, quality-metric).

THEME.  A bonsai garden displays its specimens along the search-path of a binary
search tree keyed by tag number: visitors ask the gardener to find tag k, and the
gardener walks pointers from the display root down to it.  Between look-ups the
gardener may PRUNE-AND-RESET a branch -- a single BST rotation that reshapes the
display -- to bring frequently-requested specimens closer to the entrance.  But
re-staging a branch takes real labor, and during a busy visitor SEASON every
prune costs far more (staff, foot traffic, safety rope) than during a quiet one.
The public trace tells the policy exactly how costly each moment's pruning is.

MECHANISM (this is NOT plain splaying).  Cost = 1 per pointer hop during the
walk to find the requested tag, PLUS `season_weight[i]` per rotation used to
restage the tree before that walk (hop-vs-restructure-accounting).  season_weight
varies through the trace (phase-aware-adaptation): some stretches are quiet
(weight 1, pruning is cheap) and some are peak season (weight up to 6, pruning is
expensive).  The trace also mixes visitor-behaviour regimes: tight repeat-visit
"regular clientele" phases (a small working set), one-pass processions (strictly
increasing tag order), pacing "echo walks" (alternating low/high tags), and
raffle-drawn (uniform random) order.  None of this structure is announced inline
-- a policy must infer both "is this a phase with repeat traffic?" and "is
pruning affordable right now?" from the trace itself, and decide, per visitor,
whether to rotate BEFORE searching (self-adjusting-rotation-rules).

TRAP.  A textbook "always splay the requested tag to the root" policy pays off
beautifully when pruning is free, but during a peak-season procession/echo-walk
(no repeat traffic, so restructuring buys nothing) it pays the season's inflated
per-rotation cost every single visitor for zero long-run benefit -- ruinous.  A
policy that never rotates is safe there but leaves huge, avoidable hop cost on
the table during cheap-season regular-clientele phases.  The winning policy
METERS its own amortized benefit online: it tracks recent-access statistics to
decide whether the *current phase* will repay a rotation's cost, i.e. it
implements a phase detector, not a fixed access rule.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "n": N,
             "left":  [N ints],  # left[k-1]  = left-child  tag of tag k, 0 = none
             "right": [N ints],  # right[k-1] = right-child tag of tag k, 0 = none
             "root": int,        # initial display root tag
             "season_weight": [M numbers],  # per-visit rotation-cost multiplier
             "accesses": [M ints]}          # the sequence of requested tags (1..N)
  stdout: ONE JSON object:
            {"ops": [[...], [...], ...]}    # length EXACTLY M
          ops[i] is the ordered list of tag keys to rotate-up, ONE AT A TIME,
          BEFORE visitor i's search.  `rotate_up(k)`: k trades places with its
          CURRENT parent via one standard BST single rotation (the side is
          implied: if k is a left child this is a right-rotation at the parent,
          and symmetrically) -- k's depth drops by exactly one.  Splaying a tag
          to the root is simply calling rotate_up on it repeatedly (its own
          depth many times); a real splay's zig-zig doubles instead rotate the
          PARENT first (see "Suggested strategies").  Rotating the current root
          is a legal no-op (costs its season weight, changes nothing).

  A layout is VALID iff `ops` has exactly M entries, each entry is a list of
  integers in [1,N] (booleans rejected), and the total rotation count across the
  whole answer does not exceed a generous safety cap.  Malformed output, a
  crash, a timeout, or non-JSON output makes that instance score 0.0.

## Scoring (deterministic)
For each instance the evaluator itself replays the visitor sequence against your
`ops`: it applies ops[i] (each op costs `season_weight[i]`), then walks from the
(possibly just-changed) root to `accesses[i]` and charges 1 per pointer hop.
`cost` = the sum of all these charges over the whole trace.  The reference
`baseline(inst)` is the cost of a policy that NEVER rotates (pure static-tree hop
cost).  Normalized (minimization, weak baseline anchored near 0.1):
    r = min(1, 0.1 * baseline / max(cost, 1e-12))
Beating the never-rotate baseline scores above 0.1; a wasteful policy that
rotates blindly through an expensive, repeat-free phase can score BELOW 0.1.
The reported Ratio is the mean r over 10 fixed seeded instances (several are
larger held-out cases); Vector lists the per-instance scores.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  All scoring
references are computed by THIS parent process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt_int


# ----------------------------- initial tree ---------------------------------
def _build_balanced(n):
    """Perfectly balanced BST over keys 1..N. Public schema: 0-indexed arrays of
    length n; left[k-1]/right[k-1] = child key of key k (0 = none)."""
    left = [0] * n
    right = [0] * n

    def rec(lo, hi):
        if lo > hi:
            return 0
        mid = (lo + hi) // 2
        left[mid - 1] = rec(lo, mid - 1)
        right[mid - 1] = rec(mid + 1, hi)
        return mid

    root = rec(1, n)
    return left, right, root


# ----------------------------- access-sequence generation -------------------
def _gen_segment(rng, pattern, n, length, ws_size=None):
    keys = []
    if pattern == "ws":
        # a tight repeat-visit "regular clientele": a small hot core (~88% of
        # visits) plus a slightly wider casual working set
        core = set()
        while len(core) < max(2, ws_size // 6):
            core.add(rng(1, n))
        ws = set(core)
        while len(ws) < ws_size:
            ws.add(rng(1, n))
        ws = list(ws); core = list(core)
        for _ in range(length):
            if rng(0, 99) < 88:
                keys.append(core[rng(0, len(core) - 1)])
            else:
                keys.append(ws[rng(0, len(ws) - 1)])
    elif pattern == "seq":  # one-pass procession, strictly increasing (wraps)
        s = rng(1, n)
        for t in range(length):
            keys.append(((s - 1 + t) % n) + 1)
    elif pattern == "rev":  # echo walk: alternating low/high ends
        for t in range(length):
            if t % 2 == 0:
                keys.append(1 + (t // 2) % n)
            else:
                keys.append(n - (t // 2) % n)
    elif pattern == "rnd":  # raffle-drawn order, no locality
        for _ in range(length):
            keys.append(rng(1, n))
    else:
        raise ValueError(pattern)
    return keys


def _build_instances():
    """10 fixed seeded traces: (name, seed, n, recipe). recipe = list of
    (pattern, season_weight, length, ws_size)."""
    specs = [
        ("courtyard_routine", 101, 150,
         [("ws", 1, 500, 8), ("rnd", 1, 300, None), ("ws", 1, 500, 8)]),
        ("spring_procession_peak", 202, 400,
         [("ws", 1, 900, 12), ("seq", 3, 350, None), ("ws", 1, 900, 12)]),
        ("echo_walk_peak", 303, 400,
         [("ws", 1, 900, 12), ("rev", 3, 350, None), ("ws", 1, 900, 12)]),
        ("raffle_peak", 404, 400,
         [("ws", 1, 950, 12), ("rnd", 2, 300, None), ("ws", 1, 950, 12)]),
        ("regulars_and_strangers", 505, 200,
         [("ws", 1, 600, 10), ("rnd", 1, 250, None), ("ws", 1, 600, 10)]),
        ("beloved_specimen_peak", 606, 250,
         [("ws", 2, 900, 7), ("rnd", 1, 400, None), ("ws", 1, 600, 7)]),
        ("storm_then_calm", 707, 300,
         [("rev", 3, 300, None), ("ws", 1, 1500, 9)]),
        ("whole_garden_tour", 808, 800,
         [("ws", 1, 1800, 18), ("seq", 3, 400, None), ("rnd", 2, 400, None), ("ws", 1, 1400, 18)]),
        ("small_pavilion", 909, 60,
         [("ws", 1, 1000, 6), ("rnd", 1, 400, None)]),
        ("grand_procession", 1010, 300,
         [("ws", 1, 950, 11), ("seq", 3, 500, None), ("ws", 1, 950, 11)]),
    ]
    out = []
    for name, seed, n, recipe in specs:
        rng = _rng(seed)
        left, right, root = _build_balanced(n)
        accesses = []
        weights = []
        for pattern, weight, length, ws_size in recipe:
            seg = _gen_segment(rng, pattern, n, length, ws_size)
            accesses.extend(seg)
            weights.extend([weight] * length)
        out.append({"name": name, "n": n, "left": left, "right": right, "root": root,
                    "season_weight": weights, "accesses": accesses})
    return out


# ----------------------------- tree mechanics (authoritative) ---------------
def _build_tree(inst):
    n = inst["n"]
    leftArr = inst["left"]; rightArr = inst["right"]
    parent = [0] * (n + 1)
    left = [0] * (n + 1); right = [0] * (n + 1)
    for k in range(1, n + 1):
        l = leftArr[k - 1]; r = rightArr[k - 1]
        left[k] = l; right[k] = r
        if l: parent[l] = k
        if r: parent[r] = k
    return {"left": left, "right": right, "parent": parent, "root": inst["root"]}


def _depth_of(tree, k):
    d = 0; cur = k
    par = tree["parent"]
    while par[cur] != 0:
        cur = par[cur]; d += 1
    return d


def _rotate_up(tree, k):
    """Single BST rotation: k trades places with its current parent p. Returns
    False (no-op, but caller still charges the attempted op) if k has no parent."""
    par = tree["parent"]; left = tree["left"]; right = tree["right"]
    p = par[k]
    if p == 0:
        return False
    g = par[p]
    if left[p] == k:
        b = right[k]
        left[p] = b
        if b: par[b] = p
        right[k] = p
    else:
        b = left[k]
        right[p] = b
        if b: par[b] = p
        left[k] = p
    par[p] = k
    par[k] = g
    if g == 0:
        tree["root"] = k
    else:
        if left[g] == p: left[g] = k
        else: right[g] = k
    return True


# ----------------------------- baseline + scoring ----------------------------
def baseline(inst):
    """Cost of the NEVER-ROTATE policy (pure static-tree hop cost)."""
    tree = _build_tree(inst)
    total = 0
    for key in inst["accesses"]:
        total += _depth_of(tree, key)
    return total


def score(inst, answer):
    """Validate + replay the candidate's ops against inst. Returns (True, cost)
    or (False, None)."""
    n = inst["n"]
    accesses = inst["accesses"]
    weights = inst["season_weight"]
    M = len(accesses)
    if not isinstance(answer, dict):
        return False, None
    ops = answer.get("ops")
    if not isinstance(ops, list) or len(ops) != M:
        return False, None
    CAP = 60 * M
    tree = _build_tree(inst)
    cost = 0.0
    total_ops = 0
    for i in range(M):
        oi = ops[i]
        if not isinstance(oi, list):
            return False, None
        for k in oi:
            if isinstance(k, bool) or not isinstance(k, int) or k < 1 or k > n:
                return False, None
        total_ops += len(oi)
        if total_ops > CAP:
            return False, None
        cost += weights[i] * len(oi)
        for k in oi:
            _rotate_up(tree, k)
        cost += _depth_of(tree, accesses[i])
    return True, cost


# ----------------------------- driver ---------------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        b = baseline(inst)
        public = {"name": inst["name"], "n": inst["n"], "left": list(inst["left"]),
                  "right": list(inst["right"]), "root": inst["root"],
                  "season_weight": list(inst["season_weight"]), "accesses": list(inst["accesses"])}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok, obj = False, None
        if not ok:
            vec.append(0.0)
            continue
        r = min(1.0, 0.1 * b / max(obj, 1e-12))
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
