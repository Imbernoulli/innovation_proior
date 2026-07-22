#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for the aTAM binary-
counter tile-minimization problem.

<in>  : one line, integer n.
<out> : the participant's tile system:
            K
            id  Nlabel Nstr  Slabel Sstr  Elabel Estr  Wlabel Wstr  value
            ... (K lines, ids exactly 0..K-1 in any order)
            seed_id
        label is a token matching [A-Za-z0-9_]{1,32} or the sentinel "."
        (meaning "no glue" -- must be paired with strength 0). strength is an
        integer in {0,1,2}; a non-"." label must have strength in {1,2}, and
        every occurrence of the SAME label text anywhere in the tile set must
        use the SAME strength (a glue's strength is a property of the glue,
        not of the side that happens to carry it). value is 0 or 1: the bit
        this tile type represents wherever it is placed. seed_id selects
        which declared type is pre-placed at grid position (row=0, col=0);
        its value must be 0 (target bit at the origin is 0).

Grid convention: row = bit index b (0 = LSB .. W-1 = MSB), col = counter
value c (0 .. n-1). N = row+1, S = row-1, E = col+1, W = col-1. Temperature
tau = 2: an empty cell may receive a declared tile type iff the sum, over its
occupied N/S/E/W neighbors, of "neighbor's facing glue strength" for sides
whose glue LABELS match, is >= 2. If more than one declared type is eligible
at some empty frontier cell at some point of the (canonical, synchronous-wave)
simulation, the tile system is nondeterministic and the submission is
infeasible (Ratio 0) -- this mirrors the standard aTAM "directedness"
requirement and keeps the simulator itself bit-for-bit deterministic.

Feasibility: after simulation stabilizes (or a generous, size-independent
step/cell cap is hit -- growth outside the checked window is never an error,
it is simply not inspected), EVERY cell (b,c) with 0<=b<W, 0<=c<n must be
occupied by a tile whose declared value equals (c >> b) & 1.

Objective (minimize): F = K (number of declared tile types). The checker's
own baseline B = n*W (one tile type per target cell -- always a valid,
trivially-correct construction). Ratio = min(1, 0.1*B/F), i.e.
    Ratio = min(1.0, n*W / (10*K))
so a per-cell tiling scores ~0.1, and every real reduction in tile-type count
raises the score monotonically, capped below 1.0 for headroom.
"""
import re
import sys

LABEL_RE = re.compile(r"^[A-Za-z0-9_]{1,32}$")
DIRS = ("N", "S", "E", "W")
OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
DELTA = {"N": (1, 0), "S": (-1, 0), "E": (0, 1), "W": (0, -1)}
MAX_K = 2000
MAX_OUT_BYTES = 3_000_000


def die(reason):
    print("INVALID: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    if len(sys.argv) < 3:
        die("bad invocation")
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path, "r") as f:
        in_toks = f.read().split()
    if not in_toks:
        die("empty input")
    n = int(in_toks[0])
    W = max(1, (n - 1).bit_length()) if n >= 2 else 1

    try:
        import os
        if os.path.getsize(out_path) > MAX_OUT_BYTES:
            die("output too large")
        with open(out_path, "rb") as f:
            raw_bytes = f.read()
        raw = raw_bytes.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        die("cannot read output (not valid UTF-8, or unreadable)")

    toks = raw.split()
    pos = 0

    def next_tok():
        nonlocal pos
        if pos >= len(toks):
            die("unexpected end of output")
        t = toks[pos]
        pos += 1
        return t

    def next_int(lo, hi):
        t = next_tok()
        if len(t) > 18:
            die("integer token too long")
        try:
            v = int(t)
        except ValueError:
            die("expected an integer, got %r" % t)
        if v != v or v in (float("inf"), float("-inf")):
            die("non-finite value")
        if v < lo or v > hi:
            die("integer %d out of range [%d,%d]" % (v, lo, hi))
        return v

    K = next_int(1, MAX_K)

    label_strength = {}

    def read_side():
        lbl = next_tok()
        st = next_int(0, 2)
        if lbl == ".":
            if st != 0:
                die("label '.' (no glue) must have strength 0")
        else:
            if not LABEL_RE.match(lbl):
                die("malformed glue label %r" % lbl)
            if st == 0:
                die("a real glue label must have strength 1 or 2, got 0 (%r)" % lbl)
            prev = label_strength.get(lbl)
            if prev is None:
                label_strength[lbl] = st
            elif prev != st:
                die("glue label %r used with inconsistent strengths (%d vs %d)" % (lbl, prev, st))
        return (lbl, st)

    tiles = {}
    seen_ids = set()
    for _ in range(K):
        tid = next_int(0, K - 1)
        if tid in seen_ids:
            die("duplicate tile id %d" % tid)
        seen_ids.add(tid)
        sides = {d: read_side() for d in DIRS}
        val = next_int(0, 1)
        tiles[tid] = {"sides": sides, "value": val}
    if len(seen_ids) != K:
        die("tile ids must be exactly 0..K-1")

    seed_id = next_int(0, K - 1)
    if pos != len(toks):
        die("trailing tokens after seed id")
    if tiles[seed_id]["value"] != 0:
        die("seed tile's value must be 0 (target bit at the origin is 0)")

    # ---- bounded canonical synchronous-wave simulation ----
    # The checked window is (0<=row<W, 0<=col<n), but real aTAM growth has no
    # implicit boundary at 0 -- a submitted tile system may (and its
    # directedness must be checked as if it could) grow south of row 0 or
    # west of col 0 too. Bound the simulation symmetrically around the
    # window with a generous, size-independent buffer on every side so
    # ambiguity is never silently ignored just because it would occur
    # "behind" the seed.
    buf = max(W, 4) + 4
    row_lo, row_hi = -buf, W + buf
    col_lo, col_hi = -buf, n + buf
    max_cells = (row_hi - row_lo) * (col_hi - col_lo)
    max_waves = (row_hi - row_lo) + (col_hi - col_lo) + 20

    def matches(tid, r, c, occ_state, exclude=frozenset()):
        """Neighbor positions of (r,c) (present in occ_state, not in
        `exclude`) whose facing glue matches tid's side -- and the total
        strength they contribute."""
        contribs = []
        total = 0
        for d, (dr, dc) in DELTA.items():
            nb = (r + dr, c + dc)
            if nb not in occ_state or nb in exclude:
                continue
            my_lbl, my_str = tiles[tid]["sides"][d]
            nb_lbl, _ = tiles[occ_state[nb]]["sides"][OPP[d]]
            if my_lbl != "." and my_lbl == nb_lbl:
                contribs.append(nb)
                total += my_str
        return contribs, total

    def eligible(r, c, occ_state, exclude=frozenset()):
        """Every declared tile type that could legally bind at (r,c) given the
        neighbors present in occ_state (minus `exclude`). Order-agnostic: it
        only looks at WHICH neighbor cells are occupied and what they are."""
        found = []
        for tid in tiles:
            _, total = matches(tid, r, c, occ_state, exclude)
            if total >= 2:
                found.append(tid)
        return found

    # Pass 1: grow a witness assembly with a synchronous-wave schedule
    # (attach every currently-unambiguous frontier cell each round -- one of
    # many valid aTAM attachment orders), failing fast on same-wave
    # ambiguity. While placing each cell, also record which of its occupied
    # neighbors actually contributed to ITS OWN match (its "parents" in the
    # attachment DAG growth necessarily forms: every cell is placed strictly
    # after all of its parents).
    occ = {(0, 0): seed_id}
    children = {(0, 0): set()}
    waves = 0
    changed = True
    while changed and waves < max_waves and len(occ) < max_cells:
        changed = False
        waves += 1
        frontier = set()
        for (r, c) in occ:
            for d, (dr, dc) in DELTA.items():
                nr, nc = r + dr, c + dc
                if (nr, nc) in occ:
                    continue
                if nr < row_lo or nc < col_lo or nr >= row_hi or nc >= col_hi:
                    continue
                frontier.add((nr, nc))
        to_place = {}
        for (r, c) in frontier:
            candidates = eligible(r, c, occ)
            if len(candidates) >= 2:
                die("nondeterministic tile system: >=2 tile types can bind at (row=%d,col=%d)" % (r, c))
            if len(candidates) == 1:
                tid = candidates[0]
                nbs, _ = matches(tid, r, c, occ)
                to_place[(r, c)] = (tid, frozenset(nbs))
        if to_place:
            changed = True
            for (r, c), (tid, nbs) in to_place.items():
                occ[(r, c)] = tid
                children.setdefault((r, c), set())
                for p in nbs:
                    children[p].add((r, c))

    # Pass 2: schedule-independence check. A naive recheck of a cell's
    # candidates against the FULLY GROWN witness assembly can find a
    # "competing" type whose matching neighbor only exists BECAUSE this cell
    # was already placed (a descendant in the DAG above) -- that is not a
    # real alternate schedule, since nothing can occupy a descendant without
    # occupying this cell first. So for every cell the naive recheck flags,
    # exclude all of its DESCENDANTS from the neighbor evidence (a plain
    # graph traversal, not a re-simulation) and recheck: only the
    # contributions from cells that are genuinely independent of this one
    # survive. If a second type is still eligible using only that
    # independent evidence, some valid schedule really could defer this
    # cell, let those other branches grow first, and bind the wrong type --
    # reject it.
    def descendants(start):
        seen = set()
        stack = [start]
        while stack:
            u = stack.pop()
            for v in children.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return seen

    for (r, c), tid in occ.items():
        if (r, c) == (0, 0):
            continue
        naive = eligible(r, c, occ)
        if len(naive) <= 1 and tid in naive:
            continue
        desc = descendants((r, c))
        real = eligible(r, c, occ, exclude=desc)
        if len(real) >= 2:
            die("nondeterministic tile system: independently-reachable tile types both bind at (row=%d,col=%d)" % (r, c))
        if tid not in real:
            die("nondeterministic tile system: the placed type at (row=%d,col=%d) is not the independently-reachable one" % (r, c))

    # ---- feasibility: the checked window must be exactly right ----
    for r in range(W):
        for c in range(n):
            tid = occ.get((r, c))
            if tid is None:
                die("target cell (row=%d,col=%d) never assembled" % (r, c))
            want = (c >> r) & 1
            if tiles[tid]["value"] != want:
                die("cell (row=%d,col=%d) has value %d, want %d" % (r, c, tiles[tid]["value"], want))

    F = K
    B = n * W
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("OK n=%d W=%d K=%d B=%d" % (n, W, K, B))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
