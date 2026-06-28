#!/usr/bin/env python3
"""Deterministic local scorer for "Polyomino Tiling Coverage".

Usage:
    python3 score.py <instance_file> <solution_file>

Prints a single integer: the score. HIGHER is better. INFEASIBLE -> 0.

Scoring rule (see context.md "Evaluation settings").

  * INSTANCE (read from <instance_file>):
        H W P
        then, for each of the P piece types:
            k
            dr_0 dc_0
            ...
            dr_{k-1} dc_{k-1}
            avail
    An H x W board. Piece type t is a polyomino given by k_t integer cell offsets
    (already normalized to min row = min col = 0). It may be used at most avail_t
    times.

  * SOLUTION (read from <solution_file>):
        M
        type rot r c        (M lines)
    M placements. The placement (type t, rot, r, c) occupies the cells obtained by
    rotating type t's offsets by `rot` quarter-turns CLOCKWISE, re-normalizing the
    rotated offsets so their min row and min col are 0, and then translating by
    (r, c). So if the rotated+normalized offsets are {(dr, dc)}, the occupied cells
    are {(r + dr, c + dc)}.

  * FEASIBILITY (any violation -> score 0):
      - the file parses: a leading integer M >= 0, then exactly 4*M further
        integer tokens;
      - every type in [0, P), every rot in {0,1,2,3};
      - every occupied cell is on the board: 0 <= row < H, 0 <= col < W;
      - no two occupied cells (across all placements) coincide (no overlap);
      - each type t is used at most avail_t times.
    If any of these fail, the solution is INFEASIBLE and scores 0.

  * COVERAGE (higher better) of a feasible solution = number of distinct board
    cells occupied by the placements (= sum of piece sizes, since no overlap).

  * SCORE, normalized against a deterministic greedy LARGEST-PIECE-FIRST baseline
    that the scorer recomputes itself:
        score = round(1_000_000 * solver_coverage / max(1, baseline_coverage))
    The greedy baseline scores ~1_000_000; covering more than it scores more, an
    infeasible solution scores 0.
"""
import sys


# ----------------------------------------------------------------------------- IO
def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    H = int(next(it))
    W = int(next(it))
    P = int(next(it))
    pieces = []          # list of (cells, avail); cells = list of (dr, dc)
    for _ in range(P):
        k = int(next(it))
        cells = []
        for _ in range(k):
            dr = int(next(it))
            dc = int(next(it))
            cells.append((dr, dc))
        avail = int(next(it))
        pieces.append((cells, avail))
    return H, W, P, pieces


def rotate_norm(cells, rot):
    """Rotate offsets by `rot` quarter-turns clockwise, then normalize to min 0.

    A clockwise quarter turn maps (r, c) -> (c, -r). Apply it `rot` times, then
    shift so the minimum row and minimum col are both 0. Returns sorted tuples.
    """
    pts = list(cells)
    for _ in range(rot % 4):
        pts = [(c, -r) for (r, c) in pts]
    rmin = min(r for r, _ in pts)
    cmin = min(c for _, c in pts)
    return sorted(((r - rmin, c - cmin) for (r, c) in pts))


def read_solution(path, H, W, P, pieces):
    """Parse + fully validate. Return total covered-cell count, or None if infeasible."""
    try:
        with open(path) as f:
            toks = f.read().split()
    except OSError:
        return None
    if len(toks) == 0:
        return None
    try:
        ints = [int(x) for x in toks]
    except ValueError:
        return None
    M = ints[0]
    if M < 0:
        return None
    if len(ints) != 1 + 4 * M:
        return None

    used = [0] * P
    occupied = set()
    pos = 1
    for _ in range(M):
        t = ints[pos]
        rot = ints[pos + 1]
        r = ints[pos + 2]
        c = ints[pos + 3]
        pos += 4
        if t < 0 or t >= P:
            return None
        if rot < 0 or rot > 3:
            return None
        used[t] += 1
        if used[t] > pieces[t][1]:
            return None
        shape = rotate_norm(pieces[t][0], rot)
        for (dr, dc) in shape:
            rr = r + dr
            cc = c + dc
            if rr < 0 or rr >= H or cc < 0 or cc >= W:
                return None
            if (rr, cc) in occupied:
                return None
            occupied.add((rr, cc))
    return len(occupied)


# ---------------------------------------------- baseline: greedy largest-first
def baseline_coverage(H, W, P, pieces):
    """Greedy largest-piece-first fill.

    Consider piece types in order of decreasing cell count. For each type, while
    copies remain, scan board anchors (top-to-bottom, left-to-right) and, for each
    of its 4 rotations, place the piece at the first anchor where it fits on empty
    cells. This is the natural "use big pieces first" greedy the scorer measures
    against; it never overlaps, so its coverage is a legitimate normalizer.
    """
    board = [[False] * W for _ in range(H)]
    covered = 0

    order = sorted(range(P), key=lambda t: -len(pieces[t][0]))
    # precompute the 4 rotated+normalized shapes per type
    shapes = []
    for t in range(P):
        rots = []
        seen = set()
        for rot in range(4):
            s = tuple(rotate_norm(pieces[t][0], rot))
            if s not in seen:
                seen.add(s)
                rots.append(s)
        shapes.append(rots)

    for t in order:
        avail = pieces[t][1]
        placed = 0
        if avail <= 0:
            continue
        # keep scanning the board until no placement of this type fits
        progress = True
        while placed < avail and progress:
            progress = False
            for r in range(H):
                for c in range(W):
                    if placed >= avail:
                        break
                    for s in shapes[t]:
                        ok = True
                        for (dr, dc) in s:
                            rr = r + dr
                            cc = c + dc
                            if rr < 0 or rr >= H or cc < 0 or cc >= W or board[rr][cc]:
                                ok = False
                                break
                        if ok:
                            for (dr, dc) in s:
                                board[r + dr][c + dc] = True
                            covered += len(s)
                            placed += 1
                            progress = True
                            break
                if placed >= avail:
                    break
    return covered


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py <instance> <solution>\n")
        sys.exit(1)
    H, W, P, pieces = read_instance(sys.argv[1])

    cov = read_solution(sys.argv[2], H, W, P, pieces)
    if cov is None:
        print(0)
        return

    base = baseline_coverage(H, W, P, pieces)
    score = int(round(1_000_000.0 * cov / max(1, base)))
    print(score)


if __name__ == "__main__":
    main()
