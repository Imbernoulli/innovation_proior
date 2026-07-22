#!/usr/bin/env python3
# verify.py <in> <out> <ans>   (ans is an empty placeholder; ignore it)
#
# Deterministic scorer for the H/P lattice folder.
#   <in>  : L / HP-string / weights   (see gen.py)
#   <out> : L-1 integer move codes (0=+x, 1=-x, 2=+y, 3=-y) describing a
#           self-avoiding walk that places residue 0 at the origin and residue i
#           via move i-1.
#
# Feasibility (ANY violation -> Ratio: 0.0):
#   * exactly L-1 move tokens, each an integer in {0,1,2,3};
#   * the resulting walk is self-avoiding (all L lattice cells distinct).
#
# Objective F (maximize): sum over unordered pairs {i,j} of residues that are
#   lattice-adjacent AND non-sequential (|i-j| != 1) of  w[i]*w[j].
#   Because a unit lattice step flips checkerboard colour, every adjacency joins
#   opposite-parity residues, so only opposite-parity H's can ever bind.
#
# Baseline B: the objective of the checker's own hairpin (2-row serpentine)
#   fold.  Normalised maximisation score:
#       sc = min(1000.0, 100.0 * F / max(1e-9, B)) ;  print Ratio = sc/1000.
import sys

DIRS = {0: (1, 0), 1: (-1, 0), 2: (0, 1), 3: (0, -1)}


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    L = int(toks[0])
    s = toks[1]
    wt = [int(x) for x in toks[2:2 + L]]
    return L, s, wt


def serp_coords(L, w):
    """Coordinates of a width-w boustrophedon (serpentine) fold, indices 0..L-1."""
    coords = []
    idx = 0
    y = 0
    while idx < L:
        rowlen = min(w, L - idx)
        if y % 2 == 0:
            xs = range(0, rowlen)
        else:
            xs = range(w - 1, w - 1 - rowlen, -1)
        for x in xs:
            coords.append((x, y))
            idx += 1
        y += 1
    return coords


def objective(coords, wt):
    """Sum of w[i]*w[j] over non-sequential lattice-adjacent pairs."""
    occ = {}
    for i, c in enumerate(coords):
        occ[c] = i
    total = 0
    for i, (x, y) in enumerate(coords):
        wi = wt[i]
        if wi == 0:
            continue
        for nb in ((x + 1, y), (x, y + 1)):   # each undirected pair once
            j = occ.get(nb)
            if j is not None and abs(i - j) != 1:
                total += wi * wt[j]
    return total


def parse_walk(path, L):
    """Return coords list if feasible, else None."""
    try:
        with open(path) as f:
            raw = f.read()
    except Exception:
        return None
    toks = raw.split()
    if len(toks) != L - 1:
        return None
    moves = []
    for t in toks:
        try:
            m = int(t)                   # rejects nan/inf/garbage/floats
        except ValueError:
            return None
        if m not in DIRS:                # rejects out-of-range / huge codes
            return None
        moves.append(m)
    x = y = 0
    coords = [(0, 0)]
    seen = {(0, 0)}
    for m in moves:
        dx, dy = DIRS[m]
        x += dx
        y += dy
        if (x, y) in seen:               # self-intersection
            return None
        seen.add((x, y))
        coords.append((x, y))
    return coords


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    L, s, wt = read_instance(inp)

    coords = parse_walk(outp, L)
    if coords is None:
        print("Infeasible output (bad moves or self-intersection). Ratio: 0.0")
        return

    F = objective(coords, wt)

    # checker baseline: the hairpin (2-row) fold.
    Wh = (L + 1) // 2
    B = objective(serp_coords(L, Wh), wt)
    if B <= 0:
        B = 1

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
