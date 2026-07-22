# TIER: strong
# Insight: the fold width is the real degree of freedom, and the checkerboard
# parity invariant means only opposite-parity residues (odd chain-distance) can
# ever bind.  The score-maximising width is not a round number; it is wherever
# the chain's binding residues line up into vertical (opposite-parity) stacks.
# So do a FINE sweep over every feasible serpentine width and take the best.
# This finds the hidden resonant prime width the coarse search skips, letting
# late residues still form contacts instead of being stranded on the surface.
import sys


def serp_coords(L, w):
    coords = []
    idx = 0
    y = 0
    while idx < L:
        rowlen = min(w, L - idx)
        xs = range(0, rowlen) if y % 2 == 0 else range(w - 1, w - 1 - rowlen, -1)
        for x in xs:
            coords.append((x, y))
            idx += 1
        y += 1
    return coords


def moves_from_coords(coords):
    d = {(1, 0): "0", (-1, 0): "1", (0, 1): "2", (0, -1): "3"}
    return " ".join(
        d[(coords[k][0] - coords[k - 1][0], coords[k][1] - coords[k - 1][1])]
        for k in range(1, len(coords))
    )


def objective(coords, wt):
    occ = {c: i for i, c in enumerate(coords)}
    total = 0
    for i, (x, y) in enumerate(coords):
        wi = wt[i]
        if wi == 0:
            continue
        for nb in ((x + 1, y), (x, y + 1)):
            j = occ.get(nb)
            if j is not None and abs(i - j) != 1:
                total += wi * wt[j]
    return total


def main():
    toks = sys.stdin.read().split()
    L = int(toks[0])
    wt = [int(x) for x in toks[2:2 + L]]

    best_w, best_v = 2, -1
    for w in range(2, L):
        v = objective(serp_coords(L, w), wt)
        if v > best_v:
            best_v, best_w = v, w

    sys.stdout.write(moves_from_coords(serp_coords(L, best_w)) + "\n")


if __name__ == "__main__":
    main()
