# TIER: trivial
# The checker's own baseline: fold the chain into a hairpin (2-row serpentine,
# width ceil(L/2)).  Reproduces B, so it scores ~0.1.
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
    out = []
    for k in range(1, len(coords)):
        dx = coords[k][0] - coords[k - 1][0]
        dy = coords[k][1] - coords[k - 1][1]
        out.append(d[(dx, dy)])
    return " ".join(out)


def main():
    toks = sys.stdin.read().split()
    L = int(toks[0])
    Wh = (L + 1) // 2
    sys.stdout.write(moves_from_coords(serp_coords(L, Wh)) + "\n")


if __name__ == "__main__":
    main()
