# TIER: greedy
# The obvious approach: "compaction helps, so try a handful of round fold widths
# and keep the best."  Samples widths coarsely (small widths + the square root +
# the hairpin) and picks the best-scoring one.  It never probes the odd prime
# width where the planted stacks resonate, so it leaves that value on the table.
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

    root = int(round(L ** 0.5))
    cand = {2, 4, 8, 16, 32, 64, root, (L + 1) // 2}
    cand = sorted(w for w in cand if 2 <= w <= L - 1)

    best_w, best_v = cand[0], -1
    for w in cand:
        v = objective(serp_coords(L, w), wt)
        if v > best_v:
            best_v, best_w = v, w

    sys.stdout.write(moves_from_coords(serp_coords(L, best_w)) + "\n")


if __name__ == "__main__":
    main()
