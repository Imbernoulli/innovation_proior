# TIER: greedy
# Farthest-point greedy on a dense feasible candidate grid (warm zones removed).
import sys, math


def read_instance():
    d = sys.stdin.read().split()
    it = iter(d)
    n = int(next(it)); c = int(next(it))
    disks = [(float(next(it)), float(next(it)), float(next(it))) for _ in range(c)]
    return n, disks


def feasible(x, y, disks):
    if x < 0 or x > 1 or y < 0 or y > 1:
        return False
    for (cx, cy, r) in disks:
        if (x - cx) ** 2 + (y - cy) ** 2 < r * r:
            return False
    return True


def main():
    n, disks = read_instance()
    # dense candidate grid, warm zones removed
    G = 40
    cand = []
    for i in range(G):
        for j in range(G):
            x = i / (G - 1)
            y = j / (G - 1)
            if feasible(x, y, disks):
                cand.append((x, y))

    # farthest-point greedy: start from the corner-most candidate
    chosen = [min(cand, key=lambda p: (p[0] + p[1]))]
    dist2 = [(px - chosen[0][0]) ** 2 + (py - chosen[0][1]) ** 2 for (px, py) in cand]
    while len(chosen) < n:
        # pick candidate maximizing distance to nearest chosen
        bi = max(range(len(cand)), key=lambda k: dist2[k])
        chosen.append(cand[bi])
        cx, cy = cand[bi]
        for k in range(len(cand)):
            dd = (cand[k][0] - cx) ** 2 + (cand[k][1] - cy) ** 2
            if dd < dist2[k]:
                dist2[k] = dd

    out = ["%.10f %.10f" % (x, y) for (x, y) in chosen[:n]]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
