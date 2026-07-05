# TIER: invalid
# Emits an INFEASIBLE artifact: three unblocked cells that form a corner
# (x,y),(x+d,y),(x,y+d).  The checker must detect the corner and score 0.
import sys


def main():
    t = sys.stdin.read().split()
    i = 0
    n = int(t[i]); i += 1
    k = int(t[i]); i += 1
    blocked = set()
    for _ in range(k):
        blocked.add((int(t[i]), int(t[i + 1]))); i += 2

    # find an explicit corner among unblocked cells
    for x in range(n):
        for y in range(n):
            for d in range(1, n):
                a = (x, y)
                b = ((x + d) % n, y)
                c = (x, (y + d) % n)
                if a not in blocked and b not in blocked and c not in blocked \
                        and len({a, b, c}) == 3:
                    pts = [a, b, c]
                    out = [str(len(pts))]
                    for (px, py) in pts:
                        out.append("%d %d" % (px, py))
                    sys.stdout.write("\n".join(out) + "\n")
                    return

    # fallback: dump every unblocked cell (dense -> certainly has a corner)
    S = [(x, y) for x in range(n) for y in range(n) if (x, y) not in blocked]
    out = [str(len(S))]
    for (x, y) in S:
        out.append("%d %d" % (x, y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
