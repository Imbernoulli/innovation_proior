# TIER: greedy
import sys

CLASSES = "HML"


def main():
    data = sys.stdin.read().split()
    idx = [0]

    def nxt():
        v = data[idx[0]]
        idx[0] += 1
        return v

    N = int(nxt())
    D = int(nxt())
    FIXSEP = int(nxt())
    S = [[0, 0, 0] for _ in range(3)]
    for a in range(3):
        for b in range(3):
            S[a][b] = int(nxt())
    fw = [int(nxt()) for _ in range(3)]
    r = []
    cls = []
    for i in range(N):
        r.append(int(nxt()))
        cls.append(CLASSES.index(nxt()))

    # Obvious "competent" recipe: process aircraft strictly in ready-time order
    # (never reorder by class), and at each step drop the aircraft onto whichever
    # runway currently offers the earliest feasible landing (list scheduling /
    # load balancing). Uses both runways, but is blind to the wake-class pattern.
    order = sorted(range(N), key=lambda i: (r[i], i))
    fix_prev = None
    runway_land = [None, None]
    runway_cls = [None, None]
    out = [None] * N
    for i in order:
        ft = r[i] if fix_prev is None else max(r[i], fix_prev + FIXSEP)
        fix_prev = ft
        best_rw, best_lt = None, None
        for rw in (0, 1):
            lt = ft + D
            if runway_land[rw] is not None:
                lt = max(lt, runway_land[rw] + S[runway_cls[rw]][cls[i]])
            if best_lt is None or lt < best_lt:
                best_lt, best_rw = lt, rw
        runway_land[best_rw] = best_lt
        runway_cls[best_rw] = cls[i]
        out[i] = (best_rw + 1, ft, best_lt)

    lines = ["%d %d %d" % t for t in out]
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
