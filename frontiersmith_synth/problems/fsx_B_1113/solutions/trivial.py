# TIER: trivial
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

    # Two runways, strict ready-time (FCFS) order, blind round-robin runway
    # alternation -- no reordering by class, no choosing the "better" runway.
    order = sorted(range(N), key=lambda i: (r[i], i))
    fix_prev = None
    land_prev = [None, None]
    prevcls = [None, None]
    out = [None] * N
    for k, i in enumerate(order):
        ft = r[i] if fix_prev is None else max(r[i], fix_prev + FIXSEP)
        rw = k % 2
        lt = ft + D
        if land_prev[rw] is not None:
            lt = max(lt, land_prev[rw] + S[prevcls[rw]][cls[i]])
        out[i] = (rw + 1, ft, lt)
        fix_prev = ft
        land_prev[rw] = lt
        prevcls[rw] = cls[i]

    lines = ["%d %d %d" % t for t in out]
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
