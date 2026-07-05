# TIER: greedy
# Yield-priority greedy: consider plots in decreasing water yield and switch on
# each emitter that does not complete a forbidden 3-in-a-line with two already-on
# emitters.  A single deterministic pass -- the natural FunSearch scaffold.
import sys


def str_of(idx, n):
    d = []
    for _ in range(n):
        d.append(idx % 3)
        idx //= 3
    return "".join(str(x) for x in reversed(d))


def idx_of(s):
    v = 0
    for c in s:
        v = v * 3 + (ord(c) - 48)
    return v


def third(x, y):
    return "".join(str((-(int(a) + int(b))) % 3) for a, b in zip(x, y))


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    n = int(next(it))
    m = int(next(it))
    blocked = set(next(it) for _ in range(m))
    N = 3 ** n
    weights = [int(next(it)) for _ in range(N)]

    plots = [str_of(i, n) for i in range(N) if str_of(i, n) not in blocked]
    plots.sort(key=lambda s: -weights[idx_of(s)])

    S = set()
    for p in plots:
        ok = True
        for a in S:
            if third(p, a) in S:
                ok = False
                break
        if ok:
            S.add(p)
    sys.stdout.write("\n".join(S) + "\n")


main()
