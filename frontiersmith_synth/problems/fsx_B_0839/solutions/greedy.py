# TIER: greedy
"""The obvious first attempt: a myopic nearest-value search. At each step,
from whichever tank currently holds the batch, take the feasible valve whose
resulting volume is numerically closest to the target window's midpoint (an
immediate window-hit is taken on sight). No lookahead, no notion of the
underlying multiplicative structure -- exactly the recipe an average strong
coder reaches for first, and exactly what a "went further away raw-value-wise
but exponent-correct" shortcut defeats."""
import sys
from fractions import Fraction


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); M = int(next(it))
    caps = [int(next(it)) for _ in range(N)]
    V0 = int(next(it)); target = int(next(it)); Nb = int(next(it))
    Lo = Fraction(int(next(it)), int(next(it)))
    Hi = Fraction(int(next(it)), int(next(it)))
    for _ in range(Nb):
        next(it)                      # explicit backbone valve ids, unused here
    edges = []
    for _ in range(M):
        u = int(next(it)); v = int(next(it))
        num = int(next(it)); den = int(next(it))
        edges.append((u, v, num, den))

    by_tank = {}
    for eid, (u, v, num, den) in enumerate(edges):
        by_tank.setdefault(u, []).append((eid, v, num, den))

    mid = (Lo + Hi) / 2
    cur_tank = 0
    cur_vol = Fraction(V0)
    path = []
    MAX_STEPS = 3 * N + 10

    for _ in range(MAX_STEPS):
        cands = by_tank.get(cur_tank, [])
        feas = []
        for (eid, v, num, den) in cands:
            nv = cur_vol * Fraction(num, den)
            if nv > caps[v]:
                continue
            feas.append((eid, v, nv))
        if not feas:
            break                      # dead end: nothing more we can do

        win = [c for c in feas if c[1] == target and Lo <= c[2] <= Hi]
        if win:
            win.sort(key=lambda c: c[0])
            eid, v, nv = win[0]
            path.append(eid); cur_tank, cur_vol = v, nv
            break

        feas.sort(key=lambda c: (abs(c[2] - mid), c[0]))
        eid, v, nv = feas[0]
        path.append(eid); cur_tank, cur_vol = v, nv
        if cur_tank == target and Lo <= cur_vol <= Hi:
            break

    print(len(path))
    print(" ".join(map(str, path)))


if __name__ == "__main__":
    main()
