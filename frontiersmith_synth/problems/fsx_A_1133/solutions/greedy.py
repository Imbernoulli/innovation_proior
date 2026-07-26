# TIER: greedy
# The "obvious first attempt": try each of the classic mirror/half-turn axes in a
# fixed priority order, accept the first one whose visible pairs mostly agree, and
# copy across that single axis. No rotation, no translation, no lattice search --
# and any cell whose mirror partner is also erased just gets the global mode color.
import sys, json
from collections import Counter


def transform(N, kind, r, c):
    if kind == "fh":
        return N - 1 - r, c
    if kind == "fv":
        return r, N - 1 - c
    if kind == "fd":
        return c, r
    if kind == "fa":
        return N - 1 - c, N - 1 - r
    if kind == "r180":
        return N - 1 - r, N - 1 - c
    return r, c


def main():
    inst = json.load(sys.stdin)
    N, K, grid = inst["n"], inst["k"], inst["grid"]
    flat = [grid[r][c] for r in range(N) for c in range(N)]

    def idx(r, c):
        return r * N + c

    known = [v for v in flat if v != -1]
    cnt = Counter(known) if known else Counter()
    mode = max(range(K), key=lambda k: cnt.get(k, 0)) if known else 0

    best_kind = None
    for kind in ["fh", "fv", "fd", "fa", "r180"]:
        agree = disagree = 0
        for r in range(N):
            for c in range(N):
                i = idx(r, c)
                if flat[i] == -1:
                    continue
                r2, c2 = transform(N, kind, r, c)
                j = idx(r2, c2)
                if j == i or flat[j] == -1:
                    continue
                if flat[i] == flat[j]:
                    agree += 1
                else:
                    disagree += 1
        tested = agree + disagree
        rate = (agree / tested) if tested > 0 else 0.0
        if tested >= 1 and rate >= 0.9:
            best_kind = kind
            break

    out = list(flat)
    for r in range(N):
        for c in range(N):
            i = idx(r, c)
            if out[i] != -1:
                continue
            filled = False
            if best_kind is not None:
                r2, c2 = transform(N, best_kind, r, c)
                j = idx(r2, c2)
                if flat[j] != -1:
                    out[i] = flat[j]
                    filled = True
            if not filled:
                out[i] = mode
    grid_out = [[out[idx(r, c)] for c in range(N)] for r in range(N)]
    print(json.dumps({"grid": grid_out}))


main()
