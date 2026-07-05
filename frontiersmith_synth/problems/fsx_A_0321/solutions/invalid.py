# TIER: invalid
# Turns on every non-rocky emitter.  For n>=2 the full vineyard always contains
# three collinear emitters, so the feasibility gate rejects it -> score 0.
import sys


def str_of(idx, n):
    d = []
    for _ in range(n):
        d.append(idx % 3)
        idx //= 3
    return "".join(str(x) for x in reversed(d))


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    n = int(next(it))
    m = int(next(it))
    blocked = set(next(it) for _ in range(m))
    N = 3 ** n
    out = [str_of(i, n) for i in range(N) if str_of(i, n) not in blocked]
    sys.stdout.write("\n".join(out) + "\n")


main()
