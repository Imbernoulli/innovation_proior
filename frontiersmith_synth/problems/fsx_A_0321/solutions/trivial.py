# TIER: trivial
# Structural baseline: turn on every emitter whose plot address uses only
# terraces 0 and 1 (the {0,1}^n sub-cube).  This is always a valid cap set but
# it ignores the water yields entirely -- reproduces the checker's baseline.
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
    out = []
    for mask in range(1 << n):
        s = "".join("1" if (mask >> (n - 1 - b)) & 1 else "0" for b in range(n))
        if s not in blocked:
            out.append(s)
    sys.stdout.write("\n".join(out) + "\n")


main()
