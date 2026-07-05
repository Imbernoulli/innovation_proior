# TIER: trivial
# Reproduces the checker's audit-grid baseline: phases in {0,1} on lines
# 0..n-2, phase 0 on the last line, minus reserved configs. size = 2^(n-1).
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    n = int(next(it)); k = int(next(it))
    blocked = set()
    for _ in range(k):
        blocked.add(tuple(int(next(it)) for _ in range(n)))
    S = []
    for mask in range(1 << (n - 1)):
        v = tuple((mask >> i) & 1 for i in range(n - 1)) + (0,)
        if v not in blocked:
            S.append(v)
    out = [str(len(S))] + [" ".join(map(str, v)) for v in S]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
