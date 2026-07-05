# TIER: greedy
# Lexicographic greedy: scan configs in base-3 order, deploy one whenever it
# creates no resonance cascade with the current deployment (reserved configs
# are pre-forbidden).  Discovers the full {0,1}^n cap -> ~2^n substations.
import sys


def third(a, b, n):
    return tuple((-(a[i] + b[i])) % 3 for i in range(n))


def to_vec(idx, n):
    v = []
    for _ in range(n):
        v.append(idx % 3); idx //= 3
    return tuple(v)


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    n = int(next(it)); k = int(next(it))
    blocked = set()
    for _ in range(k):
        blocked.add(tuple(int(next(it)) for _ in range(n)))
    S = []; Sset = set(); forbidden = set(blocked)
    for idx in range(3 ** n):
        v = to_vec(idx, n)
        if v in forbidden or v in Sset:
            continue
        for u in S:
            forbidden.add(third(u, v, n))
        S.append(v); Sset.add(v)
    out = [str(len(S))] + [" ".join(map(str, v)) for v in S]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
