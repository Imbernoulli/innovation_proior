# TIER: greedy
# Frozen scaffold with a WEIGHT-AGNOSTIC lexicographic priority: scan the
# unblocked cells in canonical order, keep each tower iff it completes no
# collinear line with the towers already chosen. Maximizes count, ignores value.
import sys, itertools


def main():
    raw = sys.stdin.read().splitlines()
    ptr = 0
    n = int(raw[ptr].split()[0]); ptr += 1
    b = int(raw[ptr].split()[0]); ptr += 1
    blocked = set()
    for _ in range(b):
        blocked.add(raw[ptr].strip()); ptr += 1

    allowed = []
    for v in itertools.product(range(3), repeat=n):
        s = ''.join(map(str, v))
        if s not in blocked:
            allowed.append(v)
    allowed.sort()

    S = []
    Sset = set()
    for v in allowed:
        ok = True
        for x in S:
            w = tuple((-(x[k] + v[k])) % 3 for k in range(n))
            if w in Sset:
                ok = False
                break
        if ok:
            S.append(v)
            Sset.add(v)

    sys.stdout.write('\n'.join(''.join(map(str, v)) for v in S) + '\n')


if __name__ == "__main__":
    main()
