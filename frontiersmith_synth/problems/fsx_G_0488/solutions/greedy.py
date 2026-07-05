# TIER: greedy
# Genuine greedy Stanley-sequence scan in Z_p: walk offsets in increasing order
# up to a mid register bound (p/2)/2 and add each one that does not complete a
# 3-term arithmetic progression with the set built so far. Larger than trivial,
# smaller than the full construction.
import sys


def main():
    p = int(sys.stdin.read().split()[0])
    half = (p + 1) // 2
    bound = max(2, half // 2)
    inv2 = pow(2, p - 2, p)
    S = []
    Sset = set()
    for x in range(bound):
        bad = False
        for a in S:
            if ((x + a) * inv2) % p in Sset:
                bad = True
                break
            if (2 * a - x) % p in Sset:
                bad = True
                break
            if (2 * x - a) % p in Sset:
                bad = True
                break
        if not bad:
            S.append(x)
            Sset.add(x)
    print(" ".join(map(str, S)))


if __name__ == "__main__":
    main()
