# TIER: greedy
# Geometric-spacing greedy: consider candidate codewords along a geometric ladder
# (ratio ~1.3) and keep each one that preserves the conflict-free (Sidon) property.
# Denser than the powers-of-two baseline but well short of the increasing-order greedy.
import sys


def main():
    n = int(sys.stdin.read().split()[0])
    S = []
    sums = set()
    seen = set()
    r = 1.3
    k = 0
    v = 1.0
    while True:
        x = int(round(v)) if k > 0 else 1
        v *= r
        k += 1
        if x > n:
            break
        if x < 1 or x in seen:
            continue
        seen.add(x)
        cand = [x + y for y in S]
        cand.append(2 * x)
        ok = True
        for c in cand:
            if c in sums:
                ok = False
                break
        if ok:
            S.append(x)
            sums.update(cand)
    S.sort()
    sys.stdout.write(" ".join(map(str, S)) + "\n")


if __name__ == "__main__":
    main()
