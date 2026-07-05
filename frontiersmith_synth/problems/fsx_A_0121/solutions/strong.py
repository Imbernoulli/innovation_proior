# TIER: strong
# Product construction: decompose n into blocks of size {4,3,2,1}, take the (hardcoded,
# optimal) cap set for each block, and form the Cartesian product.  The product of cap
# sets is a cap set, so this is always feasible; it beats the hypercube {0,1}^n for n>=6.
import sys, itertools

BLOCK = {
    1: [(0,), (2,)],
    2: [(0, 0), (0, 1), (2, 0), (2, 1)],
    3: [(0, 1, 2), (0, 2, 1), (0, 2, 2), (1, 0, 0), (1, 1, 0),
        (1, 1, 2), (2, 1, 0), (2, 2, 0), (2, 2, 2)],
    4: [(0, 0, 1, 0), (0, 0, 1, 2), (0, 0, 2, 0), (0, 0, 2, 2), (0, 2, 0, 1),
        (0, 2, 0, 2), (0, 2, 1, 1), (0, 2, 1, 2), (1, 0, 2, 2), (1, 1, 0, 0),
        (1, 2, 0, 1), (1, 2, 1, 0), (1, 2, 1, 2), (1, 2, 2, 1), (2, 0, 0, 0),
        (2, 0, 1, 1), (2, 0, 1, 2), (2, 0, 2, 0), (2, 1, 2, 1), (2, 2, 0, 2)],
}


def main():
    n = int(sys.stdin.read().split()[0])
    parts = []
    r = n
    for p in (4, 3, 2, 1):
        while r >= p:
            parts.append(p)
            r -= p
    cur = [()]
    for p in parts:
        cur = [a + b for a in cur for b in BLOCK[p]]
    out = [str(len(cur))]
    for v in cur:
        out.append(" ".join(map(str, v)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
