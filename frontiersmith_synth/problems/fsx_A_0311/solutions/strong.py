# TIER: strong
# Product construction: a cap set is closed under Cartesian product. We tile the n
# intersections into blocks of 3 and put a MAXIMAL 9-schedule cap on each block
# (the maximum cap in AG(3,3) has size 9 > 2^3 = 8), times a small base cap on the
# leftover 0/1/2 intersections. This yields ~9^(n/3) schedules, well above the
# {0,1}^n greedy of 2^n, and stays below the (research-hard) true maximum.
import sys

# An explicit maximal cap of size 9 in F_3^3.
CAP3 = [(0, 2, 0), (2, 2, 0), (2, 1, 0), (2, 1, 2), (0, 0, 1),
        (1, 1, 0), (1, 1, 1), (1, 2, 1), (0, 2, 1)]


def base_cap(r):
    if r == 0:
        return [()]
    if r == 1:
        return [(0,), (1,)]
    # r == 2: the {0,1}^2 square is a maximal 4-cap in F_3^2
    return [(0, 0), (0, 1), (1, 0), (1, 1)]


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    a, r = divmod(n, 3)
    blocks = [CAP3] * a + [base_cap(r)]

    # Cartesian product of the per-block caps.
    res = [()]
    for blk in blocks:
        res = [x + y for x in res for y in blk]

    out = ["".join(chr(48 + d) for d in vec) for vec in res]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
