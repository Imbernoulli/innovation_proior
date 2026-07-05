# TIER: greedy
# Difference-set product with a SMALL progression-free difference set A (size <=3):
# T = {(x,y): (x-y) mod n in A}.  Since n is odd and A has no 3-term AP, T is
# corner-free.  A few diagonals already beat the single-row baseline.
import sys


def ap_free(Aset, n):
    A = list(Aset)
    for p in A:
        for q in A:
            if p == q:
                continue
            r = (2 * q - p) % n
            if r in Aset and r != p and r != q:
                return False
    return True


def main():
    t = sys.stdin.read().split()
    i = 0
    n = int(t[i]); i += 1
    k = int(t[i]); i += 1
    blocked = set()
    for _ in range(k):
        blocked.add((int(t[i]), int(t[i + 1]))); i += 2

    # greedy progression-free difference set, capped at 3 diagonals
    A = set()
    for a in range(n):
        cand = A | {a}
        if ap_free(cand, n):
            A.add(a)
        if len(A) >= 3:
            break

    S = []
    for x in range(n):
        for y in range(n):
            if (x - y) % n in A and (x, y) not in blocked:
                S.append((x, y))

    out = [str(len(S))]
    for (x, y) in S:
        out.append("%d %d" % (x, y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
