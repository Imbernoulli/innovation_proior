# TIER: greedy
# Mian-Chowla style greedy Sidon (B_2) construction inside [0, M]: scan values
# upward and keep one only if ALL of its new pairwise sums are still distinct
# from every sum seen so far.  A Sidon set has ALL pairwise sums distinct, so its
# sumset is as large as possible per element -- but only ~sqrt(2M) elements fit,
# so it usually stops well short of the budget n, leaving room on the table.
import sys

def main():
    toks = sys.stdin.read().split()
    n, M = int(toks[0]), int(toks[1])
    A = []
    sums = set()
    x = 0
    while x <= M and len(A) < n:
        new = set()
        ok = True
        for a in A:
            s = a + x
            if s in sums or s in new:
                ok = False
                break
            new.add(s)
        if ok:
            s2 = x + x
            if s2 in sums or s2 in new:
                ok = False
            else:
                new.add(s2)
        if ok:
            A.append(x)
            sums |= new
        x += 1
    if not A:
        A = [0]
    sys.stdout.write(" ".join(map(str, A)) + "\n")

if __name__ == "__main__":
    main()
