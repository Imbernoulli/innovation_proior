import sys
from itertools import combinations

# Independent brute force for the "vote counting safe orderings" problem.
# For each query (a, b, m): count binary strings with a A's and b B's such that
# at every non-empty prefix, (#A - #B) >= -m  (B never leads by more than m),
# modulo the given prime p. We enumerate all C(a+b, a) interleavings explicitly
# by choosing which positions hold the A votes, and check the running balance.
# This is exponential but obviously correct on the small generated cases.

def safe_count(a, b, m):
    n = a + b
    cnt = 0
    for posA in combinations(range(n), a):
        s = set(posA)
        bal = 0
        ok = True
        for i in range(n):
            bal += 1 if i in s else -1
            if bal < -m:
                ok = False
                break
        if ok:
            cnt += 1
    return cnt

def main():
    data = sys.stdin.read().split()
    idx = 0
    q = int(data[idx]); idx += 1
    p = int(data[idx]); idx += 1
    out = []
    for _ in range(q):
        a = int(data[idx]); b = int(data[idx + 1]); m = int(data[idx + 2]); idx += 3
        out.append(str(safe_count(a, b, m) % p))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))

main()
