import sys
from itertools import combinations

def main():
    data = sys.stdin.read().split()
    n = int(data[0]); k = int(data[1])
    s = data[2] if len(data) > 2 else ""
    # Independent brute force: enumerate every way to KEEP (n-k) positions in
    # order and take the lexicographically smallest resulting string.
    keep = n - k
    best = None
    idxs = range(n)
    for combo in combinations(idxs, keep):
        cand = "".join(s[i] for i in combo)
        if best is None or cand < best:
            best = cand
    if best is None:
        best = ""
    print(best)

main()
