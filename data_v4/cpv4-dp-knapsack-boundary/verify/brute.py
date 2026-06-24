import sys
from itertools import combinations

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    K = int(data[idx]); idx += 1
    g = int(data[idx]); idx += 1
    s = []
    v = []
    for _ in range(n):
        si = int(data[idx]); idx += 1
        vi = int(data[idx]); idx += 1
        s.append(si)
        v.append(vi)

    U = K - g
    if U < 0:
        U = 0

    # Exhaustive over all subsets; keep those whose total space <= U.
    best = 0
    for r in range(0, n + 1):
        for combo in combinations(range(n), r):
            tot = sum(s[i] for i in combo)
            if tot <= U:
                val = sum(v[i] for i in combo)
                if val > best:
                    best = val
    print(best)

if __name__ == "__main__":
    main()
