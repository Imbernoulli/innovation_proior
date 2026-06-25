import sys
from itertools import combinations

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    K = int(data[idx]); idx += 1
    C = int(data[idx]); idx += 1
    w = []
    v = []
    for _ in range(n):
        w.append(int(data[idx])); idx += 1
        v.append(int(data[idx])); idx += 1

    # Enumerate every subset of EXACTLY K parcels (indices), keep those with weight <= C,
    # maximize total profit. If none, INFEASIBLE.
    if K < 0 or K > n:
        print("INFEASIBLE")
        return

    best = None
    for combo in combinations(range(n), K):
        tw = sum(w[i] for i in combo)
        if tw <= C:
            tv = sum(v[i] for i in combo)
            if best is None or tv > best:
                best = tv

    if best is None:
        print("INFEASIBLE")
    else:
        print(best)

if __name__ == "__main__":
    main()
