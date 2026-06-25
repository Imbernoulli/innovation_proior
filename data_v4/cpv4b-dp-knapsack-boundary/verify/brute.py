import sys
from itertools import combinations

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    L = int(data[idx]); idx += 1
    R = int(data[idx]); idx += 1
    p = []
    v = []
    for _ in range(n):
        p.append(int(data[idx])); idx += 1
        v.append(int(data[idx])); idx += 1

    NEG = None
    best = None
    # Enumerate every subset of items (2^n). For each, compute total price and joy.
    # A subset is valid iff L <= total_price <= R (inclusive on both ends).
    for mask in range(1 << n):
        s = 0
        joy = 0
        for i in range(n):
            if mask & (1 << i):
                s += p[i]
                joy += v[i]
        if L <= s <= R:
            if best is None or joy > best:
                best = joy

    if best is None:
        print("IMPOSSIBLE")
    else:
        print(best)

if __name__ == "__main__":
    main()
