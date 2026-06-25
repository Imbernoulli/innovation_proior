import sys
from itertools import combinations

def solve(n, k, x):
    # Brute force: try every size-k subset of boreholes, compute the minimum pairwise
    # distance (after sorting the chosen coordinates the min gap is between adjacent
    # chosen ones), and maximize that minimum.  O(C(n,k) * k) -- only for tiny n.
    best = None
    for combo in combinations(range(n), k):
        cs = sorted(x[i] for i in combo)
        mind = min(cs[j+1] - cs[j] for j in range(len(cs)-1))
        if best is None or mind > best:
            best = mind
    return best

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    k = int(data[idx]); idx += 1
    x = [int(data[idx+i]) for i in range(n)]
    print(solve(n, k, x))

if __name__ == "__main__":
    main()
