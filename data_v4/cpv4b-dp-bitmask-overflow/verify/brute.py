import sys
from itertools import permutations

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    c = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(int(data[idx])); idx += 1
        c.append(row)

    # Brute force: try every assignment of courier i -> zone perm[i].
    best = None
    for perm in permutations(range(n)):
        total = 0
        for i in range(n):
            total += c[i][perm[i]]
        if best is None or total < best:
            best = total
    if best is None:
        best = 0  # n == 0
    print(best)

if __name__ == "__main__":
    main()
