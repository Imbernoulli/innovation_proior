# TIER: trivial
# Naive equal-spacing guess: offsets 0,1,...,k-1 (ignores that the plaza has n=k*k
# cells and that only step-k spacing can possibly tile it), and one arbitrary
# (smallest-index) quarry-approved cell per residue class mod k for the tile.
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    k = int(data[idx]); idx += 1
    allowed = [int(x) for x in data[idx:idx + n]]; idx += n
    # cost not needed by this tier

    classes = [[] for _ in range(k)]
    for i in range(n):
        if allowed[i]:
            classes[i % k].append(i)
    B = [min(c) for c in classes]
    T = list(range(k))  # WRONG: naive contiguous offsets instead of multiples of k

    print(" ".join(map(str, B)))
    print(" ".join(map(str, T)))


if __name__ == "__main__":
    main()
