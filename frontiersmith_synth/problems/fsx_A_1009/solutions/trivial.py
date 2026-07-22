# TIER: trivial
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); m = int(next(it)); cap = int(next(it)); K = int(next(it))
    for _ in range(K):
        next(it); next(it)  # pairs are ignored by the trivial construction

    # Naive block-partition matrix: split components into contiguous blocks, one
    # probe per block. Reproduces the checker's own internal baseline.
    nb = max(1, min(m, n))
    block_size = (n + nb - 1) // nb
    rows = []
    for bi in range(nb):
        lo = bi * block_size
        hi = min(n, lo + block_size)
        row = ["0"] * n
        for j in range(lo, hi):
            row[j] = "1"
        rows.append("".join(row))

    print(len(rows))
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
