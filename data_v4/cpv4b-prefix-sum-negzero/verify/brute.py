import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    d = [int(next(it)) for _ in range(n)]

    # Build the level array L[0..n] explicitly: L[0] = 0, L[k] = L[k-1] + d[k-1].
    L = [0] * (n + 1)
    for k in range(1, n + 1):
        L[k] = L[k - 1] + d[k - 1]

    # Maximum drawdown = max over all i <= j of (L[i] - L[j]). i = j gives 0.
    best = 0
    for i in range(n + 1):
        for j in range(i, n + 1):
            drop = L[i] - L[j]
            if drop > best:
                best = drop

    print(best)

main()
