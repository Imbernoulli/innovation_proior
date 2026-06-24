import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    n = int(data[0])
    t = [int(x) for x in data[1:1+n]]
    if n == 0:
        print(0)
        return
    if n == 1:
        print(t[0])
        return

    # Fully exhaustive, NON-memoized enumeration of every legal route from
    # platform 0 to platform n-1 using steps of +1 or +2. We branch on every
    # choice and sum the tolls of all platforms landed on. This is exponential
    # but obviously correct for tiny n, and uses a different code path (plain
    # branching recursion, no DP table) than the iterative solution.
    best = [float("inf")]

    def walk(i, acc):
        # standing on platform i having already paid 'acc' (which includes t[i])
        if i == n - 1:
            best[0] = min(best[0], acc)
            return
        if i + 1 <= n - 1:
            walk(i + 1, acc + t[i + 1])
        if i + 2 <= n - 1:
            walk(i + 2, acc + t[i + 2])

    walk(0, t[0])
    print(best[0])

main()
