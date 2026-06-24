import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    try:
        n = int(next(it))
    except StopIteration:
        return
    L = int(next(it))
    R = int(next(it))
    a = [int(next(it)) for _ in range(n)]

    # Exhaustive O(n^2): for every window [l, r] (0-based, l<=r), sum and test band [L,R] inclusive.
    count = 0
    for l in range(n):
        s = 0
        for r in range(l, n):
            s += a[r]
            if L <= s <= R:
                count += 1
    print(count)

if __name__ == "__main__":
    main()
