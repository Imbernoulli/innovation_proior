import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    it = iter(data)
    n = int(next(it))
    a = [int(next(it)) for _ in range(n)]

    # Longest length L (1 <= L <= n) such that some contiguous block of length L
    # occurs at least twice at DIFFERENT starting positions (overlap allowed).
    # If no value repeats as such a block, answer is 0.
    best = 0
    # try all lengths from large to small; brute O(n^3) total with slicing tuples
    for L in range(n, 0, -1):
        seen = set()
        found = False
        for i in range(0, n - L + 1):
            t = tuple(a[i:i + L])
            if t in seen:
                found = True
                break
            seen.add(t)
        if found:
            best = L
            break
    print(best)

main()
