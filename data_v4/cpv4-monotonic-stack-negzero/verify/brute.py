import sys

def solve(a):
    n = len(a)
    # Empty subarray allowed, scores 0.
    best = 0
    for l in range(n):
        m = a[l]
        for r in range(l, n):
            if a[r] < m:
                m = a[r]
            score = m * (r - l + 1)
            if score > best:
                best = score
    return best

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    idx = 0
    n = int(data[idx]); idx += 1
    a = [int(data[idx + i]) for i in range(n)]
    print(solve(a))

if __name__ == "__main__":
    main()
