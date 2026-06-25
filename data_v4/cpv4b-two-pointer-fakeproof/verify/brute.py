import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    K = int(data[idx]); idx += 1
    a = [int(data[idx + i]) for i in range(n)]
    idx += n

    # Brute force: enumerate every subarray, compute its OR directly, count OR <= K.
    ans = 0
    for i in range(n):
        cur = 0
        for j in range(i, n):
            cur |= a[j]
            if cur <= K:
                ans += 1
    print(ans)

if __name__ == "__main__":
    main()
