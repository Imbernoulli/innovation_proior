import sys

def solve(data):
    it = iter(data)
    n = int(next(it))
    L = int(next(it)); R = int(next(it))
    f = [int(next(it)) for _ in range(n)]
    MOD = 1_000_000_007
    cnt = 0
    # Brute: every unordered pair {i,j}, i<j, count once if L <= |f[i]-f[j]| <= R.
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(f[i] - f[j])
            if L <= d <= R:
                cnt += 1
    return cnt % MOD

def main():
    data = sys.stdin.read().split()
    print(solve(data))

if __name__ == "__main__":
    main()
