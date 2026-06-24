import sys
from math import gcd

def solve(n):
    # Count unordered pairs {x, y} with 1 <= x < y <= n and lcm(x, y) <= n.
    cnt = 0
    for x in range(1, n + 1):
        for y in range(x + 1, n + 1):
            g = gcd(x, y)
            l = x // g * y
            if l <= n:
                cnt += 1
    return cnt

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    print(solve(n))

if __name__ == "__main__":
    main()
