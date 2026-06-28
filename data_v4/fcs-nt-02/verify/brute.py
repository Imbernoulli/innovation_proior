#!/usr/bin/env python3
import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0]); r = int(data[1]); m = int(data[2])
    if m == 1:
        print(0); return
    if r < 0 or r > n:
        print(0 % m); return
    # Direct: compute exact nCr via Python big integers, then mod m.
    # Use the multiplicative formula with integer division (exact).
    r = min(r, n - r)
    num = 1
    for i in range(r):
        num = num * (n - i) // (i + 1)
    print(num % m)

if __name__ == "__main__":
    main()
