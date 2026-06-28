#!/usr/bin/env python3
import sys

def digit_sum(x):
    s = 0
    while x > 0:
        s += x % 10
        x //= 10
    return s

def num_len(x):
    return len(str(x))

def solve(L, R):
    cnt = 0
    for x in range(L, R + 1):
        if x <= 0:
            continue
        if digit_sum(x) % num_len(x) == 0:
            cnt += 1
    return cnt

def main():
    data = sys.stdin.read().split()
    L = int(data[0]); R = int(data[1])
    print(solve(L, R))

if __name__ == "__main__":
    main()
