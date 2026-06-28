#!/usr/bin/env python3
import sys


def brute_force(items, capacity):
    best = 0
    n = len(items)
    for mask in range(1 << n):
        weight = 0
        value = 0
        for i, (w, v) in enumerate(items):
            if mask & (1 << i):
                weight += w
                value += v
        if weight <= capacity and value > best:
            best = value
    return best


def main():
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return
    n, capacity = data[0], data[1]
    items = [(data[i], data[i + 1]) for i in range(2, 2 + 2 * n, 2)]
    print(brute_force(items, capacity))


if __name__ == "__main__":
    main()
