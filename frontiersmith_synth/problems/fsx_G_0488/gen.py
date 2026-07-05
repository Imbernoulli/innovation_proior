#!/usr/bin/env python3
"""gen.py <testId>  -> prints ONE instance (a single odd prime p) to stdout.

testId 1..10 is a difficulty ladder: p grows geometrically so the search space
(and the size of a good progression-free set) increases with the id. Fully
deterministic: p is a pure function of testId, no randomness.
"""
import sys


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def next_prime(n: int) -> int:
    if n <= 2:
        return 2
    if n % 2 == 0:
        n += 1
    while not is_prime(n):
        n += 2
    return n


def prime_for(test_id: int) -> int:
    # geometric ladder: base 90 * 1.85^(t-1), snapped up to the next prime.
    b = 90.0
    p = next_prime(int(b))
    for _ in range(1, test_id):
        b *= 1.85
        p = next_prime(int(b))
    return p


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(2)
    t = int(sys.argv[1])
    if t < 1:
        t = 1
    if t > 10:
        t = 10
    print(prime_for(t))


if __name__ == "__main__":
    main()
