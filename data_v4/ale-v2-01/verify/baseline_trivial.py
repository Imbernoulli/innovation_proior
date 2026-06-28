#!/usr/bin/env python3
"""Trivial baseline: each client on its own route (always feasible since each
demand <= cap). Reads instance on stdin, writes a solution on stdout."""
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    n = int(next(it)); cap = int(next(it))
    next(it); next(it)  # depot
    for _ in range(n):
        next(it); next(it); next(it)
    out = [str(n)]
    for c in range(1, n + 1):
        out.append(f"1 {c}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
