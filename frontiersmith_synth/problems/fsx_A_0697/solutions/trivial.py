# TIER: trivial
"""Bump allocator: never reuse shelf space (matches the checker's own baseline)."""
import sys


def main():
    data = sys.stdin.read().split()
    pos = 0

    def next_int():
        nonlocal pos
        v = int(data[pos])
        pos += 1
        return v

    N = next_int()
    M = next_int()
    next_int()  # PAGE
    next_int()  # LAMBDA

    sizes = []
    for _ in range(N):
        size = next_int()
        next_int()  # birth
        next_int()  # death
        sizes.append(size)
    # skip checks entirely
    for _ in range(M):
        next_int()
        next_int()

    out = []
    running = 0
    for i in range(N):
        out.append(str(running))
        running += sizes[i]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
