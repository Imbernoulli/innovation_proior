# TIER: trivial
"""Pack all B sites into one contiguous block centered in the strip -- the
checker's own reference construction (always feasible, always positive)."""
import sys


def main():
    data = sys.stdin.read().split()
    pos = 0
    L = int(data[pos]); pos += 1
    B = int(data[pos]); pos += 1
    # remaining physics tokens are unused by this construction
    active = [0] * L
    start = (L - B) // 2
    for i in range(start, start + B):
        active[i] = 1
    sys.stdout.write(" ".join(str(x) for x in active) + "\n")


if __name__ == "__main__":
    main()
