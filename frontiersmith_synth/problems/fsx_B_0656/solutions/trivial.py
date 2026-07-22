# TIER: trivial
"""Uniform mixture over every pigment in the palette (the checker's own
internal baseline construction)."""
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0
    N = int(data[idx]); idx += 1
    M = int(data[idx]); idx += 1
    # K (illuminant count) not needed here
    w = 1.0 / M
    print(" ".join(f"{w:.8f}" for _ in range(M)))


if __name__ == "__main__":
    main()
