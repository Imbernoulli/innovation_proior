# TIER: trivial
"""Do-nothing adhesion: J = 0 everywhere. No preference between any pair of
types, so the deterministic swap dynamics never finds a strictly-improving
move and the arrangement stays exactly at its random initial state.
Reproduces the checker's own internal baseline construction exactly."""
import sys


def main():
    data = sys.stdin.read().split()
    T = int(data[1])
    for _ in range(T):
        print(" ".join("0" for _ in range(T)))


if __name__ == "__main__":
    main()
