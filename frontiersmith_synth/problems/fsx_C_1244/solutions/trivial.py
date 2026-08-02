# TIER: trivial
"""Identity placement: cell i -> slot i. Reproduces the checker's own baseline
construction exactly, so this always scores Ratio = 0.1 (when identity is
feasible, which the generator guarantees)."""
import sys


def main():
    data = sys.stdin.read().split()
    n_cells = int(data[0])
    print(" ".join(str(i) for i in range(n_cells)))


if __name__ == "__main__":
    main()
