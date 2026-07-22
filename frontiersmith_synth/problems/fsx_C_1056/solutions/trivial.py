# TIER: trivial
"""Identity assignment: symbol i -> slot i. Reproduces the checker's own
baseline construction exactly, so this scores ~0.1."""
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    print(" ".join(str(i) for i in range(n)))


if __name__ == "__main__":
    main()
