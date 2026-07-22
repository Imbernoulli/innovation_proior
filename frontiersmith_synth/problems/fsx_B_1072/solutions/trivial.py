# TIER: trivial
import sys


def main():
    d = sys.stdin.read().split()
    a = int(d[0]); k = int(d[1]); L = int(d[2])
    # Reproduce the checker's own baseline: a single constant symbol repeated L times.
    print("0" * L)


if __name__ == "__main__":
    main()
