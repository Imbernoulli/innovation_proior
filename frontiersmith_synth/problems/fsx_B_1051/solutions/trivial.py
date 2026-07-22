# TIER: trivial
"""March the clans in plain lexicographic (house, guild) order -- exactly the
checker's own reference construction. Always feasible; scores ~0.1."""
import sys


def main():
    data = sys.stdin.read().split()
    n1, n2 = int(data[0]), int(data[1])
    out = []
    for a in range(n1):
        for b in range(n2):
            out.append(f"{a} {b}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
