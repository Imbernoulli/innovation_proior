# TIER: invalid
"""Emits stations outside the gallery bounds -> feasibility gate fails -> 0."""
import sys


def main():
    tok = sys.stdin.read().split()
    n = int(tok[0])
    out = ["9.000000 9.000000" for _ in range(n)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
