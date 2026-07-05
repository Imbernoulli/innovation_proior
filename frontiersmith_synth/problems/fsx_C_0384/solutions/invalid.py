# TIER: invalid
"""Build block 0 at EVERY cell. This violates the Latin rule (block 0 repeats in
every row and column) and clobbers givens, so the checker must reject it -> 0."""
import sys


def main():
    tok = sys.stdin.read().split()
    n = int(tok[0])
    out = []
    for i in range(n):
        out.append(" ".join("0" for _ in range(n)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
