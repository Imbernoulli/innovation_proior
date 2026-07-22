# TIER: trivial
"""Unmodulated phase plate: theta = 0 everywhere. Reproduces the checker's own
baseline construction, so this scores ~0.1 by design."""
import sys


def main():
    toks = sys.stdin.read().split()
    N = int(toks[0])
    out = [str(N)]
    row = " ".join(["0.0"] * N)
    for _ in range(N):
        out.append(row)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
