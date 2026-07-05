# TIER: trivial
# Install nothing new: echo back exactly the pre-installed turbines.
# Reproduces the checker's baseline B (=#givens) -> Ratio ~ 0.1.
import sys


def main():
    tok = sys.stdin.read().split()
    n = int(tok[0])
    vals = tok[1:1 + n * n]
    lines = []
    idx = 0
    for i in range(n):
        row = []
        for j in range(n):
            t = vals[idx]; idx += 1
            row.append("." if t == '.' else t)
        lines.append(" ".join(row))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
