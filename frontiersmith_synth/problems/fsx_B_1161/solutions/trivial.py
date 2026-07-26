# TIER: trivial
"""Fill every torn (hidden) cell with 0. Feasible but ignores all structure --
matches the checker's own internal baseline construction almost exactly."""
import sys


def main():
    data = sys.stdin.read().split()
    m = int(data[0]); p = int(data[1])
    tokens = data[2:2 + m * m]
    out_rows = []
    idx = 0
    for i in range(m):
        row = []
        for j in range(m):
            tok = tokens[idx]; idx += 1
            row.append("0" if tok == "?" else tok)
        out_rows.append(" ".join(row))
    sys.stdout.write("\n".join(out_rows) + "\n")


if __name__ == "__main__":
    main()
