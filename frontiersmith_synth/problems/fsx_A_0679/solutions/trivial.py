# TIER: trivial
# Minimal-covering-diagonal baseline: only ~max(R,C) raw cells (one per row,
# one per column via a wrap-around diagonal), everything else is parity with
# a globally-injective Cauchy-style tag. Always fully recoverable (way more
# parity than needed) but sacrifices almost all raw data -> low score.
import sys


def main():
    R, C, p = map(int, sys.stdin.read().split())

    raw_cells = set()
    for i in range(R):
        raw_cells.add((i, i % C))
    for j in range(C):
        raw_cells.add((j % R, j))

    data_idx = {}
    cnt = 0
    for i in range(R):
        for j in range(C):
            if (i, j) in raw_cells:
                data_idx[(i, j)] = cnt
                cnt += 1

    out = []
    for i in range(R):
        for j in range(C):
            if (i, j) in raw_cells:
                out.append("D")
            else:
                x = (i * C + j + R * C + 1000) % p
                terms = []
                for (ii, jj), y in data_idx.items():
                    denom = (x - y) % p
                    if denom == 0:
                        denom = 1
                    terms.append((y, pow(denom, p - 2, p)))
                parts = ["P", str(len(terms))]
                for y, co in terms:
                    parts.append(str(y))
                    parts.append(str(co))
                out.append(" ".join(parts))

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
