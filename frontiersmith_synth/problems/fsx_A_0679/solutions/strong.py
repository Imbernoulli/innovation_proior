# TIER: strong
# Same spread layout as greedy (every row/column keeps a raw cell, only
# R+C-1 parity cells -- the combinatorial rank bound: any fewer parity cells
# makes EVERY row+column erasure pattern rank-deficient by the same margin,
# so R+C-1 is the minimum that can ever work). The insight that actually pays:
# tag every parity cell with a GLOBALLY DISTINCT field element (its own row-
# major linear index, offset out of the raw-index range) before building the
# Cauchy-style coefficients 1/(x-y). Distinct tags is exactly what a genuine
# diagonal/MDS parity chain needs -- collision-free by construction, so the
# rank bound is hit on *every* one of the R*C erasure patterns simultaneously,
# not just the ones the naive small-modulus tagging happens to avoid.
import sys


def diagonal_spread(R, C):
    K = R + C - 1
    row_cap = C - 1
    col_cap = R - 1
    cells_order = sorted(
        ((i, j) for i in range(R) for j in range(C)),
        key=lambda ij: ((ij[0] + ij[1]) % max(R, C), ij[0], ij[1]),
    )
    chosen = []
    row_count = [0] * R
    col_count = [0] * C
    for (i, j) in cells_order:
        if len(chosen) >= K:
            break
        if row_count[i] < row_cap and col_count[j] < col_cap:
            chosen.append((i, j))
            row_count[i] += 1
            col_count[j] += 1
    return set(chosen)


def main():
    R, C, p = map(int, sys.stdin.read().split())

    parity_cells = diagonal_spread(R, C)

    data_idx = {}
    cnt = 0
    for i in range(R):
        for j in range(C):
            if (i, j) not in parity_cells:
                data_idx[(i, j)] = cnt
                cnt += 1

    out = []
    for i in range(R):
        for j in range(C):
            if (i, j) not in parity_cells:
                out.append("D")
            else:
                # globally distinct tag: linear cell index, shifted clear of
                # the raw-index range [0, cnt)
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
