# TIER: greedy
# "Obvious" approach: spread the minimum number of parity cells (R+C-1) across
# the grid so every row and every column keeps at least one raw cell (avoids
# the classic mistake of dedicating one whole row + one whole column, which
# would leave those lines with zero raw data). This part is a genuine, correct
# idea -- independent-per-line parity generalized into a spread layout.
#
# The trap: for the actual parity FORMULA, this solution tags each parity
# cell with a small "diagonal offset" x = (3*i + 5*j) mod m using a modulus m
# that only grows half as fast as the number of parity cells. Many parity
# cells collide on the same tag, so their Cauchy-style coefficient rows become
# linearly DEPENDENT -- the rank bound is not hit even though the raw-cell
# count is near-optimal. Independent per-line reasoning about *placement* is
# not enough; the *coefficients* need a genuinely diagonal (collision-free)
# chain too.
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

    modulus = max(3, (R + C - 1) // 2)

    out = []
    for i in range(R):
        for j in range(C):
            if (i, j) not in parity_cells:
                out.append("D")
            else:
                x = (3 * i + 5 * j) % modulus
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
