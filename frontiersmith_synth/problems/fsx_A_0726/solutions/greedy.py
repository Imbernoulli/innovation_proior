# TIER: greedy
# The obvious approach: "flow needs a wide pipe". Scan for the vertical band of columns
# with the best cumulative rock permeability (so it also chases any mineral vein), then
# spend the WHOLE budget widening a single solid corridor there, from just below the
# aquifer down toward the outlet, using any leftover budget to fatten the bottom rows.
import sys


def main():
    toks = sys.stdin.read().split()
    idx = 0
    R, C, B, r_out, c_out = (int(toks[idx + k]) for k in range(5)); idx += 5
    P_IN, P_OUT, C_TUNNEL, ITERS = (int(toks[idx + k]) for k in range(4)); idx += 4
    perm = []
    for r in range(R):
        row = [int(toks[idx + k]) for k in range(C)]
        idx += C
        perm.append(row)

    depth = max(1, r_out - 1)  # rows 1..r_out-1 are carve-able rock rows
    col_sum = [sum(perm[r][c] for r in range(1, r_out)) for c in range(C)]

    w = max(1, min(C, B // depth))
    # the corridor must actually reach the outlet's column, so search windows containing c_out
    lo_c0 = max(0, c_out - w + 1)
    hi_c0 = min(C - w, c_out)
    best_c0, best_val = lo_c0, -1
    for c0 in range(lo_c0, hi_c0 + 1):
        val = sum(col_sum[c0:c0 + w])
        if val > best_val:
            best_val, best_c0 = val, c0

    cells = []
    for c in range(best_c0, best_c0 + w):
        for r in range(1, r_out):
            cells.append((r, c))

    # widen further with any leftover budget: add extra columns adjacent to the band
    used = len(cells)
    left, right = best_c0 - 1, best_c0 + w
    while used < B and (left >= 0 or right < C):
        if right < C:
            for r in range(1, r_out):
                if used >= B:
                    break
                cells.append((r, right))
                used += 1
            right += 1
        if used < B and left >= 0:
            for r in range(1, r_out):
                if used >= B:
                    break
                cells.append((r, left))
                used += 1
            left -= 1
        if left < 0 and right >= C:
            break

    cells = cells[:B]
    out = [str(len(cells))]
    for (r, c) in cells:
        out.append("%d %d" % (r, c))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
