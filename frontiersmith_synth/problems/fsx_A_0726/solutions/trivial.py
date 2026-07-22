# TIER: trivial
# Reproduces the checker's own baseline: a single 1-cell-wide straight line directly
# above the outlet, using min(B, R-1) cells. Ignores the rock permeability map entirely
# and never touches any leftover budget.
import sys


def main():
    toks = sys.stdin.read().split()
    R, C, B, r_out, c_out = (int(toks[i]) for i in range(5))
    # (P_IN, P_OUT, C_TUNNEL, ITERS and the permeability grid are ignored.)

    cells = []
    r = r_out - 1
    used = 0
    while r >= 1 and used < B and used < R - 1:
        cells.append((r, c_out))
        r -= 1
        used += 1

    out = [str(len(cells))]
    for (rr, cc) in cells:
        out.append("%d %d" % (rr, cc))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
