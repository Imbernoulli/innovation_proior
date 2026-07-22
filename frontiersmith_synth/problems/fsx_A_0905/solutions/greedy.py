# TIER: greedy
"""
The obvious first approach: a SINGLE row-major pass, no search, no lookahead,
no notion of rings. At each cell, among colors that still have supply and
would not (right now) break the run-cap, pick by a blended score

    score(k) = remaining[k] - gamma * |v[k] - target(cell)|,   gamma = 2.2/vrange

i.e. mostly chase the numerically nearest color, but never fully deplete a
color that is comparatively scarce (the minimal amount of supply-awareness
needed to avoid painting yourself into a run-cap corner -- without it, a
pure argmin-error pass provably deadlocks on most of these instances). This
never thinks about WHICH ring a cell belongs to, so it has no way to notice
that a whole contiguous arc of near-identical targets is about to be forced
onto whatever color happens to still be legal -- it just reacts locally,
cell by cell, in scan order.
"""
import math
import sys

GAMMA_MULT = 1.8


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); c = int(next(it)); K = int(next(it))
    v = [int(next(it)) for _ in range(c)]
    cnt = [int(next(it)) for _ in range(c)]
    tcenter = int(next(it)); tedge = int(next(it))

    cy = cx = (n - 1) / 2.0
    rmax = math.hypot(cx, cy) if n > 1 else 1.0
    if rmax <= 0:
        rmax = 1.0

    def target(i, j):
        r = math.hypot(i - cy, j - cx)
        return tcenter + (tedge - tcenter) * (r / rmax)

    vrange = max(1, v[-1] - v[0])
    gamma = GAMMA_MULT / vrange

    remaining = list(cnt)
    col_last = [-1] * n
    col_run = [0] * n
    out_lines = []
    for i in range(n):
        row_last_c = -1
        row_run_c = 0
        row_vals = []
        for j in range(n):
            tgt = target(i, j)
            feas = []
            for k in range(c):
                if remaining[k] <= 0:
                    continue
                row_ok = not (row_last_c == k and row_run_c >= K)
                col_ok = not (col_last[j] == k and col_run[j] >= K)
                if not (row_ok and col_ok):
                    continue
                feas.append(k)
            best = -1
            if feas:
                best = max(feas, key=lambda k: remaining[k] - gamma * abs(v[k] - tgt))
            if best == -1:
                for relax_col in (False, True):
                    for relax_row in (False, True):
                        cc = []
                        for k in range(c):
                            if remaining[k] <= 0:
                                continue
                            row_ok = relax_row or not (row_last_c == k and row_run_c >= K)
                            col_ok = relax_col or not (col_last[j] == k and col_run[j] >= K)
                            if row_ok and col_ok:
                                cc.append(k)
                        if cc:
                            best = min(cc, key=lambda k: abs(v[k] - tgt))
                            break
                    if best != -1:
                        break
            row_vals.append(str(best + 1))
            remaining[best] -= 1
            if row_last_c == best:
                row_run_c += 1
            else:
                row_last_c = best
                row_run_c = 1
            if col_last[j] == best:
                col_run[j] += 1
            else:
                col_last[j] = best
                col_run[j] = 1
        out_lines.append(" ".join(row_vals))
    sys.stdout.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
