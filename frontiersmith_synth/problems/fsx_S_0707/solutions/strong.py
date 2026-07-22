# TIER: strong
# Insight: peel off the planted rank-1 correction instead of synthesizing each
# row independently.
#
#   1. Classify rows by their OWN Hamming weight (adaptive largest-gap split,
#      no hardcoded threshold): a "light" cluster (weight ~3, the untouched
#      sparse rows) and a "heavy" cluster (weight ~m/2, the rows that picked
#      up the shared rank-1 offset).  Light rows are already minimal -- just
#      chain their own bits.
#   2. Recover the SHARED correction pattern T bit-by-bit via a per-column
#      MAJORITY VOTE across the heavy rows: since each heavy row is
#      T XOR (its own weight-3 residual), and residuals are sparse and
#      essentially independent across rows, the majority value at each
#      column recovers T's bit robustly -- this is exactly "detect the rank-1
#      correction via row-space differences" (pairwise heavy-row differences
#      cancel T and vote out sparse residual noise; the majority is the
#      cheapest way to read that off).  Build ONE shared chain for T (paid
#      once, shared chain #1).
#   3. For every heavy row, its own residual (row XOR T) is now sparse
#      (weight ~3) -- chain it (shared chain #2's *pattern*, one short chain
#      per row) and combine with the single precomputed T-line (1 XOR).  The
#      residue really is "sparse plus a shared correction": ~3 ops/row.
#
# This turns an O(#heavy^2) recipe into an O(m) one without ever explicitly
# recovering P, S, u, v -- it only uses row weights and row-space structure
# of the matrix actually given.
import sys


def build_chain(bitpos_list, next_line, ops):
    """XOR the input columns in bitpos_list (0-indexed) into one line.
    Returns (line_id, next_line)."""
    if not bitpos_list:
        return None, next_line
    cur = bitpos_list[0] + 1
    for c in bitpos_list[1:]:
        ops.append((cur, c + 1))
        cur = next_line
        next_line += 1
    return cur, next_line


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    m = int(next(it))
    rows = []
    for _ in range(m):
        s = next(it)
        bm = 0
        for j, c in enumerate(s):
            if c == "1":
                bm |= (1 << j)
        rows.append(bm)

    weights = [bin(bm).count("1") for bm in rows]

    # adaptive largest-gap split on the SORTED distinct weights
    distinct = sorted(set(weights))
    light_set = set(range(m))  # fallback: everyone is "light" (no pairing)
    if len(distinct) >= 2:
        gaps = [(distinct[i + 1] - distinct[i], distinct[i]) for i in range(len(distinct) - 1)]
        gaps.sort(key=lambda g: (-g[0], g[1]))
        cutoff = gaps[0][1]  # light weights are <= cutoff
        light_idx = [i for i in range(m) if weights[i] <= cutoff]
        heavy_idx = [i for i in range(m) if weights[i] > cutoff]
        if len(heavy_idx) >= 1:
            light_set = set(light_idx)
        else:
            heavy_idx = []
    else:
        heavy_idx = []
        light_idx = list(range(m))

    ops = []
    next_line = m + 1
    outrefs = [0] * m

    # light rows: direct chain each (already minimal for their own weight)
    for i in sorted(light_set):
        bits = [j for j in range(m) if (rows[i] >> j) & 1]
        line, next_line = build_chain(bits, next_line, ops)
        outrefs[i] = line if line is not None else 1

    heavy_list = [i for i in range(m) if i not in light_set]
    if len(heavy_list) == 1:
        i = heavy_list[0]
        bits = [j for j in range(m) if (rows[i] >> j) & 1]
        line, next_line = build_chain(bits, next_line, ops)
        outrefs[i] = line if line is not None else 1
    elif len(heavy_list) >= 2:
        # recover the shared correction T via per-column majority vote
        h = len(heavy_list)
        T_mask = 0
        for j in range(m):
            cnt = sum(1 for i in heavy_list if (rows[i] >> j) & 1)
            if cnt * 2 > h:
                T_mask |= (1 << j)

        t_bits = [j for j in range(m) if (T_mask >> j) & 1]
        t_line, next_line = build_chain(t_bits, next_line, ops)

        for i in heavy_list:
            resid = rows[i] ^ T_mask
            if resid == 0:
                outrefs[i] = t_line if t_line is not None else 1
                continue
            rbits = [j for j in range(m) if (resid >> j) & 1]
            rline, next_line = build_chain(rbits, next_line, ops)
            if t_line is None:
                outrefs[i] = rline
            else:
                ops.append((rline, t_line))
                outrefs[i] = next_line
                next_line += 1

    out = [str(len(ops))]
    for (a, b) in ops:
        out.append("%d %d" % (a, b))
    out.append(" ".join(map(str, outrefs)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
