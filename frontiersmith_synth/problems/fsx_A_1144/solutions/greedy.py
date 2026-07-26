# TIER: greedy
"""
The "obvious first idea": scan the voxels row by row (fixed y,z), find
maximal contiguous runs along x, and build EACH DISTINCT RUN LENGTH once
(L units + (L-1) unions for a length-L run) -- then, if the SAME run length
has already been built, just translate the cached run into place (1 op)
instead of rebuilding it. Finally union everything with a balanced tree.

This is a single, local, non-recursive optimization: "cache and reuse
identical runs". It rewards periodic / block-like structure very well (it
will do great on test 1's solid cube and test 2's repeated layer), but it
never looks for structure ACROSS rows, layers, or scales -- so on a genuine
multi-level fractal (whose rows are irregular, rarely identical, and rarely
long) it barely beats placing voxels one at a time. That gap is the trap:
"detect the one repeat I can see" is not the same insight as "exploit
self-similarity across the whole recursive structure" (see strong.py).
"""
import sys
from collections import defaultdict


def main():
    data = sys.stdin.buffer.read().split()
    v = int(data[0])
    if v == 0:
        return
    pts = [(int(data[1 + 3 * k]), int(data[2 + 3 * k]), int(data[3 + 3 * k])) for k in range(v)]

    rows = defaultdict(list)  # (y,z) -> sorted list of x
    for (x, y, z) in pts:
        rows[(y, z)].append(x)

    out = []

    def emit(line):
        out.append(line)
        return len(out) - 1

    unit = emit("U")

    run_cache = {}   # run length -> assembly index of a canonical [0,L) run along x
    piece_idx = []   # assembly index for every run actually placed (to be unioned)

    for (y, z), xs in rows.items():
        xs.sort()
        i = 0
        n = len(xs)
        while i < n:
            j = i
            while j + 1 < n and xs[j + 1] == xs[j] + 1:
                j += 1
            run_len = j - i + 1
            start_x = xs[i]

            if run_len not in run_cache:
                if run_len == 1:
                    canon = unit
                else:
                    # build a canonical [0,L) run by placing L separate unit
                    # translates and combining them -- NO doubling (that is
                    # exactly the insight this tier lacks).
                    idxs = [emit("T %d %d 0 0" % (unit, k)) for k in range(run_len)]
                    level = idxs
                    while len(level) > 1:
                        nxt = []
                        p, m = 0, len(level)
                        while p + 1 < m:
                            nxt.append(emit("M %d %d" % (level[p], level[p + 1])))
                            p += 2
                        if p < m:
                            nxt.append(level[p])
                        level = nxt
                    canon = level[0]
                run_cache[run_len] = canon

            canon = run_cache[run_len]
            placed = emit("T %d %d %d %d" % (canon, start_x, y, z))
            piece_idx.append(placed)
            i = j + 1

    level = piece_idx
    while len(level) > 1:
        nxt = []
        p, m = 0, len(level)
        while p + 1 < m:
            nxt.append(emit("M %d %d" % (level[p], level[p + 1])))
            p += 2
        if p < m:
            nxt.append(level[p])
        level = nxt

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
