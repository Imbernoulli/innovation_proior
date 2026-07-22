# TIER: greedy
"""The obvious "reuse what's identical" optimization over the trivial map:
notice a whole ROW looks the same left-to-right, so give each ROW its own
tile type (reused across all k columns) instead of one type per cell. A
single-bond (strength-2) vertical glue chains row r to row r+1; the very
last row exposes no north glue so growth halts. Row 0 needs one tile per
column just to physically span the width k (a fixed, one-time O(k) setup
cost, same for every recipe); for k > 1 all columns except the leftmost
additionally grow one row taller than it (the deterministic "flag" row
any halting construction leaves behind), so column 0 uses its own "main"
row-index chain and columns 1..k-1 share a one-row-longer "cap" chain.
This reuses types across a row but still needs one FRESH type PER ROW to
know "how far up we are" -- a plain unary counter, not a machine that
reasons about the target size -- so its cost stays linear in T. It never
lets one column "talk to" its neighbor column (no west/east glue is ever
used past row 0), which is exactly the interaction the insight below
exploits.
"""
import sys


def main():
    T = int(sys.stdin.read().split()[0])
    k = max(1, T.bit_length())
    height = (T + 1) if k == 1 else (2 * T + 1)

    label = [1]

    def new_label():
        v = label[0]
        label[0] += 1
        return v

    types = []  # [Nl,Ns,El,Es,Sl,Ss,Wl,Ws]

    def emit(N=(0, 0), E=(0, 0), S=(0, 0), W=(0, 0)):
        types.append([N[0], N[1], E[0], E[1], S[0], S[1], W[0], W[1]])
        return len(types)

    if k == 1:
        row_out = [new_label() for _ in range(height)]
        seed_id = emit(N=(row_out[0], 2))
        for r in range(1, height):
            north = (row_out[r], 2) if r < height - 1 else (0, 0)
            emit(N=north, S=(row_out[r - 1], 2))
        out = [str(len(types))]
        for row in types:
            out.append(" ".join(map(str, row)))
        out.append(str(seed_id))
        sys.stdout.write("\n".join(out) + "\n")
        return

    # k > 1: column 0 uses the "main" chain (stops at row height-1); columns
    # 1..k-1 share a "cap" chain, one row longer (stops at row height).
    row_main = [new_label() for _ in range(height)]
    row_cap = [new_label() for _ in range(height + 1)]
    CHAIN = [new_label() for _ in range(k - 1)]

    # ---- row 0: span the k columns (private per-column hand-off, else two
    # different row-0 tile types could both bind at the same frontier cell) ----
    seed_id = emit(N=(row_main[0], 2), E=(CHAIN[0], 2))
    for c in range(1, k):
        w = (CHAIN[c - 1], 2)
        n = (row_cap[0], 2)
        if c < k - 1:
            emit(N=n, E=(CHAIN[c], 2), W=w)
        else:
            emit(N=n, W=w)

    # ---- main chain: column 0 only, one fresh type per row 1..height-1 ----
    for r in range(1, height):
        north = (row_main[r], 2) if r < height - 1 else (0, 0)
        emit(N=north, S=(row_main[r - 1], 2))

    # ---- cap chain: columns 1..k-1, one fresh type per row 1..height ----
    for r in range(1, height + 1):
        north = (row_cap[r], 2) if r < height else (0, 0)
        emit(N=north, S=(row_cap[r - 1], 2))

    out = [str(len(types))]
    for row in types:
        out.append(" ".join(map(str, row)))
    out.append(str(seed_id))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
