# TIER: strong
"""The insight: the ripple-carry gadget from `greedy` does not actually
depend on WHICH bit row it sits in -- "old bit" and "carry in/out" are the
only state a cell ever needs, and that state means the same thing at every
row. So the SAME 4 cooperative (south+west, temperature-2) tile types can be
reused at every interior row, instead of instantiating a fresh copy per row.
Only the two hard grid boundaries survive as dedicated types: row 0 (no
south neighbor at all -> a 2-type west-only LSB flip-flop) and column 0 (no
west neighbor -> a 1-type south-only zero-filler). Total tile-type count is
O(1): a single reusable cooperative gadget counts out a window of ANY width
n and ANY bit-depth W, where per-cell tiling wastes linearly and even the
row-tagged recipe still wastes O(log n)."""
import sys

NONE = (".", 0)
IW0, IW1 = "sIW0", "sIW1"
IS0, IS1 = "sIS0", "sIS1"
GC0 = "gC0"


def main():
    n = int(sys.stdin.read().split()[0])
    W = max(1, (n - 1).bit_length()) if n >= 2 else 1

    types = []

    def add(N, S, E, W_, value):
        types.append({"N": N, "S": S, "E": E, "W": W_, "value": value})
        return len(types) - 1

    seed_id = add(N=(GC0, 2) if W > 1 else NONE, S=NONE, E=("R0_0", 2), W_=NONE, value=0)
    add(N=(IS0, 1) if W > 1 else NONE, S=NONE, E=("R0_1", 2), W_=("R0_0", 2), value=1)  # r0_new1
    add(N=(IS1, 1) if W > 1 else NONE, S=NONE, E=("R0_0", 2), W_=("R0_1", 2), value=0)  # r0_new0

    if W > 1:
        # column-0 vertical filler: ONE reusable type for every row 1..W-1
        add(N=(GC0, 2), S=(GC0, 2), E=(IW0, 1), W_=NONE, value=0)
        # the single reusable cooperative gadget, valid at EVERY interior row
        add(N=(IS0, 1), S=(IS0, 1), E=(IW0, 1), W_=(IW0, 1), value=0)  # copy0
        add(N=(IS0, 1), S=(IS0, 1), E=(IW1, 1), W_=(IW1, 1), value=1)  # copy1
        add(N=(IS0, 1), S=(IS1, 1), E=(IW1, 1), W_=(IW0, 1), value=1)  # flip_to1
        add(N=(IS1, 1), S=(IS1, 1), E=(IW0, 1), W_=(IW1, 1), value=0)  # flip_to0

    out = [str(len(types))]
    for tid, t in enumerate(types):
        N, S, E, Wg = t["N"], t["S"], t["E"], t["W"]
        out.append("%d %s %d %s %d %s %d %s %d %d" %
                    (tid, N[0], N[1], S[0], S[1], E[0], E[1], Wg[0], Wg[1], t["value"]))
    out.append(str(seed_id))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
