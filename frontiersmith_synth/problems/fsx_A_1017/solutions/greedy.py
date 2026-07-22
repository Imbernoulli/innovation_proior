# TIER: greedy
"""Cooperative (south+west) binary-counter tile set, implemented the direct
"textbook" way: a fresh 4-type ripple-carry gadget is instantiated PER BIT
ROW (glue vocabulary tagged by row index), plus a 2-type LSB flip-flop for
row 0 and a per-row vertical filler for the always-zero column 0. This is
the recipe an average coder writes once they know "cooperative binding
helps": O(W) tile types -- a real win over per-cell tiling, but it stops
short of noticing the row-to-row gadgets are IDENTICAL and reusable, so it
never gets below O(W)."""
import sys


def main():
    n = int(sys.stdin.read().split()[0])
    W = max(1, (n - 1).bit_length()) if n >= 2 else 1

    types = []  # list of dict(N,S,E,W,value)

    def add(N, S, E, W_, value):
        types.append({"N": N, "S": S, "E": E, "W": W_, "value": value})
        return len(types) - 1

    NONE = (".", 0)

    # row 0: seed + 2-type LSB flip-flop (west-only, strength-2 boundary glue)
    seed_id = add(
        N=("gC0_1", 2) if W > 1 else NONE,
        S=NONE, E=("R0_0", 2), W_=NONE, value=0)
    r0_new1 = add(N=("gIS0_1", 1) if W > 1 else NONE,
                   S=NONE, E=("R0_1", 2), W_=("R0_0", 2), value=1)
    r0_new0 = add(N=("gIS1_1", 1) if W > 1 else NONE,
                   S=NONE, E=("R0_0", 2), W_=("R0_1", 2), value=0)

    # column 0 (always all-zero) and the row-tagged cooperative gadget
    for b in range(1, W):
        has_above = b + 1 < W
        # column-0 vertical filler, private to row b
        add(N=("gC0_%d" % (b + 1), 2) if has_above else NONE,
            S=("gC0_%d" % b, 2), E=("gIW0_%d" % b, 1), W_=NONE, value=0)
        IW0, IW1 = "gIW0_%d" % b, "gIW1_%d" % b
        IS0, IS1 = "gIS0_%d" % b, "gIS1_%d" % b
        up0 = ("gIS0_%d" % (b + 1), 1) if has_above else NONE
        up1 = ("gIS1_%d" % (b + 1), 1) if has_above else NONE
        # copy0: carry_in=0, old=0 -> new=0, carry_out=0
        add(N=up0, S=(IS0, 1), E=(IW0, 1), W_=(IW0, 1), value=0)
        # copy1: carry_in=0, old=1 -> new=1, carry_out=0
        add(N=up0, S=(IS0, 1), E=(IW1, 1), W_=(IW1, 1), value=1)
        # flip_to1: carry_in=1, old=0 -> new=1, carry_out=0
        add(N=up0, S=(IS1, 1), E=(IW1, 1), W_=(IW0, 1), value=1)
        # flip_to0: carry_in=1, old=1 -> new=0, carry_out=1
        add(N=up1, S=(IS1, 1), E=(IW0, 1), W_=(IW1, 1), value=0)

    out = [str(len(types))]
    for tid, t in enumerate(types):
        N, S, E, Wg = t["N"], t["S"], t["E"], t["W"]
        out.append("%d %s %d %s %d %s %d %s %d %d" %
                    (tid, N[0], N[1], S[0], S[1], E[0], E[1], Wg[0], Wg[1], t["value"]))
    out.append(str(seed_id))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
