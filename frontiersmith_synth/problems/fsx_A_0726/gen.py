#!/usr/bin/env python3
"""gen.py <testId> -> one carved-percolation-throughput instance on stdout.

Dwarves are widening a mountain aquifer to feed a city cistern. Row 0 is the aquifer
layer (always a fixed-pressure water source, every column); a single deep cell is the
city cistern (a fixed-pressure sink). Between them lies porous rock of varying base
permeability, including a few high-permeability mineral veins. The dwarves may carve
up to B cells into free-flowing tunnel (permeability C_TUNNEL); everything else stays
plain rock.

testId 1..10 is a size/difficulty ladder (bigger grids, more veins on the later/odd-ish
ids). Seeded by testId only -- bit-for-bit reproducible.
"""
import random
import sys

C_TUNNEL = 5000
P_IN = 1_000_000
P_OUT = 0
ITERS_BASE = 26  # relaxation iterations = ITERS_BASE * (R + C)


def make_instance(t):
    R = 8 + (t - 1)          # 8..17
    C = 10 + (t - 1)         # 10..19
    B = int(round(6.5 * R))
    r_out, c_out = R - 1, C // 2
    iters = ITERS_BASE * (R + C)

    rnd = random.Random(913_000 + 97 * t)
    perm = [[rnd.randint(1, 6) for _ in range(C)] for _ in range(R)]

    # aquifer layer: row 0 is a fixed-pressure source, but it still feeds the rock at
    # ordinary rock permeability -- touching it is NOT a free unlimited tap; a carved
    # column only gets an ordinary seepage rate in, same as plain rock would. The real
    # payoff is spreading contact along many separate columns (parallel seepage) rather
    # than one wide corridor whose interior cells touch no rock at all.
    # city cistern: fixed-pressure sink, kept free-flowing (only the aquifer SIDE is the
    # deliberately scarce resource -- the collection problem the dwarves must solve).
    perm[r_out][c_out] = C_TUNNEL

    # planted mineral vein(s) on trap test cases: a winding high-permeability thread
    # that tempts a "find the best straight column band" search but does not itself
    # form a wide corridor -- a solid block still wastes most of its own budget on
    # cells that never touch the vein or the rock at all.
    VEIN_IDS = {3, 4, 6, 7, 9, 10}
    if t in VEIN_IDS:
        n_veins = 1 if t < 7 else 2
        for _ in range(n_veins):
            c = rnd.randint(1, C - 2)
            for r in range(1, r_out):
                if perm[r][c] < 14:
                    perm[r][c] = 14
                c += rnd.choice([-1, 0, 0, 1])
                c = max(1, min(C - 2, c))

    lines = ["%d %d %d %d %d" % (R, C, B, r_out, c_out),
             "%d %d %d %d" % (P_IN, P_OUT, C_TUNNEL, iters)]
    for r in range(R):
        lines.append(" ".join(str(x) for x in perm[r]))
    return "\n".join(lines) + "\n"


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    t = int(sys.argv[1])
    t = max(1, min(10, t))
    sys.stdout.write(make_instance(t))


if __name__ == "__main__":
    main()
