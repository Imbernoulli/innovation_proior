import sys, random

# gen.py <testId> -- prints ONE "warehouse pallet reshuffle" instance to stdout.
#
#   n m K D
#   seg[0] seg[1] ... seg[n-1]      (segment id 0..m-1 of every slot)
#   perm[0] perm[1] ... perm[n-1]   (perm[i] = pallet currently AT slot i; pallet i's home
#                                     is slot i)
#
# Plant (the trap): every instance is built from a handful of "relay cycles". A relay
# cycle picks ONE lone slot in a tiny dedicated "dock" segment (segment 0) as its anchor,
# plus a run of L-1 slots that all live in a SINGLE shared aisle segment, and chains them
# into one big permutation cycle  anchor -> a_1 -> a_2 -> ... -> a_{L-1} -> anchor.  Only
# the two ring edges touching the anchor cross a doorway (dock <-> aisle); every other
# consecutive pair in the ring lives inside the same aisle, so the ring itself is almost
# entirely local.  The anchor's raw slot INDEX is always the smallest in the whole cycle
# (dock indices are handed out first, 0..num_relay-1), so any left-to-right
# cycle-decomposition that then fans a star of transpositions out of the smallest index
# (the textbook "cycle following" algorithm) re-derives the anchor as its swap center and
# is forced to cross the doorway on EVERY single one of its L-1 star transpositions --
# even though the ring only truly needs to cross it twice. The remaining slots form short,
# fully local 2-cycles (calm control mass) and a few genuine fixed points.

BASE_SEED = 424242017


def build_instance(trap_legs, calm2, extra_fixed, m, K, D, seed):
    rng = random.Random(seed)
    num_trap = len(trap_legs)
    dock_size = num_trap
    n = dock_size + sum(trap_legs) + 2 * calm2 + extra_fixed
    seg = [0] * n
    perm = list(range(n))
    aisle_count = max(1, m - 1)
    cur = dock_size

    # -- relay (trap) cycles: anchor in dock (segment 0) + a run in one aisle segment --
    trap_order = list(range(num_trap))
    rng.shuffle(trap_order)  # which aisle each relay lands in is shuffled, not the legs
    for i in range(num_trap):
        leg = trap_legs[i]
        anchor = i
        aisle_seg = 1 + (trap_order[i] % aisle_count)
        slots = list(range(cur, cur + leg))
        cur += leg
        for s in slots:
            seg[s] = aisle_seg
        chain = [anchor] + slots
        for a, b in zip(chain, chain[1:] + chain[:1]):
            perm[a] = b

    # -- calm control mass: fully local 2-cycles inside a single aisle segment each --
    for j in range(calm2):
        aisle_seg = 1 + (j % aisle_count)
        a, b = cur, cur + 1
        cur += 2
        seg[a] = aisle_seg
        seg[b] = aisle_seg
        perm[a], perm[b] = b, a

    # -- a few genuine fixed points (already-home pallets) --
    for _ in range(extra_fixed):
        aisle_seg = 1 + (cur % aisle_count)
        seg[cur] = aisle_seg
        cur += 1

    assert cur == n
    return n, m, K, D, seg, perm


# testId -> (trap leg lengths, #calm 2-cycles, #fixed points, m segments, K registers, D)
LADDER = {
    1: ([6],                          3,  2, 3, 2, 4),
    2: ([10, 8],                      5,  3, 4, 2, 4),
    3: ([],                           30, 4, 4, 2, 5),   # calm-only: no cross edges at all
    4: ([15, 15, 10],                 6,  4, 5, 2, 5),
    5: ([25, 20, 15, 10],             6,  4, 6, 1, 6),   # single-register regime
    6: ([30, 25, 20],                 10, 5, 5, 2, 3),   # low door cost (marginal relay gain)
    7: ([40, 35, 30, 25, 20],         12, 5, 7, 2, 4),
    8: ([80, 70, 60, 50],             15, 6, 6, 1, 8),   # single-register, high door cost
    9: ([100, 90, 80, 70, 60, 50],    20, 6, 8, 2, 6),
    10: ([150, 130, 110, 90, 70, 50], 25, 8, 8, 2, 5),   # largest / stress
}


def main():
    tid = int(sys.argv[1])
    trap_legs, calm2, extra_fixed, m, K, D = LADDER[tid]
    n, m, K, D, seg, perm = build_instance(
        trap_legs, calm2, extra_fixed, m, K, D, seed=BASE_SEED + tid * 100003
    )
    out = []
    out.append("%d %d %d %d" % (n, m, K, D))
    out.append(" ".join(str(x) for x in seg))
    out.append(" ".join(str(x) for x in perm))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
