# TIER: strong
import sys

# The insight: minimal MOVE-COUNT and minimal MOVE-COST diverge once doorways aren't
# free. Textbook cycle-following (the `greedy` tier) always fans a star of transpositions
# out of one fixed anchor, so it pays the door toll once per star edge that happens to
# land off-segment from that anchor -- even when the cycle's own RING topology (the
# order pallets actually chain into each other, perm[i] -> perm[perm[i]] -> ...) only
# crosses a doorway a handful of times. This solution decomposes each cycle into two
# cost-aware primitive families instead of one uniform one:
#
#   (a) plain in-place chaining: walk the cycle's natural ring and fill each hole
#       directly from its predecessor -- free (cost 1) whenever predecessor and hole
#       share a segment;
#   (b) a scratch-register relay: whenever a ring step WOULD cross a doorway, detour the
#       pallet through a register instead (2 register touches, cost 1 each) rather than
#       paying the doorway's cost D directly -- strictly better whenever D > 2.
#
# One register (R0) is unavoidably needed just to break the ring into a chain in the
# first place (there is no natural empty slot to start from); if a SECOND register is
# available (K >= 2) it is reused, on demand, to relay every OTHER doorway-crossing ring
# step it encounters, not just the one used to open the chain. For cycles where this ring
# decomposition doesn't actually help (e.g. it never beats plain cycle-following, as on
# short or fully local cycles) the solution falls back to the star construction -- it
# always emits whichever of the two candidate plans is cheaper, per cycle, using the
# instance's own K and D.


def decompose_cycles(perm, n):
    seen = [False] * n
    cycles = []
    for i in range(n):
        if seen[i]:
            continue
        if perm[i] == i:
            seen[i] = True
            continue
        cyc = []
        j = i
        while not seen[j]:
            seen[j] = True
            cyc.append(j)
            j = perm[j]
        cycles.append(cyc)
    return cycles


def edgecost(a, b, seg, D):
    return 1 if seg[a] == seg[b] else D


def swap_chain_plan(cyc, seg, D):
    L = len(cyc)
    anchor = cyc[0]
    ops = []
    cost = 0
    for k in range(1, L):
        b = cyc[k]
        ops.append(("S", anchor, b))
        cost += edgecost(anchor, b, seg, D)
    return ops, cost


def hole_chain_plan(cyc, seg, D, K):
    """Break the ring at a doorway-crossing edge if one exists (so the register that
    MUST be spent opening the chain buys back a door toll instead of a free local step),
    then walk the ring filling each hole from its predecessor -- relaying through a
    second register whenever that particular step would cross a doorway and one is
    available."""
    L = len(cyc)
    if K < 1:
        return None  # no register at all: this plan is unavailable
    s = 0
    for t in range(L):
        a, b = cyc[t], cyc[(t + 1) % L]
        if seg[a] != seg[b]:
            s = t
            break
    ops = []
    cost = 0
    start = cyc[s]
    ops.append(("M", start, "R0"))
    cost += 1
    hole = start
    adhoc = "R1" if K >= 2 else None
    for step in range(1, L):
        pred = cyc[(s - step) % L]
        if seg[pred] == seg[hole]:
            ops.append(("M", pred, hole))
            cost += 1
        elif adhoc is not None:
            ops.append(("M", pred, adhoc))
            ops.append(("M", adhoc, hole))
            cost += 2
        else:
            ops.append(("M", pred, hole))
            cost += D
        hole = pred
    ops.append(("M", "R0", hole))
    cost += 1
    return ops, cost


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); m = int(next(it)); K = int(next(it)); D = int(next(it))
    seg = [int(next(it)) for _ in range(n)]
    perm = [int(next(it)) for _ in range(n)]

    ops = []
    for cyc in decompose_cycles(perm, n):
        L = len(cyc)
        if L < 2:
            continue
        sw_ops, sw_cost = swap_chain_plan(cyc, seg, D)
        best_ops, best_cost = sw_ops, sw_cost
        hc = hole_chain_plan(cyc, seg, D, K)
        if hc is not None:
            hc_ops, hc_cost = hc
            if hc_cost < best_cost:
                best_ops, best_cost = hc_ops, hc_cost
        ops.extend(best_ops)

    out = [str(len(ops))]
    for opc, a, b in ops:
        out.append("%s %s %s" % (opc, a, b))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
