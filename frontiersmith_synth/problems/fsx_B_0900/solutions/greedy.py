# TIER: greedy
import sys

# The obvious first-instinct recipe: textbook cycle-following. Decompose perm into
# cycles, then for each cycle realize it with the standard L-1 swaps SWAP(anchor, cyc[k])
# fanned out from the cycle's smallest-index element (anchor = cyc[0]) -- the minimal
# NUMBER of operations to sort the cycle, completely blind to which of those swaps happen
# to cross a doorway. This is exactly what minimizes move-COUNT, not move-COST: on a
# relay cycle whose anchor sits alone in the dock segment, every single one of the L-1
# star swaps touches the dock, so the WHOLE fan-out pays the door toll -- even though the
# cycle's own ring only truly needs to cross the door twice. Never uses a register.


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


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); m = int(next(it)); K = int(next(it)); D = int(next(it))
    seg = [int(next(it)) for _ in range(n)]
    perm = [int(next(it)) for _ in range(n)]

    ops = []
    for cyc in decompose_cycles(perm, n):
        L = len(cyc)
        anchor = cyc[0]
        for k in range(1, L):
            ops.append(("S", anchor, cyc[k]))

    out = [str(len(ops))]
    for opc, a, b in ops:
        out.append("%s %s %s" % (opc, a, b))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
