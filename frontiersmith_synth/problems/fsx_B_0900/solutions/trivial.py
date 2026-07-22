# TIER: trivial
import sys

# Reproduces the checker's own baseline EXACTLY: textbook cycle decomposition (leftmost
# start, follow perm forward), star-of-transpositions from the cycle's smallest-index
# element -- but this "mover" has never heard of a SWAP instruction, so every
# transposition (anchor, b) is realized the hard way through a single shared temp bay:
#   M anchor R0      (evacuate anchor into the bay)
#   M b anchor       (b's pallet slides into anchor's now-empty slot)
#   M R0 b           (the bay's pallet slides into b's now-empty slot)
# Net effect identical to SWAP(anchor, b), but costed at edgecost(anchor,b) + 2 instead
# of a bare edgecost(anchor,b) -- exactly the checker's baseline B by construction.


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
            b = cyc[k]
            ops.append(("M", anchor, "R0"))
            ops.append(("M", b, anchor))
            ops.append(("M", "R0", b))

    out = [str(len(ops))]
    for opc, a, b in ops:
        out.append("%s %s %s" % (opc, a, b))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
