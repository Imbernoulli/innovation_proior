# TIER: greedy
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    L = int(next(it)); M = int(next(it)); K = int(next(it))
    P = int(next(it)); W = int(next(it)); D = int(next(it))
    ref = next(it)
    positions = [int(next(it)) for _ in range(M)]

    # Obvious "textbook" approach: one conserved-window forward probe per locus, plus a
    # second forward probe hedging the other end of the window. No orientation duality,
    # no reading of the published SNP-cluster offsets or mutant panel.
    probes = []
    for p in positions:
        probes.append((p, "F"))
        p2 = p + (W - P)
        if p2 != p:
            probes.append((p2, "F"))
    probes = probes[:K]

    lines = [str(len(probes))]
    for pos, orient in probes:
        lines.append(f"{pos} {orient}")
    sys.stdout.write("\n".join(lines) + "\n")


main()
