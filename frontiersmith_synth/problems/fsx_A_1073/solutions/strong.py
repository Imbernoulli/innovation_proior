# TIER: strong
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    L = int(next(it)); M = int(next(it)); K = int(next(it))
    P = int(next(it)); W = int(next(it)); D = int(next(it))
    ref = next(it)
    positions = [int(next(it)) for _ in range(M)]
    snp_count = int(next(it))
    snp_offsets = [int(next(it)) for _ in range(snp_count)]
    tlen = int(next(it)); tstart = int(next(it))
    n_mut = int(next(it))

    # Which loci does the published mutant panel actually stress?  (Loci that never
    # appear in any mutant's subset never need more than the trivial forward probe.)
    stressed = set()
    for _ in range(n_mut):
        cls = int(next(it)); sz = int(next(it))
        idxs = [int(next(it)) for _ in range(sz)]
        stressed.update(idxs)

    probes = []
    for i in range(M):
        p = positions[i]
        if i in stressed:
            # Insight 1: the binding constraint is the WORST edit class at this locus, not
            # how conserved the canonical window looks. Exhaustively search every shift for
            # the forward probe so it dodges every published SNP-cluster offset (this is
            # what handles the SNP-cluster class exactly, however the offsets are laid out).
            best_shift, best_overlap = 0, None
            for shift in range(0, W - P + 1):
                lo, hi = shift, shift + P
                ov = sum(1 for o in snp_offsets if lo <= o < hi)
                if best_overlap is None or ov < best_overlap:
                    best_overlap, best_shift = ov, shift
            fpos = p + best_shift
            probes.append((fpos, "F"))
            # Insight 2: a reverse-complement probe of ANY sub-window of a locus exactly
            # matches that locus's window after segment-inversion (revcomp of a substring
            # of W equals the corresponding substring of revcomp(W)), regardless of shift.
            probes.append((fpos, "R"))
        else:
            probes.append((p, "F"))

    probes = probes[:K]
    lines = [str(len(probes))]
    for pos, orient in probes:
        lines.append(f"{pos} {orient}")
    sys.stdout.write("\n".join(lines) + "\n")


main()
