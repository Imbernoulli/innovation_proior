# TIER: greedy
# The obvious "similarity-maximizing" strategy: to be SAFE about preserving
# activity, never touch any position that carries a pharmacophoric feature
# (D/A/R/H) in the known active -- keep the exact same fragment id there.
# Off-feature ("filler", type X) positions are the only place this solver
# dares to optimize, and it does so myopically: swap each filler slot to
# whichever single fragment is cheapest in the WHOLE library. It never
# considers extending/trimming the chain, and never realizes that the
# feature positions could ALSO be re-sourced from a different (cheaper,
# off-reference) fragment carrying the same type at the same offset --
# so it keeps paying the reference's own feature-fragment identity there,
# capping novelty at 1-K/L and leaving most of the score on the table.
import sys


def main():
    tok = sys.stdin.read().split()
    p = 0
    M = int(tok[p]); p += 1
    L_max = int(tok[p]); p += 1
    STEP = int(tok[p]); p += 1
    BUDGET = int(tok[p]); p += 1

    costs = []
    types = []
    for _ in range(M):
        fid = int(tok[p]); p += 1
        cost = int(tok[p]); p += 1
        typ = tok[p]; p += 1
        p += 3  # dx dy dz
        costs.append(cost)
        types.append(typ)

    L_ref = int(tok[p]); p += 1
    ref_seq = [int(tok[p + i]) for i in range(L_ref)]; p += L_ref
    # (anchors not needed: feature-ness is read straight off the library type)

    cheapest_id = min(range(M), key=lambda fid: (costs[fid], fid))

    seq = []
    for fid in ref_seq:
        if types[fid] in ('D', 'A', 'R', 'H'):
            seq.append(fid)          # "don't touch what preserves activity"
        else:
            seq.append(cheapest_id)  # naive cost-only swap on filler slots

    print(len(seq))
    print(" ".join(str(x) for x in seq))


if __name__ == "__main__":
    main()
