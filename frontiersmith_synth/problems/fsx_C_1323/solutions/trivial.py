# TIER: trivial
# Reproduce the checker's own baseline construction exactly: the known active,
# unchanged, plus round(0.25*L_ref) (>=1) extra copies of the library's single
# cheapest fragment appended at the end. This is "do nothing clever" -- pad
# with whatever is cheapest, don't bother re-examining the reference at all.
# By construction this always lands at Ratio ~= 0.1.
import sys


def main():
    tok = sys.stdin.read().split()
    p = 0
    M = int(tok[p]); p += 1
    L_max = int(tok[p]); p += 1
    STEP = int(tok[p]); p += 1
    BUDGET = int(tok[p]); p += 1

    costs = []
    for _ in range(M):
        fid = int(tok[p]); p += 1
        cost = int(tok[p]); p += 1
        p += 4  # type dx dy dz
        costs.append(cost)

    L_ref = int(tok[p]); p += 1
    ref_seq = [int(tok[p + i]) for i in range(L_ref)]; p += L_ref
    # (anchors are not needed by this tier)

    cheapest_id = min(range(M), key=lambda fid: (costs[fid], fid))
    append_n = max(1, round(0.25 * L_ref))
    seq = ref_seq + [cheapest_id] * append_n
    seq = seq[:L_max]

    print(len(seq))
    print(" ".join(str(x) for x in seq))


if __name__ == "__main__":
    main()
