# TIER: greedy
# The "obvious" common-subexpression pass a strong coder writes first: count how
# often each RAW-VARIABLE pair co-occurs across all targets, ONE SHOT (no
# recomputation), and greedily commit to build the most frequent still-free
# pairs as shared wires. Every target then reuses whichever committed pairs it
# contains and finishes any leftover variables on its own.
#
# This is a real, valid, single-level common-subexpression optimization -- but
# it can only ever discover PAIRS of ORIGINAL variables, once, up front. It
# never re-mines candidates after a commitment (so it can't notice that a
# freshly built pair-wire, extended by one more variable, is ALSO shared by
# the same group of targets), and it never considers pairing two already-built
# wires together. So it only ever recovers PART of any larger shared atom.
import sys
from collections import defaultdict


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    R = int(next(it)); C = int(next(it))
    rows = []
    for _ in range(R):
        k = int(next(it))
        idxs = [int(next(it)) for _ in range(k)]
        rows.append(idxs)

    # ---- one-shot frequency count of raw-variable pairs ----
    freq = defaultdict(int)
    for idxs in rows:
        n = len(idxs)
        for i in range(n):
            for j in range(i + 1, n):
                freq[(idxs[i], idxs[j])] += 1

    pairs_sorted = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))

    gates = []

    def new_gate(a, b):
        gates.append((a, b))
        return C + len(gates)

    used = set()
    pair_wire = []  # list of (a, b, wire_id), in commit order
    for (a, b), f in pairs_sorted:
        if f < 2:
            break
        if a in used or b in used:
            continue
        w = new_gate(a, b)
        pair_wire.append((a, b, w))
        used.add(a); used.add(b)

    out_wire = [0] * R
    for r, idxs in enumerate(rows):
        remaining = set(idxs)
        terms = []
        for a, b, w in pair_wire:
            if a in remaining and b in remaining:
                terms.append(w)
                remaining.discard(a); remaining.discard(b)
        terms.extend(sorted(remaining))
        w = terms[0]
        for t in terms[1:]:
            w = new_gate(w, t)
        out_wire[r] = w

    lines = [str(len(gates))]
    for a, b in gates:
        lines.append("%d %d" % (a, b))
    for w in out_wire:
        lines.append(str(w))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
