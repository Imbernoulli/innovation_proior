# TIER: strong
# The insight: don't ask "how do I build THIS row" (a per-output pass) -- ask
# "which partial sum, if I commit to materializing it as a wire right now,
# pays for itself across the most targets", and keep re-asking that question
# after every commitment. Concretely: maintain a pool of already-built wires
# (starting from the raw variables). Each round, greedily set-cover EVERY
# target's variable set using the CURRENT pool (largest pieces first), tally
# how often each unordered PAIR of pieces co-occurs together inside some
# target's cover, and commit (build a new wire for) the most frequent
# co-occurring pair. Because the pool is re-mined after every commitment, a
# first-round pair-wire that is itself reused with a third piece by many
# targets gets promoted into a single larger reusable wire in a later round --
# recovering a whole shared atom, not just one pairwise fragment of it. This
# global, re-evaluated commitment is exactly what a one-shot per-target or
# one-shot raw-pair pass cannot see.
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
        rows.append(frozenset(idxs))

    gates = []

    def new_gate(a, b):
        gates.append((a, b))
        return C + len(gates)

    pool = {}  # frozenset(varset) -> wire id
    for v in range(1, C + 1):
        pool[frozenset((v,))] = v

    def sorted_pool_items():
        return sorted(pool.items(), key=lambda kv: (-len(kv[0]), tuple(sorted(kv[0]))))

    def cover_of(rowset, items):
        remaining = set(rowset)
        cov = []
        for piece, wid in items:
            if not remaining:
                break
            if piece and piece <= remaining:
                cov.append((piece, wid))
                remaining -= piece
        return cov

    max_iters = 4 * C + 20
    for _ in range(max_iters):
        items = sorted_pool_items()
        pair_freq = defaultdict(int)
        pair_pieces = {}
        for rowset in rows:
            cov = cover_of(rowset, items)
            n = len(cov)
            for i in range(n):
                for j in range(i + 1, n):
                    p1, _ = cov[i]; p2, _ = cov[j]
                    key = tuple(sorted((tuple(sorted(p1)), tuple(sorted(p2)))))
                    pair_freq[key] += 1
                    if key not in pair_pieces:
                        pair_pieces[key] = (p1, p2)

        candidates = [k for k, f in pair_freq.items() if f >= 2]
        if not candidates:
            break
        best_key = min(candidates, key=lambda k: (-pair_freq[k], k))
        p1, p2 = pair_pieces[best_key]
        new_piece = p1 | p2
        if new_piece in pool:
            # already materialized via another route -- nothing to do; the
            # cover recomputation next round will naturally prefer it.
            continue
        w = new_gate(pool[p1], pool[p2])
        pool[new_piece] = w

    # ---- final assembly: combine each row's remaining cover pieces ----
    items = sorted_pool_items()
    out_wire = [0] * R
    for r, rowset in enumerate(rows):
        cov = cover_of(rowset, items)
        w = cov[0][1]
        for piece, wid in cov[1:]:
            w = new_gate(w, wid)
        out_wire[r] = w

    lines = [str(len(gates))]
    for a, b in gates:
        lines.append("%d %d" % (a, b))
    for w in out_wire:
        lines.append(str(w))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
