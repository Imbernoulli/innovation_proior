# TIER: strong
"""Insight: a pointing's value is its MARGINAL pair-separation power against the
CURRENTLY-confusable narrative pairs of the CURRENT worst family -- not its raw
quality weight, and not its total coverage summed across all families. At each of
the B booking steps we (a) find the family with the lowest current separation
fraction, (b) score every still-available pointing by how many of THAT family's
still-tied pairs it would newly split, and (c) book the best one. This directly
optimizes the max-min objective and is exactly what makes 'boring', low-weight
pointings worth booking: they can be the only lever on the worst family, while a
quality- or total-coverage-greedy scan never looks for them."""
import sys


def reading(secs, sector, val):
    return val if sector in secs else 0


def main():
    data = sys.stdin.read().split()
    it = iter(data)

    def nxt():
        return next(it)

    N = int(nxt()); D = int(nxt()); K = int(nxt()); B = int(nxt()); T = int(nxt())
    slots = {}  # sid -> (night, weight, secs)
    for _ in range(T):
        sid = int(nxt()); night = int(nxt()); w = int(nxt()); ns = int(nxt())
        secs = set(int(nxt()) for _ in range(ns))
        slots[sid] = (night, w, secs)
    families = []
    for _ in range(K):
        Mf = int(nxt())
        narrs = []
        for _ in range(Mf):
            sec = int(nxt()); val = int(nxt())
            narrs.append((sec, val))
        families.append(narrs)

    # tied[f] = set of (i,j) index pairs in family f not yet distinguished by anything booked
    tied = []
    for narrs in families:
        m = len(narrs)
        tied.append(set((i, j) for i in range(m) for j in range(i + 1, m)))

    available = dict(slots)  # sid -> (night, weight, secs)
    used_nights = set()
    chosen = []

    for _ in range(B):
        if not available:
            break
        # current separation fraction per family (for picking the worst)
        fracs = []
        for f, narrs in enumerate(families):
            m = len(narrs)
            total = m * (m - 1) // 2
            fracs.append(1.0 if total == 0 else 1.0 - len(tied[f]) / total)
        worst_f = min(range(K), key=lambda f: (fracs[f], f))

        best_sid, best_gain, best_w = None, -1, -1
        narrs = families[worst_f]
        for sid, (night, w, secs) in available.items():
            if night in used_nights:
                continue
            gain = 0
            for (i, j) in tied[worst_f]:
                si, vi = narrs[i]
                sj, vj = narrs[j]
                if reading(secs, si, vi) != reading(secs, sj, vj):
                    gain += 1
            if gain > best_gain or (gain == best_gain and w > best_w):
                best_sid, best_gain, best_w = sid, gain, w

        if best_sid is None:
            break

        if best_gain <= 0:
            # nothing helps the worst family right now: fall back to whichever
            # available pointing helps the MOST across all families combined,
            # so the booking is not wasted.
            best_sid2, best_total = None, -1
            for sid, (night, w, secs) in available.items():
                if night in used_nights:
                    continue
                total_gain = 0
                for f, narrs2 in enumerate(families):
                    for (i, j) in tied[f]:
                        si, vi = narrs2[i]
                        sj, vj = narrs2[j]
                        if reading(secs, si, vi) != reading(secs, sj, vj):
                            total_gain += 1
                if total_gain > best_total:
                    best_sid2, best_total = sid, total_gain
            if best_sid2 is not None:
                best_sid = best_sid2

        night, w, secs = available[best_sid]
        used_nights.add(night)
        chosen.append(best_sid)
        del available[best_sid]

        # update ALL families' tied sets using the newly booked pointing
        for f, narrs2 in enumerate(families):
            still = set()
            for (i, j) in tied[f]:
                si, vi = narrs2[i]
                sj, vj = narrs2[j]
                if reading(secs, si, vi) == reading(secs, sj, vj):
                    still.add((i, j))
            tied[f] = still

    print(len(chosen))
    print(" ".join(map(str, chosen)))


if __name__ == "__main__":
    main()
