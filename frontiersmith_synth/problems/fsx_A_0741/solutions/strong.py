# TIER: strong
"""The insight: a compaction is a perishable investment whose payoff is the
integral of lookups it will shield until the timeline ends (or until the next
decision point makes it worth re-pricing). Since the WHOLE timeline -- every
box arrival and every reading-room request -- is visible upfront, we don't
have to react to box counts; we can price each candidate 'compact everything
alive right now' action against exactly the read traffic it would spare.

At each point a lookup burst is about to start, price the action:
    cost(merge now)   = bytes rewritten now + (1 probe per future lookup
                         that still falls in the merged range)
    cost(do nothing)  = sum over ALL remaining lookups of how many
                         currently-alive boxes their key would hit
Commit only if merging now is cheaper over the REST of the timeline. This
naturally clusters compactions right before demand and lets boxes pile up
for free during write-only stretches, which no fixed cadence or box-count
threshold can express."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)

    def nx():
        return next(it)

    N = int(nx())
    M = int(nx())
    sizes = [0] * (N + 1)
    lo = [0] * (N + 1)
    hi = [0] * (N + 1)
    for i in range(1, N + 1):
        sizes[i] = int(nx())
        lo[i] = int(nx())
        hi[i] = int(nx())
    T = int(nx())
    events = []
    for _ in range(T):
        e = nx()
        if e == "I":
            events.append(("I", None))
        else:
            events.append(("L", int(nx())))

    # cluster consecutive lookups (small timeline-gap) into "hot windows";
    # every remaining lookup after a window is still counted in the full
    # lookahead sum below, windows only mark CANDIDATE decision points.
    GAP = 4
    l_positions = [(t + 1, events[t][1]) for t in range(T) if events[t][0] == "L"]
    clusters = []
    cur = []
    prev_pos = None
    for (pos, q) in l_positions:
        if prev_pos is not None and pos - prev_pos > GAP:
            clusters.append(cur)
            cur = []
        cur.append((pos, q))
        prev_pos = pos
    if cur:
        clusters.append(cur)

    boundary = 0          # last id folded into the current merged prefix block (0 = none)
    arrived = 0
    merged_lo = None
    merged_hi = None
    merged_size = 0
    merges = []

    def alive_ranges():
        rs = []
        if boundary > 0:
            rs.append((merged_lo, merged_hi))
        for i in range(boundary + 1, arrived + 1):
            rs.append((lo[i], hi[i]))
        return rs

    cluster_idx = 0
    for t in range(1, T + 1):
        if cluster_idx < len(clusters) and clusters[cluster_idx][0][0] == t:
            ranges = alive_ranges()
            if len(ranges) > 1:
                # price against EVERY remaining lookup in the timeline (the
                # full payoff horizon), not just the upcoming cluster.
                future_qs = [q for (pos2, q) in l_positions if pos2 >= t]
                cost_no_merge = 0
                for q in future_qs:
                    cost_no_merge += sum(1 for (blo, bhi) in ranges if blo <= q <= bhi)
                glo = min(r[0] for r in ranges)
                ghi = max(r[1] for r in ranges)
                cost_with_merge_reads = sum(1 for q in future_qs if glo <= q <= ghi)
                merge_cost = (merged_size if boundary > 0 else 0) + \
                    sum(sizes[i] for i in range(boundary + 1, arrived + 1))
                if merge_cost + cost_with_merge_reads < cost_no_merge:
                    merges.append((t - 1, 1, arrived))
                    merged_lo, merged_hi = glo, ghi
                    merged_size = merge_cost
                    boundary = arrived
            cluster_idx += 1

        kind, _ = events[t - 1]
        if kind == "I":
            arrived += 1

    out = [str(len(merges))]
    for (g, f, l) in merges:
        out.append(f"{g} {f} {l}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
