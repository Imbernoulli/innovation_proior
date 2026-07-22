# TIER: strong
# Two insights compose here:
#
# 1. Cross-community routing: for each good, build each community's TRUE
#    concave "value of receiving j units optimally" curve -- cap[g] copies of
#    each member's weight, flattened over the whole community and sorted
#    descending -- and water-fill the shipment across communities by always
#    giving the next unit to whichever community's next-in-line marginal is
#    currently highest. A community that hosts a couple of hugely-weighted
#    "champion" households for a foreign good absorbs exactly their bounded
#    capacity and no more; the rest correctly flows to the good's owning
#    community. This is the basin-preimage decision -- the one thing the
#    friction-limited, community-local replay can never undo once misrouted,
#    because it can never move a unit across a community boundary.
#
# 2. Within a community's own share: because utility is FLAT per unit up to
#    the satiation cap, the raw total utility captured is identical whether
#    the community's allotment is concentrated on its top few members (as an
#    unaware per-unit-argmax rule would do) or spread round-robin across all
#    of its members. But the objective is a GEOMETRIC mean: leaving any
#    member with literally zero collapses their log-utility to the
#    subsistence floor, and a household holding nothing has nothing to offer
#    in a barter, so the replay can never fix it either. Spreading the
#    allotment round-robin costs nothing in raw utility and gives every
#    member something to trade with -- letting the friction-limited replay
#    polish the fine-grained, preference-matched split for free.
import sys


def main():
    data = sys.stdin.read().split()
    idx = [0]

    def nxt():
        v = data[idx[0]]
        idx[0] += 1
        return v

    N = int(nxt()); G = int(nxt()); K = int(nxt()); R = int(nxt()); eps = int(nxt())
    cap = [int(nxt()) for _ in range(G)]
    S = [int(nxt()) for _ in range(G)]
    comm = [0] * N
    W = [[0] * G for _ in range(N)]
    for i in range(N):
        comm[i] = int(nxt())
        for g in range(G):
            W[i][g] = int(nxt())

    groups = {}
    for i in range(N):
        groups.setdefault(comm[i], []).append(i)
    community_ids = sorted(groups)

    x = [[0] * G for _ in range(N)]
    for g in range(G):
        cap_g = cap[g]
        curve, used = {}, {}
        for c in community_ids:
            vals = []
            for i in groups[c]:
                vals.extend([W[i][g]] * cap_g)
            vals.sort(reverse=True)
            curve[c] = vals
            used[c] = 0

        y = {c: 0 for c in community_ids}
        remaining = S[g]
        while remaining > 0:
            best_c, best_v = None, -1
            for c in community_ids:
                p = used[c]
                v = curve[c][p] if p < len(curve[c]) else -1
                if v > best_v:
                    best_v, best_c = v, c
            if best_c is None or best_v < 0:
                y[community_ids[0]] += remaining
                remaining = 0
                break
            used[best_c] += 1
            y[best_c] += 1
            remaining -= 1

        for c in community_ids:
            members = groups[c]
            m = len(members)
            take = y[c]
            start = g % m
            for k in range(take):
                x[members[(start + k) % m]][g] += 1

    print("\n".join(" ".join(map(str, row)) for row in x))


if __name__ == "__main__":
    main()
