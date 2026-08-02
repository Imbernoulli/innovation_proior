# TIER: greedy
import sys


def lpt_slots(core_neurons, inrate, slot_cap, T):
    order = sorted(core_neurons, key=lambda x: (-inrate[x], x))
    slot_load = [0] * T
    slot_count = [0] * T
    assign = {}
    for nid in order:
        best_s = None
        for s in range(T):
            if slot_count[s] < slot_cap:
                if best_s is None or slot_load[s] < slot_load[best_s]:
                    best_s = s
        assign[nid] = best_s
        slot_load[best_s] += inrate[nid]
        slot_count[best_s] += 1
    return assign


def main():
    data = sys.stdin.read().split()
    it = iter(data)

    def nxt():
        return int(next(it))

    N = nxt(); C_max = nxt(); T = nxt(); slot_cap = nxt()
    fanout_budget = nxt(); slot_rate_budget = nxt()
    INTER = nxt(); LOCAL = nxt(); OVER = nxt()
    M = nxt()
    outdeg = [0] * N
    inrate = [0] * N
    for _ in range(M):
        u = nxt(); v = nxt(); r = nxt()
        outdeg[u] += 1
        inrate[v] += r

    # The "obvious" recipe: this is a bin-packing problem, so minimize the
    # number of cores used. Sort neurons by descending fanout (classic
    # first-fit-decreasing bin-packing heuristic) and pack them as tightly
    # as the capacity/fanout/rate constraints allow, using the LOWEST-index
    # cores first. This never looks at which neurons talk to which -- it
    # treats every neuron as an interchangeable "item" of a given size.
    order = sorted(range(N), key=lambda x: (-outdeg[x], x))
    core_list = [[] for _ in range(C_max)]
    fanout_sum = [0] * C_max
    core_of = [-1] * N
    for i in order:
        for c in range(C_max):
            if len(core_list[c]) < T * slot_cap and fanout_sum[c] + outdeg[i] <= fanout_budget:
                core_list[c].append(i)
                fanout_sum[c] += outdeg[i]
                core_of[i] = c
                break

    slot_of = [0] * N
    for members in core_list:
        if not members:
            continue
        a = lpt_slots(members, inrate, slot_cap, T)
        for i, s in a.items():
            slot_of[i] = s

    out = [str(N)]
    for i in range(N):
        out.append("%d %d" % (core_of[i], slot_of[i]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
