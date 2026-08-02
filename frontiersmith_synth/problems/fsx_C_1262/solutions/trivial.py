# TIER: trivial
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

    # Naive: spread neurons round-robin across ALL available cores, in ID
    # order, no notion of connectivity or minimizing core count at all.
    core_list = [[] for _ in range(C_max)]
    fanout_sum = [0] * C_max
    core_of = [-1] * N
    for i in range(N):
        c = i % C_max
        tries = 0
        while tries < C_max and not (len(core_list[c]) < T * slot_cap and fanout_sum[c] + outdeg[i] <= fanout_budget):
            c = (c + 1) % C_max
            tries += 1
        core_list[c].append(i)
        fanout_sum[c] += outdeg[i]
        core_of[i] = c

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
