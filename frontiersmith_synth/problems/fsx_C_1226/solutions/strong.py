# TIER: strong
# Insight: paging against the allocation RATE (not the heap capacity) keeps
# every pause under budget by construction, at the cost of a few extra
# collection events. Whenever a scheduled collection would also be cheap
# enough to sweep the tenured (old) generation within budget, do a major
# collection instead of a minor one -- this reclaims dead old objects and
# keeps the ongoing footprint cost down (exploiting generational survival),
# something "collect only when the young generation is literally full" never
# even considers.
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0
    T = int(data[idx]); idx += 1
    Y = int(data[idx]); idx += 1
    K = int(data[idx]); idx += 1
    Bpause = int(data[idx]); idx += 1
    c0_minor = int(data[idx]); idx += 1
    c0_major = int(data[idx]); idx += 1
    k_young = int(data[idx]); idx += 1
    k_old = int(data[idx]); idx += 1
    idx += 1  # f_rate (not needed for the decision rule below)
    idx += 1  # penalty
    alloc = [int(x) for x in data[idx:idx + T]]; idx += T
    lifetime = [int(x) for x in data[idx:idx + T]]; idx += T

    # Largest minor-GC batch that still fits the pause budget: pace to this,
    # not to the heap capacity Y.
    batch_cap = max(1, (Bpause - c0_minor) // max(1, k_young))
    effective_cap = min(batch_cap, Y)
    cooldown = max(5, 2 * K)

    young = []  # [remaining, death_time, survive_count]
    old = []    # [remaining, death_time]
    actions = [0] * T
    since_last = 0

    for t in range(1, T + 1):
        pre_y = sum(c[0] for c in young)
        since_last += 1
        need_fire = (pre_y + alloc[t - 1] > effective_cap) or (since_last >= cooldown and pre_y > 0)

        if need_fire:
            pre_o = sum(c[0] for c in old)
            major_cost = c0_major + k_young * pre_y + k_old * pre_o
            use_major = pre_o > 0 and major_cost <= Bpause
            actions[t - 1] = 2 if use_major else 1

            nxt_y = []
            for rem, death, surv in young:
                if t >= death:
                    continue
                surv += 1
                if surv >= K:
                    old.append([rem, death])
                else:
                    nxt_y.append([rem, death, surv])
            young = nxt_y

            if use_major:
                nxt_o = []
                for rem, death in old:
                    if t >= death:
                        continue
                    nxt_o.append([rem, death])
                old = nxt_o

            since_last = 0

        if alloc[t - 1] > 0:
            young.append([alloc[t - 1], t + lifetime[t - 1], 0])

    print(" ".join(map(str, actions)))


if __name__ == "__main__":
    main()
