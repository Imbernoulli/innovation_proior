# TIER: greedy
"""The obvious first recipe: literally 'staff to the average, by shift'.

For each shift block, average low/high demand ACROSS the held-out days AND
across the block's own hours into one flat per-shift staffing level, sized
independently per skill (junior off mean low demand only, senior off mean
high demand only -- no substitution reasoning). This is exactly the
standard roster: it nails flat/typical days, where the average IS the
right level. It structurally cannot see that (a) a multi-hour surge that
hits only SOME days gets doubly diluted -- smoothed across the day-mean
AND across the block's other, quieter hours -- and (b) idle senior
capacity could have covered low-acuity overflow for free. So on days that
actually carry the recurring surge it under-provisions senior badly and
pays for it in expensive agency cover, and it never exploits the cheaper,
flexible substitution the senior tier already offers.
"""
import sys, math


def main():
    data = sys.stdin.read().split()
    p = 0
    def nxt():
        nonlocal p
        v = data[p]; p += 1
        return v

    T = int(nxt())
    n_starts = int(nxt())
    starts = [int(nxt()) for _ in range(n_starts)]
    max_per_slot = int(nxt())
    _cost_base_J = int(nxt()); _cost_base_S = int(nxt())
    _ot = [int(nxt()) for _ in range(4)]
    _cost_ot = [int(nxt()) for _ in range(2)]
    _cost_agency = [int(nxt()) for _ in range(2)]
    K = int(nxt())
    days = []
    for _d in range(K):
        L = [0] * T
        H = [0] * T
        for t in range(T):
            L[t] = int(nxt()); H[t] = int(nxt())
        days.append((L, H))

    meanL = [sum(days[d][0][t] for d in range(K)) / K for t in range(T)]
    meanH = [sum(days[d][1][t] for d in range(K)) / K for t in range(T)]

    block_len = T // n_starts
    roster_J = []
    roster_S = []
    for b in range(n_starts):
        hrs = list(range(b * block_len, (b + 1) * block_len))
        avgL = sum(meanL[t] for t in hrs) / block_len
        avgH = sum(meanH[t] for t in hrs) / block_len
        needJ = math.ceil(avgL - 1e-9)
        needS = math.ceil(avgH - 1e-9)
        roster_J.append(min(max_per_slot, max(0, needJ)))
        roster_S.append(min(max_per_slot, max(0, needS)))

    for b in range(n_starts):
        print(roster_J[b], roster_S[b])


if __name__ == "__main__":
    main()
