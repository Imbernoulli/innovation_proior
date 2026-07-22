# TIER: strong
# Simulate ONE truck's descent once to read off exactly what the fleet-wide cadence must
# be: the slowest block sets a floor of t ticks between consecutive same-direction trucks,
# and any mandatory heat cooldown -- computed exactly, not rounded to any grid -- raises
# that floor further (fold the cooldown straight into the spacing, "spend heat's forced
# wait as part of the beat"). Launch the whole fleet down at that exact spacing, let the
# corridor clear, send it back at the (looser, cooldown-free) ascent spacing, and repeat.
# Every truck in a wave follows the identical timeline, offset by exactly the cadence, so
# nobody ever discovers mid-wave that they arrived too early and has to queue for a bay
# someone else is still cooling in -- the fleet-wide plan already accounted for it before
# a single truck moved. A reactive dispatcher has no such foresight: it packs trucks in
# at whatever the road looks like clear "right now", finds out only once it reaches the
# hot block that everyone needs the same cooldown stop, and re-discovers the true cadence
# by queuing for it -- one truck at a time, every single wave, not just once.
import sys


def main():
    data = sys.stdin.read().split()
    p = iter(data)
    M = int(next(p)); K = int(next(p)); T = int(next(p))
    H_MAX = int(next(p)); idle_cool = int(next(p)); heat_loss = int(next(p))
    t_list = [int(next(p)) for _ in range(M)]
    g_list = [int(next(p)) for _ in range(M)]
    t = max(t_list)  # cadence floor = the SLOWEST block, not block 1's own time

    cps0 = [(0, 0)]
    heat = 0
    t_cur = 0
    max_wait = 0
    for i in range(M):
        g = g_list[i]
        if heat + g > H_MAX:
            need = heat + g - H_MAX
            w = -(-need // idle_cool)
            heat = max(0, heat - w * idle_cool)
            t_cur += w
            cps0.append((t_cur, i))
            max_wait = max(max_wait, w)
        t_cur += t_list[i]
        heat += g
        cps0.append((t_cur, i + 1))
    down_tpl = cps0
    Ddown = down_tpl[-1][0]
    Dup = sum(t_list)

    # pullouts use CLOSED occupancy intervals, so a nonzero dwell needs a strictly larger
    # same-direction gap than the dwell itself (a 0-dwell pass-through only needs the
    # slowest block's own t ticks of separation).
    r_down = max(t, (max_wait + 1) if max_wait > 0 else 0, 1)
    r_up = max(t, 1)

    up_tpl = [(0, M)]
    tc = 0
    for i in reversed(range(M)):
        tc += t_list[i]
        up_tpl.append((tc, i))

    def extend(cps_k, start, template):
        last = cps_k[-1]
        base_node = template[0][1]
        if start > last[0]:
            cps_k.append((start, base_node))
            last = cps_k[-1]
        for (dt, nd) in template[1:]:
            tk = start + dt
            if tk > T:
                return
            if (tk, nd) != last:
                cps_k.append((tk, nd))
                last = cps_k[-1]

    cps = [[(0, 0)] for _ in range(K)]
    T0 = 0
    while T0 <= T:
        reached = [False] * K
        for j in range(K):
            base = T0 + j * r_down
            if base > T:
                continue
            extend(cps[j], base, down_tpl)
            if cps[j][-1] == (base + Ddown, M):
                reached[j] = True
        T1 = T0 + (K - 1) * r_down + Ddown
        for j in range(K):
            if not reached[j]:
                continue
            base2 = T1 + j * r_up
            if base2 > T:
                continue
            extend(cps[j], base2, up_tpl)
        T2 = T1 + (K - 1) * r_up + Dup
        if T2 <= T0:
            break
        T0 = T2

    # Never let a truck's plan end mid-corridor: near the horizon cutoff, two trucks
    # offset by the same cadence can both get truncated while dwelling at the SAME
    # pullout, which the checker (correctly) treats as occupying it forever -- a genuine
    # collision. Back off to the last depot visit (unconstrained capacity) instead.
    for k in range(K):
        while len(cps[k]) > 1 and 0 < cps[k][-1][1] < M:
            cps[k].pop()

    out = [str(K)]
    for k in range(K):
        line = [str(len(cps[k]))]
        for (tk, nd) in cps[k]:
            line.append(str(tk)); line.append(str(nd))
        out.append(" ".join(line))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
