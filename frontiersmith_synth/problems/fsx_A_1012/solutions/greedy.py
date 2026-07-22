# TIER: greedy
# Reactive dispatcher: every truck independently advances the instant its next segment,
# target pullout, and heat allow it to -- no fleet-wide plan, no global cadence, just
# "go the moment it's legal, else wait one tick and recheck". This is the obvious first
# algorithm for a congestion-avoidance problem: it never explicitly violates a rule (block
# and pullout capacity, and the heat cap, are checked every tick), so it always produces
# a technically legal plan. But because it has no notion of a fleet-wide rhythm, nothing
# stops it from launching descending and returning trucks into the SAME shared corridor
# at the same time; on the denser, tighter-heat instances this reactive interleaving runs
# two opposing convoys straight at each other and they lock up nose-to-tail with nowhere
# to go -- exactly the failure a pre-computed, phase-locked cadence is built to avoid.
import sys


def main():
    data = sys.stdin.read().split()
    p = iter(data)
    M = int(next(p)); K = int(next(p)); T = int(next(p))
    H_MAX = int(next(p)); idle_cool = int(next(p)); heat_loss = int(next(p))
    t_list = [int(next(p)) for _ in range(M)]
    g_list = [int(next(p)) for _ in range(M)]

    cps = [[(0, 0)] for _ in range(K)]
    node = [0] * K
    going_down = [True] * K
    transit_end = [None] * K
    transit_target = [None] * K
    new_heat_pc = [0] * K
    last_cp_tick = [0] * K

    # heat is tracked ANALYTICALLY while dwelling: heat_at_node[k] is the truck's heat at
    # the instant it arrived at its current node (node_arrival_tick[k]); its heat at any
    # later tick t while still dwelling there is heat_at_node[k] cooled by idle_cool for
    # (t - node_arrival_tick[k]) ticks. This is what a real reactive dispatcher would also
    # need to get right -- it is NOT the innovation, just correct bookkeeping.
    heat_at_node = [0] * K
    node_arrival_tick = [0] * K

    block_busy_until = [0] * (M + 1)     # index 1..M
    node_claim = [-1] * (M + 1)          # index 1..M-1 ; -1 = free

    for t in range(0, T + 1):
        for k in range(K):
            if transit_end[k] is not None:
                if t < transit_end[k]:
                    continue
                if t == transit_end[k]:
                    target = transit_target[k]
                    node[k] = target
                    heat_at_node[k] = new_heat_pc[k]
                    node_arrival_tick[k] = t
                    cps[k].append((t, target))
                    last_cp_tick[k] = t
                    transit_end[k] = None
                    if target == M:
                        going_down[k] = False
                    elif target == 0:
                        going_down[k] = True

            nd = node[k]
            if nd == 0:
                target = 1; desc = True
            elif nd == M:
                target = M - 1; desc = False
            else:
                desc = going_down[k]
                target = nd + 1 if desc else nd - 1

            block_idx = target if desc else nd
            ttime = t_list[block_idx - 1]

            if block_busy_until[block_idx] > t:
                continue
            if 1 <= target <= M - 1 and node_claim[target] not in (-1, k):
                continue

            cur_heat = max(0, heat_at_node[k] - idle_cool * (t - node_arrival_tick[k]))
            if desc:
                nh = cur_heat + g_list[block_idx - 1]
                if nh > H_MAX:
                    continue
            else:
                nh = max(0, cur_heat - heat_loss)

            if t > last_cp_tick[k]:
                cps[k].append((t, nd))

            block_busy_until[block_idx] = t + ttime
            if 1 <= target <= M - 1:
                node_claim[target] = k
            if 1 <= nd <= M - 1:
                node_claim[nd] = -1

            transit_end[k] = t + ttime
            transit_target[k] = target
            new_heat_pc[k] = nh
            last_cp_tick[k] = t

    out = [str(K)]
    for k in range(K):
        line = [str(len(cps[k]))]
        for (tk, nd) in cps[k]:
            line.append(str(tk)); line.append(str(nd))
        out.append(" ".join(line))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
