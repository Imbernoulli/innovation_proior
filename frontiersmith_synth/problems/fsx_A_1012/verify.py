#!/usr/bin/env python3
# Deterministic checker for "Haul-Switchback Pipeline" (format C, maximize completed loads).
# CLI: python3 verify.py <in> <out> <ans>   (ans is ignored)
# Prints "... Ratio: <r>" with r in [0,1]. Always exits 0.
import sys

MAX_L = 8000  # generous upper bound on checkpoints per truck for our instance sizes


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def simulate_single_truck_deliveries(t_list, g_list, heat_loss_up, idle_cool, H_MAX, T_horizon):
    """Checker's internal trivial baseline: ONE truck cycles pit<->crusher continuously,
    with just-in-time cooldown stops when needed; all other trucks are ignored (idle)."""
    M = len(t_list)
    t_cur = 0
    heat = 0
    deliveries = 0
    while True:
        for i in range(M):
            g = g_list[i]
            if heat + g > H_MAX:
                need = heat + g - H_MAX
                w = -(-need // idle_cool)
                t_cur += w
                if t_cur > T_horizon:
                    return deliveries
                heat = max(0, heat - w * idle_cool)
            t_cur += t_list[i]
            if t_cur > T_horizon:
                return deliveries
            heat += g
        deliveries += 1
        for i in reversed(range(M)):
            t_cur += t_list[i]
            if t_cur > T_horizon:
                return deliveries
            heat = max(0, heat - heat_loss_up)


def main():
    try:
        itoks = open(sys.argv[1]).read().split()
        ip = iter(itoks)
        M = int(next(ip)); K = int(next(ip)); T_horizon = int(next(ip))
        H_MAX = int(next(ip)); idle_cool = int(next(ip)); heat_loss_up = int(next(ip))
        t_list = [int(next(ip)) for _ in range(M)]
        g_list = [int(next(ip)) for _ in range(M)]
    except Exception:
        fail("bad instance")

    try:
        otoks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")

    if not otoks:
        fail("empty output")

    try:
        op = iter(otoks)

        def nxt_int():
            tok = next(op)
            v = int(tok)  # raises on 'nan'/'inf'/floats/garbage -> caught below
            return v

        K_out = nxt_int()
        if K_out != K:
            fail("truck count %d != K=%d" % (K_out, K))

        block_events = [[] for _ in range(M + 1)]   # block index 1..M -> list of (start,end)
        node_events = [[] for _ in range(M + 1)]    # node index 1..M-1 -> list of (start,end_or_INF)
        INF = 10 ** 15
        deliveries = 0

        for k in range(K):
            L = nxt_int()
            if L < 1 or L > MAX_L:
                fail("truck %d: bad checkpoint count %d" % (k, L))
            cps = []
            for _ in range(L):
                tick = nxt_int()
                node = nxt_int()
                cps.append((tick, node))

            if cps[0] != (0, 0):
                fail("truck %d: must start at (tick=0, node=0)" % k)

            heat = 0
            # A delivery only counts if the truck reached the crusher via an unbroken
            # ascent of node values all the way from the pit (i.e. it actually loaded at
            # node 0 and never turned back before reaching M) -- NOT merely any M-1->M
            # crossing. Otherwise a truck could shuttle back and forth over the LAST
            # block forever and rack up unbounded deliveries without ever visiting the
            # pit. `since_pit` is True exactly while the truck's node history since its
            # last pit visit has been non-decreasing.
            since_pit = True
            j = 0
            while j < L - 1:
                ta, na = cps[j]
                tb, nb = cps[j + 1]
                if tb <= ta:
                    fail("truck %d: ticks not strictly increasing" % k)
                if tb > T_horizon:
                    fail("truck %d: tick %d exceeds horizon %d" % (k, tb, T_horizon))
                if not (0 <= na <= M) or not (0 <= nb <= M):
                    fail("truck %d: node out of range" % k)
                if na == nb:
                    # dwell at na for (tb-ta) ticks: cools down
                    heat = max(0, heat - idle_cool * (tb - ta))
                elif nb == na + 1:
                    block_idx = nb
                    if tb - ta != t_list[block_idx - 1]:
                        fail("truck %d: block %d crossing time wrong" % (k, block_idx))
                    g = g_list[block_idx - 1]
                    new_heat = heat + g
                    if new_heat > H_MAX:
                        fail("truck %d: heat cap exceeded on block %d" % (k, block_idx))
                    heat = new_heat
                    block_events[block_idx].append((ta, tb))
                    if nb == M and since_pit:
                        deliveries += 1
                        since_pit = False  # must return to the pit before the next delivery counts
                elif nb == na - 1:
                    block_idx = na
                    if tb - ta != t_list[block_idx - 1]:
                        fail("truck %d: block %d crossing time wrong" % (k, block_idx))
                    heat = max(0, heat - heat_loss_up)
                    block_events[block_idx].append((ta, tb))
                    since_pit = (nb == 0)
                else:
                    fail("truck %d: illegal move %d -> %d" % (k, na, nb))
                j += 1

            # node (pullout) presence intervals: group consecutive same-node checkpoints
            idx = 0
            while idx < L:
                node = cps[idx][1]
                start = cps[idx][0]
                run_end_idx = idx
                while run_end_idx + 1 < L and cps[run_end_idx + 1][1] == node:
                    run_end_idx += 1
                if 1 <= node <= M - 1:
                    # departure tick = the run's OWN last timestamp (that's when the truck
                    # starts crossing away); NOT the next checkpoint's tick, which is the
                    # arrival time at the *following* node.
                    end = cps[run_end_idx][0] if run_end_idx < L - 1 else INF
                    node_events[node].append((start, end))
                idx = run_end_idx + 1

        # block capacity 1 (half-open intervals: back-to-back crossings are fine)
        for b in range(1, M + 1):
            evs = sorted(block_events[b])
            for a in range(1, len(evs)):
                if evs[a][0] < evs[a - 1][1]:
                    fail("block %d capacity violated" % b)

        # pullout capacity 1 (closed intervals: simultaneous opposite-side arrival forbidden)
        for p in range(1, M):
            evs = sorted(node_events[p])
            for a in range(1, len(evs)):
                if evs[a][0] <= evs[a - 1][1]:
                    fail("pullout %d capacity violated" % p)

        F = float(deliveries)
    except SystemExit:
        raise
    except Exception:
        fail("malformed output")

    B = float(simulate_single_truck_deliveries(t_list, g_list, heat_loss_up, idle_cool, H_MAX, T_horizon))
    if B <= 0:
        B = 1e-9

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.2f B=%.2f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
