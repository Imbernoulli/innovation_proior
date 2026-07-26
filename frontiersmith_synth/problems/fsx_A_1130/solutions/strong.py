# TIER: strong
import sys, math
from collections import Counter

BUCKET = 6          # time-bucket width used to keep the marginal-value reservation table
HOLD = 8             # planned dwell time per thermal visit


def read_instance():
    tokens = sys.stdin.read().split()
    it = iter(tokens)
    G = int(next(it)); K = int(next(it)); M = int(next(it)); T = int(next(it))
    Vmax = int(next(it)); Sink = int(next(it)); Amax = int(next(it)); A0 = int(next(it))
    gliders = [(int(next(it)), int(next(it))) for _ in range(G)]
    thermals = [tuple(int(next(it)) for _ in range(6)) for _ in range(K)]
    beacons = [tuple(int(next(it)) for _ in range(4)) for _ in range(M)]
    return G, K, M, T, Vmax, Sink, Amax, A0, gliders, thermals, beacons


def thermal_used(pos, centers):
    for k, (ccx, ccy, R, L) in enumerate(centers):
        ddx = pos[0] - ccx; ddy = pos[1] - ccy
        if ddx * ddx + ddy * ddy <= R * R:
            return k
    return -1


def step_toward(px, py, tx, ty, vmax):
    ddx = tx - px; ddy = ty - py
    if ddx == 0 and ddy == 0:
        return (0, 0)
    best = None; bestd = None
    for dx in range(-vmax, vmax + 1):
        rem = vmax * vmax - dx * dx
        if rem < 0:
            continue
        maxdy = int(math.isqrt(rem))
        for dy in range(-maxdy, maxdy + 1):
            ndx = ddx - dx; ndy = ddy - dy
            nd = ndx * ndx + ndy * ndy
            if best is None or nd < bestd:
                best = (dx, dy); bestd = nd
    return best


def estimate_arrival(px, py, cx, cy, vx, vy, vmax, cap):
    t = 0
    for _ in range(5):
        tx = cx + vx * t; ty = cy + vy * t
        d = math.hypot(px - tx, py - ty)
        t = max(0, math.ceil(d / max(1, vmax)))
    return min(t, cap)


def nearest_unclaimed(px, py, beacons, unclaimed):
    return min(unclaimed, key=lambda m: (px - beacons[m][0]) ** 2 + (py - beacons[m][1]) ** 2)


def nearest_any(px, py, beacons):
    return min(range(len(beacons)), key=lambda m: (px - beacons[m][0]) ** 2 + (py - beacons[m][1]) ** 2)


def best_reachable_unclaimed(px, py, t_now, T, Vmax, Sink, alt_now, beacons, unclaimed, ambitious):
    """Value-weighted pick restricted to beacons that are actually still reachable.
    `ambitious=True` (post-thermal-plan) only checks the TIME budget, trusting the
    planned climb for altitude; `ambitious=False` also checks CURRENT altitude.
    Returns None (no silent fallback) if nothing is genuinely reachable."""
    best_i, best_s = None, -1.0
    remaining_T = T - t_now
    for m in unclaimed:
        bx, by, Rb, val = beacons[m]
        d = math.hypot(px - bx, py - by)
        steps_needed = math.ceil(d / Vmax)
        if steps_needed > remaining_T - 1:
            continue
        if not ambitious:
            cost = steps_needed * Sink
            if cost >= alt_now - Sink:
                continue
        s = val / (d + 1.0)
        if s > best_s:
            best_s = s; best_i = m
    return best_i


def main():
    G, K, M, T, Vmax, Sink, Amax, A0, gliders, thermals, beacons = read_instance()

    # ---- pre-pass: flow-over-time assignment of gliders to a drifting thermal timetable.
    # Each thermal is a machine with CONCAVE shared capacity: n co-present gliders each get
    # only floor(L/(1+n^2)). Instead of racing every glider to the single richest thermal
    # (absolute lift), rank candidates by MARGINAL lift given who has already reserved that
    # thermal in that time window, so the fleet naturally time-shares rich thermals and
    # spreads onto otherwise-ignored weak ones when the marginal value there is higher.
    res = {}            # (thermal_idx, time_bucket) -> reserved count
    plan_thermal = [None] * G
    plan_arrival = [0] * G
    unclaimed = set(range(M))
    beacon_target = [None] * G

    order = sorted(range(G), key=lambda g: gliders[g][0] ** 2 + gliders[g][1] ** 2)
    for g in order:
        sx, sy = gliders[g]
        best_k, best_val, best_t = None, 0.0, 0
        for k in range(K):
            cx, cy, vx, vy, R, L = thermals[k]
            t_arr = estimate_arrival(sx, sy, cx, cy, vx, vy, Vmax, T)
            if t_arr + HOLD + 2 > T:
                continue
            transit_cost = t_arr * Sink
            if A0 - transit_cost <= 2 * Sink:
                continue
            bucket = t_arr // BUCKET
            n_next = res.get((k, bucket), 0) + 1
            rate = L // (1 + n_next * n_next)
            net = rate * HOLD - transit_cost
            if net > best_val:
                best_val = net; best_k = k; best_t = t_arr
        bi = None
        if best_k is not None and best_val > 0:
            bucket = best_t // BUCKET
            cx, cy, vx, vy, R, L = thermals[best_k]
            proj_t = best_t + HOLD
            proj_x, proj_y = cx + vx * proj_t, cy + vy * proj_t
            if unclaimed:
                bi = best_reachable_unclaimed(proj_x, proj_y, proj_t, T, Vmax, Sink, A0, beacons, unclaimed, True)
            if bi is not None:
                res[(best_k, bucket)] = res.get((best_k, bucket), 0) + 1
                plan_thermal[g] = best_k
                plan_arrival[g] = best_t
            # else: the climb would leave no time to reach anything -- abandon it below

        if bi is None:
            # no thermal plan, or the plan led nowhere useful: fall back to whatever is
            # honestly reachable straight from the launch point on sink alone.
            plan_thermal[g] = None
            if unclaimed:
                bi = best_reachable_unclaimed(sx, sy, 0, T, Vmax, Sink, A0, beacons, unclaimed, False)
            if bi is None:
                # nothing UNCLAIMED is reachable either -- match trivial's own floor
                # (nearest beacon regardless of any other glider's claim) rather than
                # forcing a doomed detour to a claimed-away or out-of-reach target.
                bi = nearest_any(sx, sy, beacons)

        unclaimed.discard(bi)
        beacon_target[g] = (beacons[bi][0], beacons[bi][1])

    # ---- live execution: chase the assigned thermal window (if any), circle it, then
    # dash for the claimed beacon; opportunistically pick a second unclaimed beacon after.
    pos = [[gliders[g][0], gliders[g][1]] for g in range(G)]
    alt = [A0] * G
    landed = [False] * G
    phase = ["chase_thermal" if plan_thermal[g] is not None else "chase_beacon" for g in range(G)]
    hold_c = [0] * G
    second_leg_done = [False] * G
    moves_out = [[] for _ in range(G)]

    for t in range(1, T + 1):
        centers = [(cx + vx * t, cy + vy * t, R, L) for (cx, cy, vx, vy, R, L) in thermals]
        for g in range(G):
            if landed[g]:
                moves_out[g].append((0, 0))
                continue
            px, py = pos[g]
            k = plan_thermal[g]
            if phase[g] == "chase_thermal":
                ccx, ccy, R, L = centers[k]
                dx, dy = step_toward(px, py, ccx, ccy, Vmax)
                if (px + dx - ccx) ** 2 + (py + dy - ccy) ** 2 <= R * R:
                    phase[g] = "circling"; hold_c[g] = 0
            elif phase[g] == "circling":
                vx_, vy_ = thermals[k][2], thermals[k][3]
                dx, dy = vx_, vy_
                hold_c[g] += 1
                if hold_c[g] >= HOLD or alt[g] >= 0.85 * Amax:
                    phase[g] = "chase_beacon"
                    dx2, dy2 = step_toward(px, py, beacon_target[g][0], beacon_target[g][1], Vmax)
                    dx, dy = dx2, dy2
            else:  # chase_beacon (possibly a second leg)
                tx, ty = beacon_target[g]
                if px == tx and py == ty and not second_leg_done[g] and unclaimed:
                    m2 = best_reachable_unclaimed(px, py, t, T, Vmax, Sink, alt[g], beacons, unclaimed, False)
                    if m2 is None:
                        m2 = nearest_unclaimed(px, py, beacons, unclaimed)
                    unclaimed.discard(m2)
                    beacon_target[g] = (beacons[m2][0], beacons[m2][1])
                    second_leg_done[g] = True
                    tx, ty = beacon_target[g]
                dx, dy = step_toward(px, py, tx, ty, Vmax)

            moves_out[g].append((dx, dy))
            pos[g][0] += dx; pos[g][1] += dy

        used = [thermal_used(pos[g], centers) if not landed[g] else -1 for g in range(G)]
        cnt = Counter(kk for kk in used if kk >= 0)
        for g in range(G):
            if landed[g]:
                continue
            kk = used[g]
            gain = (centers[kk][3] // (1 + cnt[kk] ** 2)) if kk >= 0 else -Sink
            newalt = alt[g] + gain
            if newalt <= 0:
                alt[g] = 0; landed[g] = True
            elif newalt >= Amax:
                alt[g] = Amax
            else:
                alt[g] = newalt

    out = []
    for g in range(G):
        out.append(" ".join("%d %d" % mv for mv in moves_out[g]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
