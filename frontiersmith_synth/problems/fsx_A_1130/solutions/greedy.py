# TIER: greedy
import sys, math
from collections import Counter

HOLD_CAP = 8            # steps a glider will patiently sit in its chosen thermal


def estimate_arrival(px, py, cx, cy, vx, vy, vmax, cap):
    t = 0
    for _ in range(5):
        tx = cx + vx * t; ty = cy + vy * t
        d = math.hypot(px - tx, py - ty)
        t = max(0, math.ceil(d / max(1, vmax)))
    return min(t, cap)


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


def best_beacon(px, py, beacons):
    """Naive per-glider heuristic: maximize value / (distance+1); no coordination
    with other gliders, so several may aim at the very same beacon."""
    best_i, best_s = -1, -1.0
    for m, (bx, by, Rb, val) in enumerate(beacons):
        d = math.hypot(px - bx, py - by)
        s = val / (d + 1.0)
        if s > best_s:
            best_s = s; best_i = m
    return best_i


def nearest_beacon(px, py, beacons):
    return min(range(len(beacons)), key=lambda m: (px - beacons[m][0]) ** 2 + (py - beacons[m][1]) ** 2)


def reachable_best_beacon(px, py, alt, remaining_T, Vmax, Sink, beacons):
    """Value-weighted pick, but only among beacons it can plausibly still glide to;
    falls back to the closest beacon (best effort) if none look reachable."""
    best_i, best_s = -1, -1.0
    for m, (bx, by, Rb, val) in enumerate(beacons):
        d = math.hypot(px - bx, py - by)
        steps_needed = math.ceil(d / Vmax)
        cost = steps_needed * Sink
        if steps_needed <= remaining_T and cost < alt - Sink:
            s = val / (d + 1.0)
            if s > best_s:
                best_s = s; best_i = m
    if best_i < 0:
        return nearest_beacon(px, py, beacons)
    return best_i


def ambitious_best_beacon(px, py, remaining_T, Vmax, beacons):
    """Value-weighted pick trusting a planned climb for altitude; only filters on
    whether there is even enough TIME left to physically cover the distance.
    Returns None (no silent fallback) if nothing is even time-reachable."""
    best_i, best_s = -1, -1.0
    for m, (bx, by, Rb, val) in enumerate(beacons):
        d = math.hypot(px - bx, py - by)
        steps_needed = math.ceil(d / Vmax)
        if steps_needed > remaining_T - 1:
            continue
        s = val / (d + 1.0)
        if s > best_s:
            best_s = s; best_i = m
    return best_i if best_i >= 0 else None


def main():
    G, K, M, T, Vmax, Sink, Amax, A0, gliders, thermals, beacons = read_instance()

    # "Obvious" choice: each glider independently grabs the CLOSEST thermal that is
    # worth the detour ASSUMING IT HAS IT ALL TO ITSELF (n=1: rate = floor(L/2) per
    # step beats the transit sink cost). No glider looks at what any other glider is
    # doing or reserves anything, so whenever several gliders happen to launch near the
    # same rich thermal, they all independently reach the same conclusion and pile in.
    dom_idx = [None] * G
    dom_arr = [0] * G
    for g in range(G):
        sx, sy = gliders[g]
        cand = []
        for k in range(K):
            cx, cy, vx, vy, R, L = thermals[k]
            t_arr = estimate_arrival(sx, sy, cx, cy, vx, vy, Vmax, T)
            if t_arr + HOLD_CAP + 2 > T:
                continue
            transit_cost = t_arr * Sink
            if transit_cost >= A0 - 2 * Sink:
                continue
            solo_rate = L // 2                      # n=1, crowding never considered
            net = solo_rate * HOLD_CAP - transit_cost
            if net > 0:
                cand.append((t_arr, k))
        if cand:
            dom_arr[g], dom_idx[g] = min(cand)

    # Beacon choice is also naive/uncoordinated and value-weighted -- fixed once from the
    # LAUNCH point (diverse across gliders since they start in different places) so a rich
    # thermal is worth detouring for: it is the only way to *extend range* to a beacon that
    # would otherwise be out of reach on sink alone. Gliders WITH a thermal plan may aim
    # ambitiously (assume the climb will pay for the extra reach, filtered only by whether
    # there is even enough TIME left to physically cover the distance); gliders with no
    # thermal plan (or whose climb would strand them with nothing reachable afterward) size
    # the target to what current altitude can actually still reach.
    beacon_target = [None] * G
    for g in range(G):
        sx, sy = gliders[g]
        bi = None
        if dom_idx[g] is not None:
            proj_t = dom_arr[g] + HOLD_CAP
            kk = dom_idx[g]
            tcx, tcy, tvx, tvy, tR, tL = thermals[kk]
            proj_x, proj_y = tcx + tvx * proj_t, tcy + tvy * proj_t
            bi = ambitious_best_beacon(proj_x, proj_y, T - proj_t, Vmax, beacons)
            if bi is None:
                dom_idx[g] = None    # the climb would leave nothing reachable: skip it
        if bi is None:
            bi = reachable_best_beacon(sx, sy, A0, T, Vmax, Sink, beacons)
        beacon_target[g] = (beacons[bi][0], beacons[bi][1])

    pos = [[gliders[g][0], gliders[g][1]] for g in range(G)]
    alt = [A0] * G
    landed = [False] * G
    phase = ["chase_thermal" if dom_idx[g] is not None else "chase_beacon" for g in range(G)]
    hold = [0] * G
    chase_t = [0] * G                 # steps spent chasing the thermal
    moves_out = [[] for _ in range(G)]

    for t in range(1, T + 1):
        centers = [(cx + vx * t, cy + vy * t, R, L) for (cx, cy, vx, vy, R, L) in thermals]
        for g in range(G):
            if landed[g]:
                moves_out[g].append((0, 0))
                continue
            px, py = pos[g]
            if phase[g] == "chase_thermal":
                ccx, ccy, R, L = centers[dom_idx[g]]
                chase_t[g] += 1
                giving_up = alt[g] < 3 * Sink
                if not giving_up:
                    dx, dy = step_toward(px, py, ccx, ccy, Vmax)
                    if (px + dx - ccx) ** 2 + (py + dy - ccy) ** 2 <= R * R:
                        phase[g] = "circling"; hold[g] = 0
                else:
                    bi = reachable_best_beacon(px, py, alt[g], T - t, Vmax, Sink, beacons)
                    beacon_target[g] = (beacons[bi][0], beacons[bi][1])
                    phase[g] = "chase_beacon"
                    dx, dy = step_toward(px, py, beacon_target[g][0], beacon_target[g][1], Vmax)
            elif phase[g] == "circling":
                ccx, ccy, vx_, vy_ = centers[dom_idx[g]][0], centers[dom_idx[g]][1], thermals[dom_idx[g]][2], thermals[dom_idx[g]][3]
                dx, dy = vx_, vy_             # ride the drift exactly, stay locked in; hold
                hold[g] += 1                  # for the FULL planned window (no early exit),
                if hold[g] >= HOLD_CAP or alt[g] >= Amax:   # so drift keeps gliders apart
                    phase[g] = "chase_beacon"
            else:  # chase_beacon
                dx, dy = step_toward(px, py, beacon_target[g][0], beacon_target[g][1], Vmax)

            moves_out[g].append((dx, dy))
            pos[g][0] += dx; pos[g][1] += dy

        used = [thermal_used(pos[g], centers) if not landed[g] else -1 for g in range(G)]
        cnt = Counter(k for k in used if k >= 0)
        for g in range(G):
            if landed[g]:
                continue
            k = used[g]
            gain = (centers[k][3] // (1 + cnt[k] ** 2)) if k >= 0 else -Sink
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
