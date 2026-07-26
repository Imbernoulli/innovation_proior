# TIER: trivial
import sys, math
from collections import Counter


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


def main():
    G, K, M, T, Vmax, Sink, Amax, A0, gliders, thermals, beacons = read_instance()

    # Thermals are ignored entirely: every glider dashes straight for the beacon
    # nearest its own launch point and just glides down.
    targets = []
    for g in range(G):
        sx, sy = gliders[g]
        best = min(range(M), key=lambda m: (sx - beacons[m][0]) ** 2 + (sy - beacons[m][1]) ** 2)
        targets.append((beacons[best][0], beacons[best][1]))

    pos = [[gliders[g][0], gliders[g][1]] for g in range(G)]
    alt = [A0] * G
    landed = [False] * G
    moves_out = [[] for _ in range(G)]

    for t in range(1, T + 1):
        centers = [(cx + vx * t, cy + vy * t, R, L) for (cx, cy, vx, vy, R, L) in thermals]
        for g in range(G):
            if landed[g]:
                moves_out[g].append((0, 0))
                continue
            dx, dy = step_toward(pos[g][0], pos[g][1], targets[g][0], targets[g][1], Vmax)
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
