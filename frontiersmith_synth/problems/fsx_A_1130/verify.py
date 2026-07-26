import sys, math
from collections import Counter


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def parse_ints(tokens, n, ctx):
    vals = []
    for _ in range(n):
        if not tokens:
            fail("truncated input (%s)" % ctx)
        t = tokens.pop(0)
        try:
            vals.append(int(t))
        except ValueError:
            fail("non-integer token %r (%s)" % (t, ctx))
    return vals


def read_instance(path):
    tokens = open(path).read().split()
    G, K, M, T, Vmax, Sink, Amax, A0 = parse_ints(tokens, 8, "header")
    gliders = [tuple(parse_ints(tokens, 2, "glider")) for _ in range(G)]
    thermals = [tuple(parse_ints(tokens, 6, "thermal")) for _ in range(K)]
    beacons = [tuple(parse_ints(tokens, 4, "beacon")) for _ in range(M)]
    return dict(G=G, K=K, M=M, T=T, Vmax=Vmax, Sink=Sink, Amax=Amax, A0=A0,
                gliders=gliders, thermals=thermals, beacons=beacons)


def thermal_used(pos, centers):
    """Lowest-index thermal whose disc currently covers pos, else -1 (cruising)."""
    for k, (ccx, ccy, R, L) in enumerate(centers):
        ddx = pos[0] - ccx; ddy = pos[1] - ccy
        if ddx * ddx + ddy * ddy <= R * R:
            return k
    return -1


def check_beacons(pos_list, beacons, captured, total):
    for g in range(len(pos_list)):
        px, py = pos_list[g]
        for m, (bx, by, Rb, val) in enumerate(beacons):
            if captured[m]:
                continue
            ddx = px - bx; ddy = py - by
            if ddx * ddx + ddy * ddy <= Rb * Rb:
                captured[m] = True
                total[0] += val


def simulate_strict(inst, moves):
    """Score a FULLY-SPECIFIED, externally-supplied move set. Returns (value, err)."""
    G, K, M, T = inst["G"], inst["K"], inst["M"], inst["T"]
    Vmax, Sink, Amax, A0 = inst["Vmax"], inst["Sink"], inst["Amax"], inst["A0"]
    gliders, thermals, beacons = inst["gliders"], inst["thermals"], inst["beacons"]

    pos = [[gliders[g][0], gliders[g][1]] for g in range(G)]
    alt = [A0] * G
    landed = [False] * G
    captured = [False] * M
    total = [0]

    check_beacons(pos, beacons, captured, total)

    for t in range(1, T + 1):
        centers = [(cx + vx * t, cy + vy * t, R, L) for (cx, cy, vx, vy, R, L) in thermals]
        for g in range(G):
            dx, dy = moves[g][t - 1]
            if landed[g]:
                if dx != 0 or dy != 0:
                    return None, "glider %d moved after landing (step %d)" % (g, t)
                continue
            if dx * dx + dy * dy > Vmax * Vmax:
                return None, "glider %d exceeds speed limit at step %d" % (g, t)
            pos[g][0] += dx; pos[g][1] += dy

        used = [thermal_used(pos[g], centers) if not landed[g] else -1 for g in range(G)]
        cnt = Counter(k for k in used if k >= 0)
        for g in range(G):
            if landed[g]:
                continue
            k = used[g]
            if k >= 0:
                L = centers[k][3]
                n = cnt[k]
                gain = L // (1 + n * n)
            else:
                gain = -Sink
            newalt = alt[g] + gain
            if newalt <= 0:
                alt[g] = 0
                landed[g] = True
            elif newalt >= Amax:
                alt[g] = Amax
            else:
                alt[g] = newalt

        check_beacons(pos, beacons, captured, total)

    return total[0], None


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


def build_baseline_value(inst):
    """Checker's own trivial feasible construction: every glider dashes in a straight line
    (thermals ignored) toward the beacon nearest its own launch point. Self-legalizing: it
    tracks its own altitude/landing as it plans, so it never needs a thermal to survive."""
    G, T = inst["G"], inst["T"]
    Vmax, Sink, Amax, A0 = inst["Vmax"], inst["Sink"], inst["Amax"], inst["A0"]
    gliders, thermals, beacons = inst["gliders"], inst["thermals"], inst["beacons"]

    targets = []
    for g in range(G):
        sx, sy = gliders[g]
        best = min(range(len(beacons)), key=lambda m: (sx - beacons[m][0]) ** 2 + (sy - beacons[m][1]) ** 2)
        targets.append((beacons[best][0], beacons[best][1]))

    pos = [[gliders[g][0], gliders[g][1]] for g in range(G)]
    alt = [A0] * G
    landed = [False] * G
    captured = [False] * len(beacons)
    total = [0]
    check_beacons(pos, beacons, captured, total)

    for t in range(1, T + 1):
        for g in range(G):
            if landed[g]:
                continue
            dx, dy = step_toward(pos[g][0], pos[g][1], targets[g][0], targets[g][1], Vmax)
            pos[g][0] += dx; pos[g][1] += dy
            newalt = alt[g] - Sink
            if newalt <= 0:
                alt[g] = 0
                landed[g] = True
            elif newalt >= Amax:
                alt[g] = Amax
            else:
                alt[g] = newalt
        check_beacons(pos, beacons, captured, total)

    return max(1, total[0])


def main():
    inst = read_instance(sys.argv[1])
    G, T = inst["G"], inst["T"]

    out_tokens = open(sys.argv[2]).read().split()
    need = G * 2 * T
    if len(out_tokens) != need:
        fail("expected %d move tokens (G=%d,T=%d), got %d" % (need, G, T, len(out_tokens)))

    moves = []
    idx = 0
    for g in range(G):
        gm = []
        for t in range(T):
            try:
                dx = int(out_tokens[idx]); dy = int(out_tokens[idx + 1])
            except ValueError:
                fail("non-integer move token near glider %d step %d" % (g, t))
            if not (math.isfinite(dx) and math.isfinite(dy)):
                fail("non-finite move")
            gm.append((dx, dy))
            idx += 2
        moves.append(gm)

    F, err = simulate_strict(inst, moves)
    if err is not None:
        fail(err)

    B = build_baseline_value(inst)
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
