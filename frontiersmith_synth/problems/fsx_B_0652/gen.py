#!/usr/bin/env python3
"""gen.py <testId> -> one 'conveyor-dual-arm-expiry' instance on stdout.

testId 1..10 = difficulty ladder. Trap cases (3..10) use a large belt with a
NARROW, saturated zone: a continuous stream of expensive, tightly-windowed
zone parts (throughput-bound) plus flank parts scattered far from the zone
(expensive to shuttle to). A single arm juggling both duties -- or two arms
racing for the zone without proactively trading off access -- provably falls
behind a plan that treats the zone's mutex as a schedulable resource.
"""
import sys


def rng(seed):
    # small deterministic xorshift-ish PRNG (no external randomness)
    state = seed & 0xFFFFFFFF
    if state == 0:
        state = 0x9E3779B9
    while True:
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= (state >> 17)
        state ^= (state << 5) & 0xFFFFFFFF
        state &= 0xFFFFFFFF
        yield state


def main():
    testId = int(sys.argv[1])
    seed = 1000 + 37 * testId
    g = rng(seed)

    def nxt():
        return next(g)

    def randint(a, b):
        return a + nxt() % (b - a + 1)

    # ---- difficulty ladder ----
    # base sizes grow with testId; trap tests get an item-rich, contention-heavy zone
    trap_tests = {3, 4, 5, 6, 7, 8, 9, 10}
    is_trap = testId in trap_tests

    if is_trap:
        # Large belt, NARROW zone: zone parts sit close together (cheap to
        # shuttle between), while flank parts are scattered far from the zone
        # (expensive to shuttle between). This rewards SPECIALIZING one arm on
        # the zone while the other camps its own flank, over ping-ponging.
        W = 70 + 6 * testId
        zone_width = 3 + (testId % 3)
    else:
        W = 20 + 4 * testId
        zone_width = max(4, W // 4)
    zlo = W // 2 - zone_width // 2
    zhi = zlo + zone_width
    zlo = max(1, zlo)
    zhi = min(W - 1, zhi)
    if zhi <= zlo:
        zhi = zlo + 2
        zhi = min(W, zhi)

    posL0 = max(0, zlo - randint(1, 3))      # left home, in left flank/near zone
    posR0 = min(W, zhi + randint(1, 3))      # right home, in right flank/near zone

    n = 8 + 2 * testId
    if is_trap:
        n += 16

    speedL_num, speedL_den = 10, 10          # arm L: speed 1.0
    speedR_num, speedR_den = 11 + (testId % 3), 10  # arm R: slightly different kinematics

    items = []  # (pos, t, e, value, pickdur)

    # guaranteed "freebie" items near each home, safely on that arm's own
    # flank (never in the zone) so the checker's flank-only baseline B>0 always
    items.append((max(0, posL0 - 1), 0, 200 + W, 5, 1))
    items.append((min(W, posR0 + 1), 0, 200 + W, 5, 1))

    n_zone_target = (n * 11) // 20 if is_trap else (n * 2) // 5
    horizon = 10 + W
    # a SATURATED, continuous stream: a new zone part becomes available every
    # few ticks, each one expensive to service (large pickdur) and tightly
    # windowed relative to that cost. Zone throughput is thus the true
    # bottleneck: every tick an arm spends away on flank duty is a part that
    # slips past its deadline forever, never serviced by anyone. A dedicated
    # zone arm stays near the throughput ceiling; an arm juggling both duties
    # provably falls behind it.
    zone_gap = 5
    for k in range(n - 2):
        in_zone = (k < n_zone_target)
        if in_zone:
            pos = randint(zlo, zhi)             # zone parts stay tightly clustered
        else:
            # alternate which flank so both arms get exclusive items; parts
            # are spread across the FULL flank (often far from the zone) so a
            # single arm juggling both zone and flank duty pays a real travel
            # tax every time it switches
            if k % 2 == 0:
                hi = max(0, zlo - 1)
                pos = randint(0, hi) if hi > 0 else zlo
            else:
                lo = min(W, zhi + 1)
                pos = randint(lo, W) if lo < W else zhi

        if is_trap and in_zone:
            appear = k * zone_gap + randint(0, 2)
            pickdur = randint(4, 6)
            window = pickdur + randint(2, 5)
        else:
            appear = randint(0, horizon)
            window = randint(6, 14)
            pickdur = randint(1, 3)
        expire = appear + window

        if in_zone:
            value = randint(14, 30) if is_trap else randint(6, 16)
        else:
            value = randint(8, 18) if is_trap else randint(4, 12)

        items.append((pos, appear, expire, value, pickdur))

    n = len(items)
    nodes_pos = [posL0, posR0] + [it[0] for it in items]
    m = len(nodes_pos)

    def build_table(sp_num, sp_den):
        rows = []
        for i in range(m):
            row = []
            for j in range(m):
                if i == j:
                    row.append(0)
                else:
                    d = abs(nodes_pos[i] - nodes_pos[j])
                    t = (d * sp_num + sp_den - 1) // sp_den  # ceil(d*speed)
                    row.append(max(1, t))
            rows.append(row)
        return rows

    TL = build_table(speedL_num, speedL_den)
    TR = build_table(speedR_num, speedR_den)

    out = []
    out.append(f"{W} {zlo} {zhi}")
    out.append(f"{posL0} {posR0}")
    out.append(f"{n}")
    for (pos, t, e, v, pd) in items:
        out.append(f"{pos} {t} {e} {v} {pd}")
    for row in TL:
        out.append(" ".join(map(str, row)))
    for row in TR:
        out.append(" ".join(map(str, row)))

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
