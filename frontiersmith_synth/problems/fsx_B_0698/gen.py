#!/usr/bin/env python3
# Generator for order-sensitive-blend-ladders (format C). `python3 gen.py <testId>`
# prints ONE instance to stdout. Deterministic: seeded ONLY from testId.
import sys, random

K = 3
EXP_POOL = [0.4, 0.5, 0.6, 0.65, 1.5, 1.7, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.2, 3.4]


def power_mean(items, a):
    tw = 0.0
    s = 0.0
    for w, v in items:
        tw += w
        s += w * (v ** a)
    return (s / tw) ** (1.0 / a)


def best_price(z, corridors, p0):
    best = p0
    for (lo, hi, price) in corridors:
        if all(lo[k] <= z[k] <= hi[k] for k in range(K)) and price > best:
            best = price
    return best


def direct_reachable(lo, hi, cap, feed, exps):
    """Exhaustive check: can ANY direct (<=2 distinct feedstock) integer-volume recipe of
    total volume=cap land inside [lo,hi] on all K attributes, under the TRUE power-law
    mixing rule? feed[i] = (avail, [x1,x2,x3])."""
    F = len(feed)
    for i in range(F):
        avail_i, xi = feed[i]
        if avail_i >= cap:
            if all(lo[k] <= xi[k] <= hi[k] for k in range(K)):
                return True
    for i in range(F):
        avail_i, xi = feed[i]
        if avail_i <= 0:
            continue
        for j in range(F):
            if j == i:
                continue
            avail_j, xj = feed[j]
            if avail_j <= 0:
                continue
            lo_v = max(1, cap - avail_j)
            hi_v = min(cap - 1, avail_i)
            for v in range(lo_v, hi_v + 1):
                w2 = cap - v
                ok = True
                for k in range(K):
                    zk = power_mean([(v, xi[k]), (w2, xj[k])], exps[k])
                    if not (lo[k] <= zk <= hi[k]):
                        ok = False
                        break
                if ok:
                    return True
    return False


def best_direct_price(cap, feed, exps, corridors, p0):
    """Best price achievable by ANY direct (<=2 distinct feedstock) integer-volume recipe
    of total volume=cap, under the TRUE power-law mixing rule (availability-blind upper
    bound -- ignores cross-recipe availability contention, so it's conservative/generous
    in favour of 'selling directly')."""
    F = len(feed)
    best = p0
    for i in range(F):
        avail_i, xi = feed[i]
        if avail_i >= cap:
            pr = best_price(xi, corridors, p0)
            if pr > best:
                best = pr
    for i in range(F):
        avail_i, xi = feed[i]
        if avail_i <= 0:
            continue
        for j in range(F):
            if j == i:
                continue
            avail_j, xj = feed[j]
            if avail_j <= 0:
                continue
            lo_v = max(1, cap - avail_j)
            hi_v = min(cap - 1, avail_i)
            for v in range(lo_v, hi_v + 1):
                w2 = cap - v
                z = [power_mean([(v, xi[k]), (w2, xj[k])], exps[k]) for k in range(K)]
                pr = best_price(z, corridors, p0)
                if pr > best:
                    best = pr
    return best


def gen_normal_corridor(rng, feed, cap_pool, exps, price_lo, price_hi):
    """A corridor guaranteed directly reachable (built around a real 2-feedstock blend),
    used as filler/background pricing tiers."""
    F = len(feed)
    i = rng.randrange(F)
    j = rng.randrange(F)
    while j == i:
        j = rng.randrange(F)
    cap = rng.choice(cap_pool)
    v = rng.randint(1, cap - 1) if cap > 1 else 0
    w2 = cap - v
    z = [power_mean([(max(v, 1e-9), feed[i][1][k]), (max(w2, 1e-9), feed[j][1][k])], exps[k]) for k in range(K)]
    width = [rng.uniform(5.5, 8.5) for _ in range(K)]
    lo = [max(5.0, z[k] - width[k]) for k in range(K)]
    hi = [min(95.0, z[k] + width[k]) for k in range(K)]
    price = rng.randint(price_lo, price_hi)
    return (lo, hi, price)


def build_case(testId):
    rng = random.Random(1000003 * testId + 7)

    sizes = {
        1: (4, 2, 3, False),
        2: (5, 3, 4, False),
        3: (5, 3, 4, False),
        4: (6, 3, 4, False),
        5: (6, 4, 5, False),
        6: (7, 4, 5, True),
        7: (7, 4, 5, True),
        8: (8, 5, 6, True),
        9: (8, 5, 7, True),
        10: (9, 6, 7, True),
    }
    F, M, R, trap = sizes[testId]

    exps = []
    for k in range(K):
        pool = EXP_POOL if testId <= 5 else [e for e in EXP_POOL if e <= 0.55 or e >= 1.7]
        exps.append(rng.choice(pool))

    # feedstocks
    feed = []  # (avail, [x1,x2,x3])
    for _ in range(F):
        avail = rng.randint(10, 34)
        x = [rng.randint(12, 88) for _ in range(K)]
        feed.append([avail, x])

    # tanks (capacities); index 0 == tank 1 ... index M-1 == tank M
    cap = [rng.randint(6, 15) for _ in range(M)]

    p0 = 20
    n_plant = (1 if testId <= 8 else 2) if trap else 0
    plant_slots = []
    if trap:
        # reserve early/late tank slots for premix -> final pairs up front, so filler
        # corridors (built below) are always sized against the FINAL capacities.
        if n_plant >= 1:
            plant_slots.append((0, M - 1))
        if n_plant >= 2:
            mid = 1 if M > 3 else 0
            if mid < M - 1 and mid != 0:
                plant_slots.append((mid, M - 1))
            else:
                plant_slots = plant_slots[:1]
                n_plant = 1

    reserved_slot_set = {s for pair in plant_slots for s in pair}
    free_slots = [i for i in range(M) if i not in reserved_slot_set]
    filler_cap_pool = [cap[i] for i in free_slots] if free_slots else list(cap)

    # filler corridors are built FIRST (against tank slots the trap plants never touch),
    # so the plant loop's economic gate below can see the true final corridor set.
    corridors = []
    n_fill = max(0, R - n_plant)
    for _ in range(n_fill):
        hi_price = 55
        corridors.append(gen_normal_corridor(rng, feed, filler_cap_pool, exps, 32, hi_price))

    planted_price = 0
    plants_done = []
    for plant_idx, (p_slot, f_slot) in enumerate(plant_slots):
        attempts = 0
        planted = False
        while attempts < 40 and not planted:
            attempts += 1
            idxs = rng.sample(range(F), 3)
            A, B, C = idxs
            vA = rng.randint(1, 5)
            vB = rng.randint(1, 5)
            capP = vA + vB
            vC = rng.randint(1, 6)
            capF = capP + vC
            if capF > 20 or capF < 4:
                continue
            # guarantee availability for the planted ingredients (own reserved slack)
            feed[A][0] = max(feed[A][0], vA + rng.randint(6, 16))
            feed[B][0] = max(feed[B][0], vB + rng.randint(6, 16))
            feed[C][0] = max(feed[C][0], vC + rng.randint(6, 16))

            zp = [power_mean([(vA, feed[A][1][k]), (vB, feed[B][1][k])], exps[k]) for k in range(K)]
            zf = [power_mean([(capP, zp[k]), (vC, feed[C][1][k])], exps[k]) for k in range(K)]

            cap[p_slot] = capP
            cap[f_slot] = capF

            width = [1.6, 1.6, 1.6]
            shrink_tries = 0
            while shrink_tries < 12:
                lo = [zf[k] - width[k] for k in range(K)]
                hi = [zf[k] + width[k] for k in range(K)]
                if not direct_reachable(lo, hi, capF, feed, exps):
                    break
                width = [w * 0.6 for w in width]
                shrink_tries += 1
            lo = [zf[k] - width[k] for k in range(K)]
            hi = [zf[k] + width[k] for k in range(K)]
            if direct_reachable(lo, hi, capF, feed, exps):
                continue  # could not carve an unreachable-but-plantable corridor; retry

            price = rng.randint(95, 135) + plant_idx * 10

            # economic gate: selling the premix tank (p_slot) directly plus letting the
            # final tank (f_slot) settle for its best OTHER corridor must be clearly worse
            # than sacrificing the premix -- else an "insightful" solver would rationally
            # decline the trade and the trap loses its bite.
            direct_p_price = best_direct_price(capP, feed, exps, corridors, p0)
            direct_f_price = best_direct_price(capF, feed, exps, corridors, p0)
            margin = price * capF - (direct_f_price * capF + direct_p_price * capP)
            if margin < 0.22 * price * capF:
                continue

            planted_price = max(planted_price, price)
            corridors.append((lo, hi, price))
            plants_done.append((p_slot, f_slot))
            planted = True
        # if genuinely never satisfiable within the attempt budget, fall through with
        # whatever the ladder looked like before this plant (extremely rare in practice).

    rng.shuffle(corridors)

    return F, M, K, len(corridors), exps, feed, cap, corridors, p0


def main():
    if len(sys.argv) != 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    testId = int(sys.argv[1])
    F, M, KK, R, exps, feed, cap, corridors, p0 = build_case(testId)

    out = []
    out.append("%d %d %d %d" % (F, M, KK, R))
    out.append(" ".join("%.6f" % e for e in exps))
    for (avail, x) in feed:
        out.append("%d %s" % (avail, " ".join(str(v) for v in x)))
    for c in cap:
        out.append(str(c))
    for (lo, hi, price) in corridors:
        parts = []
        for k in range(K):
            parts.append("%.6f" % lo[k])
            parts.append("%.6f" % hi[k])
        parts.append(str(price))
        out.append(" ".join(parts))
    out.append(str(p0))
    print("\n".join(out))


if __name__ == "__main__":
    main()
