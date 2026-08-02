import sys, random

# Input layout (all on whitespace-separated tokens, printed on the lines shown):
#   T
#   n0 a0 w0 K
#   ts_1 ... ts_K
#   m d_lo d_hi
#   cost_split cost_delay value_frac
#   Lc
#   nl_1 tl_1 wl_1 al_1
#   ...
#   nl_Lc tl_Lc wl_Lc al_Lc
#   c1 c2 p
#   Vmax Wmax Amax

VMAX = 500
WMAX = 80
AMAX = 1000.0

# testId -> "trap" cases (cheap adaptation: evasion much cheaper than the value at
# stake, so a threshold that merely reproduces the last observed attack gets evaded
# the very next wave). >=3 of the 10 cases are trap cases (we use 5).
TRAP_IDS = {3, 4, 6, 8, 9}


def main():
    i = int(sys.argv[1])
    rng = random.Random(88000 + 17 * i)

    if i <= 3:
        K, Lc, Tbase = 3, 4, 90
    elif i <= 7:
        K, Lc, Tbase = 4, 6, 160
    else:
        K, Lc, Tbase = 6, 8, 240

    n0 = rng.randint(4, 8)
    a0 = round(rng.uniform(70, 200), 2)
    w0 = rng.randint(1, 3)
    m = rng.randint(3, 5)
    d_lo = rng.randint(2, 4)
    d_hi = d_lo + rng.randint(3, 7)

    if i in TRAP_IDS:
        cost_split = round(rng.uniform(0.4, 1.6), 2)
        cost_delay = round(rng.uniform(0.4, 1.6), 2)
        value_frac = round(rng.uniform(0.70, 0.95), 2)
    else:
        cost_split = round(rng.uniform(14.0, 28.0), 2)
        cost_delay = round(rng.uniform(8.0, 18.0), 2)
        value_frac = round(rng.uniform(0.28, 0.50), 2)

    # wave start times, spread across the horizon (spacing is cosmetic here since
    # each wave is scored on its own entity-local transaction stream)
    span_needed = w0 + d_hi + 2
    gap = max(span_needed + 4, Tbase // (K + 1))
    ts_list = []
    t = rng.randint(3, gap)
    for _ in range(K):
        ts_list.append(t)
        t += gap + rng.randint(-2, 4)
    T = ts_list[-1] + span_needed + rng.randint(2, 8)

    # legit clusters: mix of steady / bursty (velocity-trap) / big-ticket (amount-trap)
    clusters = []
    n_bursty = max(1, Lc // 3)
    n_big = max(1, Lc // 3)
    n_steady = Lc - n_bursty - n_big
    slots = list(range(1, T))
    rng.shuffle(slots)
    slot_i = 0

    def next_slot():
        nonlocal slot_i
        s = slots[slot_i % len(slots)]
        slot_i += 1
        return s

    for _ in range(n_steady):
        nl = rng.randint(1, 2)
        wl = rng.randint(1, 3)
        al = round(rng.uniform(25, 70), 2)
        tl = max(1, min(T - wl - 1, next_slot()))
        clusters.append((nl, tl, wl, al))
    for _ in range(n_bursty):
        # elevated velocity like the attacker's own signature, but clearly LOWER
        # count and modest amount -- ordinary "quick errands" legit traffic that a
        # threshold matched to n0 should not need to catch, but an over-tightened
        # velocity cap will.
        nl = max(2, n0 - rng.randint(2, 4))
        wl = max(1, w0 + rng.randint(0, 1))
        al = round(rng.uniform(18, 45), 2)
        tl = max(1, min(T - wl - 1, next_slot()))
        clusters.append((nl, tl, wl, al))
    for _ in range(n_big):
        nl = 1
        wl = 1
        al = round(rng.uniform(1.3, 2.0) * a0, 2)
        al = min(al, AMAX * 0.9)
        tl = max(1, min(T - wl - 1, next_slot()))
        clusters.append((nl, tl, wl, al))
    rng.shuffle(clusters)

    # Calibrate c1 (linear friction) and c2 (convex over-blocking penalty) so that
    # "block every transaction" costs a fixed, substantial FRACTION of the total
    # fraud value at risk -- guarantees the checker's own reference construction
    # (below) is always positive and never trivially near-optimal, at any scale.
    p = round(rng.uniform(1.25, 1.6), 3)
    total_max = value_frac * n0 * a0 * K
    raw_amt = sum(nl * al for (nl, tl, wl, al) in clusters)
    raw_cnt = sum(nl for (nl, tl, wl, al) in clusters)
    ratio = rng.uniform(0.55, 0.85)
    target_friction = ratio * total_max
    lin_target = 0.65 * target_friction
    conv_target = 0.35 * target_friction
    c1 = round(lin_target / max(1.0, raw_amt), 5)
    c2 = round(conv_target / max(1.0, raw_cnt ** p), 5)

    out = []
    out.append(str(T))
    out.append("%d %.2f %d %d" % (n0, a0, w0, K))
    out.append(" ".join(str(x) for x in ts_list))
    out.append("%d %d %d" % (m, d_lo, d_hi))
    out.append("%.4f %.4f %.4f" % (cost_split, cost_delay, value_frac))
    out.append(str(len(clusters)))
    for (nl, tl, wl, al) in clusters:
        out.append("%d %d %d %.2f" % (nl, tl, wl, al))
    out.append("%.4f %.4f %.4f" % (c1, c2, p))
    out.append("%d %d %.2f" % (VMAX, WMAX, AMAX))

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
