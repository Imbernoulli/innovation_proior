import sys, math, random

# ---- Catastrophe portfolio steer: hurricane-book underwriting ----
# gen.py <testId>  -> prints ONE instance to stdout. testId 1..10 = size/trap ladder.
#
# Layout:
#   N C K OVER_MULT
#   K lines: cx cy R L sev            (storm s: center, footprint radius, accumulation
#                                       limit, severity per-mille)
#   N lines: x y e p tech             (candidate i: location, exposure, premium, technical
#                                       (non-catastrophe) expected loss)

OVER_MULT = 2


def dist2(ax, ay, bx, by):
    return (ax - bx) ** 2 + (ay - by) ** 2


def gen(testId):
    rng = random.Random(700000 + 97 * testId)

    N = 10 + (testId - 1) * 2          # 10 .. 28
    C = max(4, (N * 2) // 5)           # underwriting capacity (~40% of N)
    K = 3 + (testId - 1) // 3          # 3 .. 6 storms

    hot = 0
    hcx, hcy, hR = rng.randint(150, 850), rng.randint(150, 850), rng.randint(120, 190)
    storms = [[hcx, hcy, hR, None, rng.randint(500, 820)]]

    # other storms are kept spatially clear of the hot storm's footprint so their
    # severities never compound with the planted trap by sheer coincidence -- the
    # accumulation-limit trap stays a clean, localized story.
    for s in range(1, K):
        R = rng.randint(120, 190)
        buf = hR + R + 160
        for _ in range(200):
            cx = rng.randint(60, 940)
            cy = rng.randint(60, 940)
            if dist2(cx, cy, hcx, hcy) > buf * buf:
                break
        sev = rng.randint(500, 820)
        storms.append([cx, cy, R, None, sev])

    policies = []  # [x, y, e, p, tech]

    # test ids 1-3 are calm ladder-warmup cases; 4-10 (7 of 10) are the planted traps
    severe_trap = testId >= 4

    # --- planted trap cluster: many high-margin policies packed inside the hot
    #     storm's footprint. A single-pass margin-greedy cannot see that writing
    #     all of them blows the storm's accumulation limit. ---
    extra = rng.randint(1, 4)
    cluster_n = C + extra
    cluster_exposures = []
    for k in range(cluster_n):
        rad = rng.uniform(0.0, 0.85) * hR
        ang = rng.uniform(0.0, 2.0 * math.pi)
        x = int(round(hcx + rad * math.cos(ang)))
        y = int(round(hcy + rad * math.sin(ang)))
        x = min(max(x, 0), 1000)
        y = min(max(y, 0), 1000)
        e = rng.randint(30, 55)
        tech = rng.randint(20, 40)
        margin = rng.randint(45, 70)      # deliberately the richest margins in the file
        p = tech + margin
        policies.append([x, y, e, p, tech])
        cluster_exposures.append(e)

    total_cluster_exposure = sum(cluster_exposures)
    # severe: writing the whole cluster breaches badly, but a partial slice fits.
    # calm: the limit is generous enough that even the whole cluster mostly fits.
    frac = 0.45 if severe_trap else 0.92
    L_hot = max(20, int(total_cluster_exposure * frac))
    storms[hot][3] = L_hot

    # --- diversifier / decoy candidates: kept OUT of the hot footprint so a
    #     strategy that steers away from storm 0 has somewhere real to go. Mostly
    #     positive margin (lower than the cluster's), a few genuinely inadequate
    #     decoys to keep the "price-adequacy" filter meaningful. ---
    remaining = N - len(policies)
    diversifiers = []
    for k in range(remaining):
        for _ in range(50):
            x = rng.randint(0, 1000)
            y = rng.randint(0, 1000)
            if dist2(x, y, hcx, hcy) > (hR + 40) ** 2:
                break
        e = rng.randint(20, 60)
        tech = rng.randint(15, 45)
        if rng.random() < 0.15:
            margin = rng.randint(-20, -1)   # inadequate decoy
        else:
            margin = rng.randint(8, 38)
        p = tech + margin
        policies.append([x, y, e, p, tech])
        diversifiers.append((x, y, e))

    # --- set limits for the other storms generously from ACTUAL realized coverage
    #     so they never bind (keeps the trap story localized to storm `hot`,
    #     while remaining a real, checkable constraint). ---
    for s in range(K):
        if s == hot:
            continue
        cx, cy, R, _, sev = storms[s]
        cov = 0
        for (x, y, e, p, tech) in policies:
            if dist2(x, y, cx, cy) <= R * R:
                cov += e
        mult = rng.uniform(1.5, 2.2)
        storms[s][3] = max(20, int(cov * mult) + 5)

    out = []
    out.append(f"{N} {C} {K} {OVER_MULT}")
    for (cx, cy, R, L, sev) in storms:
        out.append(f"{cx} {cy} {R} {L} {sev}")
    for (x, y, e, p, tech) in policies:
        out.append(f"{x} {y} {e} {p} {tech}")
    return "\n".join(out) + "\n"


def main():
    testId = int(sys.argv[1])
    sys.stdout.write(gen(testId))


if __name__ == "__main__":
    main()
