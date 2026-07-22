import sys, random

def main():
    i = int(sys.argv[1])
    rng = random.Random(4000 + i)

    # ---- difficulty ladder ----
    if i <= 2:
        S, D = 6, 4
    elif i <= 4:
        S, D = 8, 5
    elif i <= 6:
        S, D = 10, 6
    elif i <= 8:
        S, D = 12, 7
    else:
        S, D = 14, 8
    C = 3
    R = 2                           # exactly two scarce-skill (skill 2) holders -> sharp bottleneck
    rare = list(range(R))          # holders are the LOW-index staff (the trap)

    # ---- skills ----
    # everyone can do skill 0; a pool of staff can do skill 1; only `rare` can do skill 2
    skills = [set([0]) for _ in range(S)]
    # guarantee a healthy pool of skill-1 staff among the NON-rare, high-index staff
    pool1 = [s for s in range(S) if s not in rare]
    rng.shuffle(pool1)
    for s in pool1[: max(4, S // 2)]:
        skills[s].add(1)
    for s in rare:
        skills[s].add(2)

    maxshift = [D] * S             # capacity is per-day (one slot/day); the game is per-day freedom
    base = [3] * S                 # shifts beyond `base` incurred during REPAIR cost overtime

    # ---- month of shifts: every day has 1 scarce slot (slot 0) + a few common slots ----
    days = []                      # days[d] = list of (skill, hours) in slot order
    for d in range(D):
        slots = []
        rh = rng.choice([10, 12])             # scarce-shift hours (long)
        slots.append((2, rh))
        n1 = 1 if rng.random() < 0.5 else 0    # at most one skill-1 common slot
        for _ in range(n1):
            slots.append((1, rng.choice([4, 6, 8])))
        nc = rng.choice([1, 2])                # skill-0 common slots
        for _ in range(nc):
            slots.append((0, rng.choice([4, 6, 8])))
        days.append(slots)
    # the FLOOR day gets a long scarce slot; both holders are out there in the floor scenario, so
    # this shift is unavoidably uncovered for ANY roster -> a genuine, sizeable score ceiling.
    fday = rng.randrange(D)
    days[fday][0] = (2, 10)

    # ---- absence scenarios ----
    # Discriminating scenarios: ONE scarce specialist is out for a STRETCH of days. The other
    # specialist can re-cover each scarce hole ONLY if the roster kept them FREE that day, so the
    # cost is a continuum in how well reserves were placed. Plus a small FLOOR (both out one day).
    K = {1: 4, 2: 4, 3: 5, 4: 5, 5: 7, 6: 7, 7: 9, 8: 9, 9: 11, 10: 11}[i]
    half = max(1, D // 2)
    # candidate stretch specs: (holder, day-range)
    stretches = [
        (0, range(0, half)),
        (1, range(half, D)),
        (0, range(half, D)),
        (1, range(0, half)),
        (0, range(0, D)),
        (1, range(0, D)),
    ]
    scen = []
    n_stretch = min(len(stretches), K - 2)          # reserve 1 floor + >=1 noise
    for s in range(n_stretch):
        srng = random.Random(90000 + 131 * i + 17 * s)
        h, rng_days = stretches[s]
        callouts = set((h, d) for d in rng_days)
        for st in range(S):                          # light common noise
            for d in range(D):
                if st not in rare and srng.random() < 0.08:
                    callouts.add((st, d))
        scen.append(sorted(callouts))
    # FLOOR scenario: every scarce holder out on the short-scarce day
    scen.append(sorted(set((h, fday) for h in rare)))
    # NOISE scenarios (common-only callouts; cheap for any sane roster) fill the rest
    while len(scen) < K:
        srng = random.Random(70000 + 131 * i + 7 * len(scen))
        callouts = set()
        for st in range(S):
            for d in range(D):
                if st not in rare and srng.random() < 0.14:
                    callouts.add((st, d))
        scen.append(sorted(callouts))

    U, OT = 100, 40

    # ---- emit ----
    out = []
    out.append("%d %d %d" % (S, D, C))
    out.append(" ".join(map(str, maxshift)))
    out.append(" ".join(map(str, base)))
    for s in range(S):
        sk = sorted(skills[s])
        out.append("%d %s" % (len(sk), " ".join(map(str, sk))))
    for d in range(D):
        out.append(str(len(days[d])))
        for (k, h) in days[d]:
            out.append("%d %d" % (k, h))
    out.append(str(K))
    for cs in scen:
        parts = [str(len(cs))]
        for (st, d) in cs:
            parts.append("%d %d" % (st, d))
        out.append(" ".join(parts))
    out.append("%d %d" % (U, OT))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
