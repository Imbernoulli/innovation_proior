import sys, random

def main():
    tid = int(sys.argv[1])
    rng = random.Random(20250202 + tid * 2711)

    # ---- difficulty / structure ladder (small scale) ----
    if tid <= 2:
        n = 12 + 4 * tid            # 16, 20
    elif tid <= 5:
        n = 24 + 6 * (tid - 2)      # 30, 36, 42
    elif tid <= 8:
        n = 45 + 8 * (tid - 5)      # 53, 61, 69
    else:
        n = 70 + 10 * (tid - 8)     # 80, 90

    density = 8 + tid               # zones per well grows with tid
    m = min(n * density, 1600)

    # fraction of zones that are "injection-favorable": all positive literals so the
    # all-production baseline satisfies 0 of their literals -> never baseline-cleared.
    # These are the weight the baseline MISSES and heuristics must earn back.
    pp = 0.40 + 0.20 * (tid - 1) / 9.0   # 0.40 .. 0.60

    # zone arity range widens with difficulty
    kmax = 3 if tid <= 3 else (4 if tid <= 7 else 5)

    lines = []
    baseline_cleared = 0
    for _ in range(m):
        k = rng.randint(2, kmax)
        k = min(k, n)
        vs = rng.sample(range(1, n + 1), k)

        # weight: mostly small, occasionally a heavy high-enthalpy zone (rewards weighting)
        if rng.random() < 0.15:
            w = rng.randint(20, 60)
        else:
            w = rng.randint(1, 8)

        injection_favorable = (rng.random() < pp)
        if injection_favorable:
            # all literals positive (each watched well must INJECT to satisfy it)
            lits = [v for v in vs]
            # threshold: need t of k wells injecting; skew toward the middle so a single
            # flip rarely clears the zone (creates real local-search structure)
            t = rng.randint(1, k)
            neg_count = 0
        else:
            # mixed polarity zone
            lits = []
            for v in vs:
                if rng.random() < 0.45:
                    lits.append(v)      # +v : well v injecting satisfies
                else:
                    lits.append(-v)     # -v : well v producing satisfies
            neg_count = sum(1 for l in lits if l < 0)
            t = rng.randint(1, k)

        if neg_count >= t:
            baseline_cleared += 1

        lines.append("%d %d %d %s" % (w, k, t, " ".join(str(l) for l in lits)))

    # guarantee the all-production baseline clears at least one zone (B >= 1):
    # force zone 0 to be all-production-favorable with threshold 1.
    if baseline_cleared == 0:
        k0 = min(2, n)
        vs0 = rng.sample(range(1, n + 1), k0)
        lits0 = [-v for v in vs0]
        lines[0] = "%d %d %d %s" % (rng.randint(1, 8), k0, 1,
                                    " ".join(str(l) for l in lits0))

    out = ["%d %d" % (n, m)]
    out.extend(lines)
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
