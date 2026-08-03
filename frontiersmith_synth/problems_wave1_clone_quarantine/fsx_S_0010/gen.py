import sys, random

def main():
    tid = int(sys.argv[1])
    rng = random.Random(90210 + tid * 17)

    # ---- difficulty / structure ladder (small scale) ----
    if tid <= 2:
        n = 12 + 4 * tid            # 16, 20
    elif tid <= 5:
        n = 24 + 6 * (tid - 2)      # 30, 36, 42
    elif tid <= 8:
        n = 45 + 8 * (tid - 5)      # 53, 61, 69
    else:
        n = 70 + 10 * (tid - 8)     # 80, 90

    density = 8 + tid               # constraints per satellite grows with tid
    m = min(n * density, 1600)

    # fraction of clauses that are pure-prograde (all positive literals):
    # these are UNsatisfied by the all-retrograde baseline, tuning B/W across tests.
    pp = 0.40 + 0.20 * (tid - 1) / 9.0   # 0.40 .. 0.60

    # clause arity range widens with difficulty
    kmax = 3 if tid <= 3 else (4 if tid <= 7 else 5)

    lines = []
    have_negative = False
    for _ in range(m):
        k = rng.randint(2, kmax)
        k = min(k, n)
        vs = rng.sample(range(1, n + 1), k)

        # weight: mostly small, occasionally a heavy debris field (rewards weighting)
        if rng.random() < 0.15:
            w = rng.randint(20, 60)
        else:
            w = rng.randint(1, 8)

        pure_pos = (rng.random() < pp)
        lits = []
        if pure_pos:
            for v in vs:
                lits.append(v)          # all prograde -> not cleared by baseline
        else:
            for v in vs:
                if rng.random() < 0.45:
                    lits.append(v)
                else:
                    lits.append(-v)
            # force at least one negative literal so the clause is in the baseline
            if all(l > 0 for l in lits):
                idx = rng.randrange(len(lits))
                lits[idx] = -lits[idx]
            have_negative = True

        lines.append("%d %d %s" % (w, k, " ".join(str(l) for l in lits)))

    # guarantee B >= 1 : ensure at least one clause with a negative literal exists
    if not have_negative:
        v = rng.randint(1, n)
        lines[0] = "%d %d %d" % (rng.randint(1, 8), 1, -v)

    out = ["%d %d" % (n, m)]
    out.extend(lines)
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
