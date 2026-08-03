import sys, random

def main():
    tid = int(sys.argv[1])
    rng = random.Random(424242 + tid * 97)

    # ---- difficulty / structure ladder (medium scale) ----
    if tid <= 2:
        n = 10 + 6 * tid            # 16, 22
    elif tid <= 5:
        n = 24 + 8 * (tid - 2)      # 32, 40, 48
    elif tid <= 8:
        n = 50 + 9 * (tid - 5)      # 59, 68, 77
    else:
        n = 80 + 15 * (tid - 8)     # 95, 110

    # domain size grows with difficulty (D=2 is plain weighted Max-SAT)
    if tid <= 2:
        D = 2
    elif tid <= 5:
        D = 3
    elif tid <= 8:
        D = 4
    else:
        D = 5

    density = 6 + tid               # requirements per module grows with tid
    m = min(n * density, 1500)

    # fraction of requirements that contain NO default-config (a=0) demand:
    # these are UNmet by the all-default baseline, tuning B/W across tests so the
    # baseline is a genuinely mediocre reference.
    pp = 0.45 + 0.20 * (tid - 1) / 9.0   # 0.45 .. 0.65

    # requirement arity range widens with difficulty
    kmax = 3 if tid <= 3 else (4 if tid <= 7 else 5)

    lines = []
    have_default = False
    for _ in range(m):
        k = rng.randint(2, kmax)
        k = min(k, n)
        vs = rng.sample(range(1, n + 1), k)

        # weight: mostly small, occasionally a heavy critical requirement (rewards weighting)
        if rng.random() < 0.15:
            w = rng.randint(20, 60)
        else:
            w = rng.randint(1, 8)

        no_default = (rng.random() < pp)
        pairs = []
        if no_default:
            # every demand asks for a non-default config -> not met by baseline
            for v in vs:
                a = rng.randint(1, D - 1)
                pairs.append((v, a))
        else:
            for v in vs:
                a = rng.randint(0, D - 1)
                pairs.append((v, a))
            # force at least one default demand so the requirement is in the baseline
            if all(a != 0 for (_, a) in pairs):
                idx = rng.randrange(len(pairs))
                pairs[idx] = (pairs[idx][0], 0)
            have_default = True

        flat = []
        for (v, a) in pairs:
            flat.append(str(v)); flat.append(str(a))
        lines.append("%d %d %s" % (w, k, " ".join(flat)))

    # guarantee B >= 1 : ensure at least one requirement with a default demand exists
    if not have_default:
        v = rng.randint(1, n)
        lines[0] = "%d %d %d %d" % (rng.randint(1, 8), 1, v, 0)

    out = ["%d %d %d" % (n, m, D)]
    out.extend(lines)
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
