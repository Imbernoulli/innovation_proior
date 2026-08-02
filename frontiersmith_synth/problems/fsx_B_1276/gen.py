import sys, random


def main():
    testId = int(sys.argv[1])
    rng = random.Random(20000 + testId * 97)

    # ---- difficulty ladder: universe size grows; trap cases (4,7,10) plant decoys ----
    if testId <= 3:
        N_normal = 22
    elif testId <= 6:
        N_normal = 27
    elif testId <= 8:
        N_normal = 32
    else:
        N_normal = 37

    trap = testId in (4, 7, 10)
    n_decoy = (4 + (testId % 3)) if trap else 0

    f0 = rng.randint(3, 7)
    a0 = rng.randint(3, 7)
    r0 = rng.randint(3, 7)

    comps = []  # each: (margin, f, a, r, doc_cost)
    for _ in range(N_normal):
        df = rng.randint(-3, 3)
        da = rng.randint(-3, 3)
        dr = rng.randint(-3, 3)
        f = min(10, max(0, f0 + df))
        a = min(10, max(0, a0 + da))
        r = min(10, max(0, r0 + dr))
        margin = min(4500, max(200, 1500 + rng.randint(-350, 350)))
        doc_cost = rng.randint(1, 4)
        comps.append((margin, f, a, r, doc_cost))

    # planted trap: comparables that look fine in aggregate but are extreme on ONE
    # functional axis, carrying eye-catching (high) reported margins
    for _ in range(n_decoy):
        axis = rng.randint(0, 2)
        skew = rng.randint(6, 7) * rng.choice([-1, 1])
        vals = [f0, a0, r0]
        for ax in range(3):
            if ax == axis:
                vals[ax] = min(10, max(0, vals[ax] + skew))
            else:
                vals[ax] = min(10, max(0, vals[ax] + rng.randint(-1, 1)))
        f, a, r = vals
        margin = rng.randint(3200, 4300)
        doc_cost = rng.randint(1, 2)
        comps.append((margin, f, a, r, doc_cost))

    rng.shuffle(comps)
    N = len(comps)
    REV = 1_000_000
    MIN_COMPS = 3
    avg_doc_cost = sum(c[4] for c in comps) / N
    # sized so roughly (MIN_COMPS+9) comparables fit at baseline (depth 0) with a little
    # left over for depth upgrades -- deliberately NOT enough to submit the whole universe
    # for free (every submitted comparable carries a baseline documentation cost -- see
    # verify.py's total_doc_cost)
    BUDGET = int(round(1.35 * avg_doc_cost * (MIN_COMPS + 9)))

    out = [str(N), "%d %d %d" % (f0, a0, r0), "%d %d %d" % (REV, BUDGET, MIN_COMPS)]
    for margin, f, a, r, doc_cost in comps:
        out.append("%d %d %d %d %d" % (margin, f, a, r, doc_cost))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
