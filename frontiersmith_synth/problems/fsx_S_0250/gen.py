import sys, random

def main():
    tid = int(sys.argv[1])
    rng = random.Random(720250 + tid * 1337)

    # ---- difficulty / structure ladder: an R x C archaeology survey grid ----
    # tid 1 tiny (example scale), growing to large/adversarial by tid 10.
    if tid <= 2:
        R, C = 3 + tid, 4 + tid            # (4,5) (5,6)
    elif tid <= 5:
        R, C = 5 + (tid - 2), 6 + (tid - 2)  # (6,7)(7,8)(8,9)
    elif tid <= 8:
        R, C = 8 + (tid - 5), 9 + (tid - 5)  # (9,10)(10,11)(11,12)
    else:
        R, C = 11 + (tid - 8), 12 + (tid - 8)  # (12,13)(13,14)
    n = R * C

    def vid(r, c):
        return r * C + c + 1               # 1-indexed variable id for cell (r,c)

    # clause density (find-hypotheses per cell) grows with difficulty
    density = 6 + tid                       # 7 .. 16
    m = min(n * density, 3600)

    # fraction of PURE-DEEP hypotheses (all positive literals): these are NOT
    # recovered by the all-shallow baseline, tuning the baseline B downward and
    # opening headroom for real optimization. grows a little with tid.
    pp = 0.42 + 0.16 * (tid - 1) / 9.0      # 0.42 .. 0.58

    # spatial cluster radius (Chebyshev window) and max arity widen with tid
    radius = 1 if tid <= 3 else 2
    kmax = 3 if tid <= 4 else (4 if tid <= 8 else 5)

    lines = []
    have_negative = False
    for _ in range(m):
        # pick a cluster center cell and gather nearby cells (a "dig cluster")
        r0 = rng.randrange(R)
        c0 = rng.randrange(C)
        window = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r0 + dr, c0 + dc
                if 0 <= rr < R and 0 <= cc < C:
                    window.append((rr, cc))
        k = rng.randint(2, min(kmax, len(window)))
        cells = rng.sample(window, k)

        # weight: mostly modest finds, occasionally a rare high-value artifact
        # (rewards weight-aware heuristics).
        if rng.random() < 0.15:
            w = rng.randint(20, 60)
        else:
            w = rng.randint(1, 8)

        pure_deep = (rng.random() < pp)
        lits = []
        if pure_deep:
            for (rr, cc) in cells:
                lits.append(vid(rr, cc))           # all "dig deep" -> baseline misses it
        else:
            # STRATUM lean: upper rows (small rr) tend to hold artifacts when dug
            # DEEP (positive literal); lower rows tend to be productive when left
            # SHALLOW (negative literal). This spatial frustration means neither
            # all-shallow nor all-deep is near-optimal.
            for (rr, cc) in cells:
                p_pos = 0.75 - 0.5 * (rr / max(1, R - 1))   # 0.75 (top) .. 0.25 (bottom)
                if rng.random() < p_pos:
                    lits.append(vid(rr, cc))
                else:
                    lits.append(-vid(rr, cc))
            # force at least one negative literal so this clause is in the baseline B
            if all(l > 0 for l in lits):
                idx = rng.randrange(len(lits))
                lits[idx] = -lits[idx]
            have_negative = True

        lines.append("%d %d %s" % (w, k, " ".join(str(l) for l in lits)))

    # guarantee B >= 1: ensure at least one clause has a negative literal
    if not have_negative:
        v = rng.randint(1, n)
        lines.append("%d %d %d" % (rng.randint(1, 8), 1, -v))
        m += 1

    out = []
    out.append("%d %d" % (n, m))
    out.extend(lines)
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
