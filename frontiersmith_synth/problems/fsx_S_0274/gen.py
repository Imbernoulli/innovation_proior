import sys, random

def draw_channel(rng, D):
    # bias channel 1 to appear at ~half the uniform rate, so the all-channel-1
    # baseline is a genuinely mediocre reference (keeps B positive but modest).
    c = rng.randint(1, D)
    if c == 1 and rng.random() < 0.5:
        c = rng.randint(2, D)
    return c

def main():
    tid = int(sys.argv[1])
    rng = random.Random(730274 + tid * 29)

    # ---- difficulty / structure ladder (small scale) ----
    if tid <= 2:
        n = 8 + 4 * tid          # 12, 16
        D = 3
    elif tid <= 5:
        n = 20 + 6 * (tid - 2)   # 26, 32, 38
        D = 4
    elif tid <= 8:
        n = 40 + 6 * (tid - 5)   # 46, 52, 58
        D = 5
    else:
        n = 60 + 10 * (tid - 8)  # 70, 80
        D = 6

    density = 10 + tid           # requirements per station grows with tid
    m = min(n * density, 1500)

    kmax = 2 if tid <= 3 else (3 if tid <= 7 else 4)

    lines = []
    for j in range(m):
        k = rng.randint(1, kmax)
        k = min(k, n)
        vs = rng.sample(range(1, n + 1), k)

        # weight: mostly small, occasionally a heavy high-priority relay
        if rng.random() < 0.15:
            w = rng.randint(20, 60)
        else:
            w = rng.randint(1, 8)

        pairs = []
        for v in vs:
            a = draw_channel(rng, D)
            pairs.append((v, a))

        # guarantee B >= 1: force the very first requirement to name channel 1
        if j == 0:
            pairs[0] = (pairs[0][0], 1)

        toks = [str(w), str(k)]
        for (v, a) in pairs:
            toks.append(str(v)); toks.append(str(a))
        lines.append(" ".join(toks))

    out = ["%d %d %d" % (n, m, D)]
    out.extend(lines)
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
