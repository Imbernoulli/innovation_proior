import sys, random, math

M = 1000       # neutral price at non-window hours (never buyable/sellable)
POWB = 10
POWV = 10
CAPB = 15
CAPV = 15
GAP = 5
PAD = 6

# (n_short, n_trap) per test id -- difficulty + trap-heaviness pattern.
# Short-heavy early (greedy ~= strong), trap-heavy later (greedy far from strong).
PATT = {1: (5, 1), 2: (4, 2), 3: (4, 2), 4: (3, 2), 5: (2, 4),
        6: (2, 4), 7: (3, 4), 8: (2, 4), 9: (3, 3), 10: (2, 4)}


def main():
    tid = int(sys.argv[1])
    rng = random.Random(90210 + 1009 * tid)

    rB = round(rng.uniform(0.975, 0.990), 4)     # battery hourly retention (proportional leak)
    etaV = round(rng.uniform(0.820, 0.895), 4)   # vault round-trip efficiency (fixed tax)
    dstar = math.log(etaV) / math.log(rB)        # loss-crossover horizon

    nS, nT = PATT.get(tid, (3, 3))

    opps = []
    for k in range(nS):
        hi = max(3, int(dstar) - 1)
        ds = rng.randint(3, hi)
        if ds >= dstar:
            ds = max(2, int(math.floor(dstar)) - 1)
        opps.append(['S', ds, (k == 0)])        # first short = "big" -> sets baseline B
    for k in range(nT):
        lo = int(math.ceil(dstar)) + 10
        hiL = int(math.ceil(dstar)) + 34
        if hiL < lo:
            hiL = lo + 5
        dl = rng.randint(lo, hiL)
        opps.append(['T', dl, False])
    rng.shuffle(opps)

    windows = []   # (tb, ts, dip, spike)
    t = PAD
    for kind, dur, isbig in opps:
        tb = t
        dip = rng.randint(760, 860)
        if kind == 'S':
            q = rng.randint(112, 150) if isbig else rng.randint(45, 80)
            ds = dur
            spike = (dip + q) / (rB ** ds)       # battery per-unit profit ~= q
            spike = int(round(spike))
            # guarantee battery profitable after rounding
            while (rB ** ds) * spike - dip <= 6:
                spike += 1
            ts = tb + ds
        else:  # trap
            qv = rng.randint(120, 168)
            spike = (dip + qv) / etaV            # vault per-unit profit ~= qv
            spike = int(round(spike))
            ts = tb + dur
            # trap must LOSE on the battery and WIN in the vault
            while (rB ** (ts - tb)) * spike - dip > -22:
                ts += 2                          # deepen the hold so the leak bites
            while etaV * spike - dip <= 20:
                spike += 2
            while (rB ** (ts - tb)) * spike - dip > -22:
                ts += 2
        windows.append((tb, ts, int(dip), int(spike)))
        t = ts + GAP
    T = t + PAD

    price = [M] * T
    canBuy = [0] * T
    canSell = [0] * T
    for tb, ts, dip, spike in windows:
        price[tb] = dip
        canBuy[tb] = 1
        price[ts] = spike
        canSell[ts] = 1

    out = []
    out.append("%d %.6f %.6f %d %d %d %d" % (T, rB, etaV, POWB, POWV, CAPB, CAPV))
    out.append(" ".join(str(x) for x in price))
    out.append(" ".join(str(x) for x in canBuy))
    out.append(" ".join(str(x) for x in canSell))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
