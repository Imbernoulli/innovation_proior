import sys, math


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def read_input(path):
    toks = open(path).read().split()
    it = iter(toks)
    T = int(next(it))
    rB = float(next(it)); etaV = float(next(it))
    powB = int(next(it)); powV = int(next(it))
    capB = int(next(it)); capV = int(next(it))
    price = [int(next(it)) for _ in range(T)]
    canBuy = [int(next(it)) for _ in range(T)]
    canSell = [int(next(it)) for _ in range(T)]
    return T, rB, etaV, powB, powV, capB, capV, price, canBuy, canSell


def main():
    T, rB, etaV, powB, powV, capB, capV, price, canBuy, canSell = read_input(sys.argv[1])

    toks = open(sys.argv[2]).read().split()
    if len(toks) != 2 * T:
        fail("token count %d != %d" % (len(toks), 2 * T))
    vals = []
    for tk in toks:
        try:
            v = float(tk)
        except ValueError:
            fail("non-numeric token")
        if not math.isfinite(v):
            fail("non-finite value")
        vals.append(v)

    EPS = 1e-6
    cash = 0.0
    sB = 0.0
    sV = 0.0
    for t in range(T):
        uB = vals[2 * t]
        uV = vals[2 * t + 1]
        if uB > powB + EPS or uB < -powB - EPS:
            fail("battery power cap at t=%d" % t)
        if uV > powV + EPS or uV < -powV - EPS:
            fail("vault power cap at t=%d" % t)
        # availability: charging needs a buy window, discharging needs a sell window
        if uB > EPS and not canBuy[t]:
            fail("battery charge without buy window at t=%d" % t)
        if uB < -EPS and not canSell[t]:
            fail("battery discharge without sell window at t=%d" % t)
        if uV > EPS and not canBuy[t]:
            fail("vault charge without buy window at t=%d" % t)
        if uV < -EPS and not canSell[t]:
            fail("vault discharge without sell window at t=%d" % t)
        p = price[t]
        # battery: proportional leak, then act (loss-free round trip)
        sB = sB * rB + uB
        cash -= uB * p
        if sB < -EPS or sB > capB + EPS:
            fail("battery bounds at t=%d" % t)
        # vault: no leak; discharge pays the fixed conversion tax etaV
        sV += uV
        if uV >= 0.0:
            cash -= uV * p
        else:
            cash += etaV * (-uV) * p
        if sV < -EPS or sV > capV + EPS:
            fail("vault bounds at t=%d" % t)

    F = cash

    # internal baseline B: best single BATTERY round trip over legal buy/sell hours.
    buys = [t for t in range(T) if canBuy[t]]
    sells = [t for t in range(T) if canSell[t]]
    pw = [1.0] * T
    for d in range(1, T):
        pw[d] = pw[d - 1] * rB
    B = 1e-9
    for t1 in buys:
        p1 = price[t1]
        for t2 in sells:
            if t2 <= t1:
                continue
            prof = powB * (pw[t2 - t1] * price[t2] - p1)
            if prof > B:
                B = prof

    Fpos = F if F > 0.0 else 0.0
    sc = min(1000.0, 100.0 * Fpos / max(1e-9, B))
    print("F=%.3f B=%.3f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
