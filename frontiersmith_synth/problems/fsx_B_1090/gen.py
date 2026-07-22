#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE instance of the scenario-consensus battery
commitment problem to stdout.  Deterministic: all randomness is seeded by
testId only.

Instance layout (stdout):
    line 1:  T S
    line 2:  Emax Pmax eta E0 lam mu
    then S scenario blocks of 3 lines each:
        rho_s                  (terminal value per unit of stored energy)
        p_s[1..T]              (prices)
        o_s[1..T]              (outage flags, 0/1)

Difficulty ladder (testId 1..10): benign decorrelated outages first, then
planted adversarial structure an expected-value dispatcher cannot see:
  1-3   benign: outages uncorrelated with price
  4-5   storm scenario voids the top mean-price hours
  6-7   two storms (morning + evening peaks), steeper shortfall penalty
  8     mirage: one scenario pumps the mean price in hours it itself voids
  9     low-floor scenario (depressed prices, no outages) + storm
  10    mirage + storm + low floor combined, highest penalties
"""
import sys
import random


def bump(h, c, w):
    d = abs(h - c)
    d = min(d, 24 - d)
    return pow(2.718281828459045, -0.5 * (d / w) ** 2)


def build(t):
    rng = random.Random(1000003 * t + 77)
    T = 168
    S = 18 + t                       # 19..28 scenarios
    Emax = 88.0 + 8.0 * t            # 96..168
    Pmax = round(0.25 * Emax, 3)
    eta = 0.88 + 0.01 * (t % 3)
    E0 = round(0.5 * Emax, 3)
    lam = 46.0 + 9.0 * t             # voided-commitment penalty per unit
    mu = 34.0 + 6.0 * t              # shortfall penalty per unit

    # ---- base weekly price curve: morning + evening peaks, weekend dip ----
    base = []
    for tt in range(T):
        h = tt % 24
        day = tt // 24
        wk = 1.0 - (0.12 if day >= 5 else 0.0)
        v = (28.0 + 18.0 * bump(h, 8, 2.6) + 34.0 * bump(h, 19, 2.6)
             + 4.0 * bump(h, 13, 3.5)) * wk
        base.append(v)

    # ---- scenario prices ----
    prices = []
    for s in range(S):
        row = []
        tilt = rng.uniform(0.9, 1.1)
        for tt in range(T):
            v = base[tt] * tilt * (1.0 + rng.uniform(-0.14, 0.14)) + rng.gauss(0.0, 2.0)
            row.append(max(2.0, round(v, 4)))
        prices.append(row)

    # ---- outage masks ----
    outages = [[0] * T for _ in range(S)]

    def rand_windows(s, nwin, lo=2, hi=5):
        for _ in range(nwin):
            st = rng.randrange(0, T - hi - 1)
            ln = rng.randrange(lo, hi + 1)
            for tt in range(st, min(T, st + ln)):
                outages[s][tt] = 1

    # every scenario gets mild random windows (background unreliability)
    for s in range(S):
        if t <= 3:
            rand_windows(s, 1, lo=2, hi=3)
        else:
            rand_windows(s, rng.randrange(1, 3), lo=2, hi=4)

    meanp = [sum(prices[s][tt] for s in range(S)) / S for tt in range(T)]

    def storm(s, hours):
        for tt in hours:
            outages[s][tt] = 1

    top_hours = sorted(range(T), key=lambda tt: -meanp[tt])
    if t >= 4:
        # storm scenario voids the top mean-price hours (evening-peak heavy)
        storm(0, top_hours[:14 + t])
    if t >= 6:
        # second storm voids the morning peak: hours with h in [7,11] ranked by price
        morn = sorted((tt for tt in range(T) if 7 <= tt % 24 <= 11), key=lambda tt: -meanp[tt])
        storm(1, morn[:10 + t])
    if t == 8 or t >= 10:
        # mirage scenario: inflates the mean price in hours it voids itself
        m = 2 if t == 8 else 3
        mh = []
        for day in (2, 4):
            for h in range(17, 23):
                mh.append(day * 24 + h)
        for tt in mh:
            prices[m][tt] = round(base[tt] * 3.4, 4)
        storm(m, mh)
        # refresh mean prices after the mirage pump
        meanp = [sum(prices[s][tt] for s in range(S)) / S for tt in range(T)]
    if t >= 9:
        # low-floor scenario: depressed prices, no outages -> binds aggressive plans
        lf = S - 1
        for tt in range(T):
            prices[lf][tt] = round(max(2.0, base[tt] * 0.55), 4)
            outages[lf][tt] = 0

    # ---- terminal values ----
    rho = []
    for s in range(S):
        sp = sorted(prices[s])
        med = sp[int(0.45 * (T - 1))]
        rho.append(round(eta * med * rng.uniform(0.70, 0.85), 4))
    if t >= 9:
        rho[S - 1] = round(rho[S - 1] * 1.15, 4)   # low-floor scenario values storage

    return T, S, Emax, Pmax, eta, E0, lam, mu, prices, outages, rho


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    T, S, Emax, Pmax, eta, E0, lam, mu, prices, outages, rho = build(t)
    out = ["%d %d" % (T, S),
           "%r %r %r %r %r %r" % (Emax, Pmax, eta, E0, lam, mu)]
    for s in range(S):
        out.append("%r" % rho[s])
        out.append(" ".join("%r" % v for v in prices[s]))
        out.append(" ".join(str(v) for v in outages[s]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
