# TIER: strong
# Insight: route each parcel by its intended HOLD DURATION, not by a static ranking.
# Walk the buy/sell windows, pair each buy with the next sell, and for every pair compare what
# the battery keeps (rB^d, compounding with hold time d) against what the vault keeps (a fixed
# tax etaV). Short holds (d below the crossover d*=ln(etaV)/ln(rB)) go to the battery; long holds
# go to the vault, capturing the wide-spread trades the leak would otherwise destroy.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    T = int(next(it)); rB = float(next(it)); etaV = float(next(it))
    powB = int(next(it)); powV = int(next(it)); capB = int(next(it)); capV = int(next(it))
    price = [int(next(it)) for _ in range(T)]
    canBuy = [int(next(it)) for _ in range(T)]
    canSell = [int(next(it)) for _ in range(T)]

    uB = [0.0] * T; uV = [0.0] * T
    pending = -1
    for t in range(T):
        if canBuy[t]:
            pending = t
        elif canSell[t] and pending >= 0:
            tb = pending; ts = t; d = ts - tb
            ret = powB * (rB ** d)
            prof_b = ret * price[ts] - powB * price[tb]
            prof_v = powV * (etaV * price[ts] - price[tb])
            if prof_b >= prof_v and prof_b > 0.0:
                uB[tb] = float(powB)
                uB[ts] = -ret
            elif prof_v > 0.0:
                uV[tb] = float(powV)
                uV[ts] = -float(powV)
            pending = -1

    out = ["%.12g %.12g" % (uB[t], uV[t]) for t in range(T)]
    sys.stdout.write("\n".join(out) + "\n")


main()
