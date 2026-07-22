# TIER: greedy
# Obvious approach: the battery has the cheaper round trip (loss-free), so route EVERY parcel
# to it. Walk the buy/sell windows, pair each buy with the next sell, and execute the trade on
# the BATTERY whenever it turns a profit -- ignoring the vault and the leak-crossover horizon.
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
            if prof_b > 0.0:
                uB[tb] = float(powB)
                uB[ts] = -ret
            pending = -1

    out = ["%.12g %.12g" % (uB[t], uV[t]) for t in range(T)]
    sys.stdout.write("\n".join(out) + "\n")


main()
