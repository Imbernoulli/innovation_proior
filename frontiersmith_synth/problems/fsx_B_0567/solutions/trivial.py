# TIER: trivial
# Reproduces the checker baseline: execute only the single best battery-only round trip.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    T = int(next(it)); rB = float(next(it)); etaV = float(next(it))
    powB = int(next(it)); powV = int(next(it)); capB = int(next(it)); capV = int(next(it))
    price = [int(next(it)) for _ in range(T)]
    canBuy = [int(next(it)) for _ in range(T)]
    canSell = [int(next(it)) for _ in range(T)]

    buys = [t for t in range(T) if canBuy[t]]
    sells = [t for t in range(T) if canSell[t]]
    pw = [1.0] * T
    for d in range(1, T):
        pw[d] = pw[d - 1] * rB
    best = 0.0; bt1 = -1; bt2 = -1
    for t1 in buys:
        p1 = price[t1]
        for t2 in sells:
            if t2 <= t1:
                continue
            prof = powB * (pw[t2 - t1] * price[t2] - p1)
            if prof > best:
                best = prof; bt1 = t1; bt2 = t2

    uB = [0.0] * T; uV = [0.0] * T
    if bt1 >= 0:
        uB[bt1] = float(powB)
        ret = float(powB)
        for _ in range(bt2 - bt1):
            ret *= rB
        uB[bt2] = -ret

    out = ["%.12g %.12g" % (uB[t], uV[t]) for t in range(T)]
    sys.stdout.write("\n".join(out) + "\n")


main()
