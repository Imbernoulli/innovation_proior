# TIER: greedy
# The obvious recipe: assume the heater is a MEMORYLESS static curve of the
# current drive temperature.  Fit a logistic threshold y = off + amp*sig((thr-d)*g)
# by picking the threshold that minimises training error.  This is near-perfect
# on the slow quasi-static training branch, but it can represent only ONE heater
# value per drive value -- so it cannot see the hysteresis band's two branches
# and it ignores the actuation delay entirely.  Numbers are emitted as separate
# tokens (spaced) as the DSL expects.
import sys, math


def main():
    data = sys.stdin.read().split()
    if not data:
        print("OUT 0.5"); return
    n = int(data[0])
    vals = data[2:]
    drive = []
    y = []
    for i in range(n):
        drive.append(float(vals[2 * i]))
        y.append(float(vals[2 * i + 1]))

    ys = sorted(y)
    off = ys[int(0.05 * len(ys))]
    hi = ys[int(0.95 * len(ys))]
    amp = max(1e-6, hi - off)

    best = None
    bthr = 0.5
    for gi in range(20, 80):
        thr = gi / 100.0
        e = 0.0
        for d, yv in zip(drive, y):
            p = off + amp / (1.0 + math.exp(-(thr - d) * 10.0))
            e += (p - yv) ** 2
        if best is None or e < best:
            best = e
            bthr = thr

    print("OUT %.6f + %.6f * sig ( ( %.6f - d ) * 10.0 )" % (off, amp, bthr))


if __name__ == "__main__":
    main()
