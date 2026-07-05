# TIER: greedy
# Terminal log-linear method (standard PK "terminal half-life" estimate):
# fit log C = log(a) - k*t by least squares using only the LATEST half of the
# draws, where the slow elimination phase dominates.  This gets a much better
# terminal slope than a global mono-exponential, but it discards the early
# distribution structure and rests on few, noisy late points -> a decent but
# clearly sub-optimal tail extrapolation.
import sys, math


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    vals = data[2:]
    pts = []
    for i in range(n):
        tt = float(vals[2 * i])
        c = float(vals[2 * i + 1])
        pts.append((tt, math.log(max(c, 1e-9))))
    pts.sort()
    late = pts[len(pts) // 2:]          # latest half of the draws
    ts = [p[0] for p in late]
    ys = [p[1] for p in late]
    n = len(ts)
    # least squares y = c0 + c1*t
    mt = sum(ts) / n
    my = sum(ys) / n
    sxx = sum((x - mt) ** 2 for x in ts)
    sxy = sum((ts[i] - mt) * (ys[i] - my) for i in range(n))
    c1 = sxy / sxx if sxx > 0 else 0.0
    c0 = my - c1 * mt
    a = math.exp(c0)
    k = -c1
    print("0.0 + %r*exp(%r*t)" % (a, -k))


if __name__ == "__main__":
    main()
