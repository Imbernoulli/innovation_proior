# TIER: greedy
# Naive pure power law y = A * x**(-al) fitted by ordinary least squares in
# log-log space. Captures the decay trend but assumes NO irreducible floor, so it
# undershoots in the extrapolation region -> intermediate score.
import sys, math


def linfit(xs, ys):
    n = len(xs)
    sx = sum(xs); sy = sum(ys)
    sxx = sum(v * v for v in xs); sxy = sum(a * b for a, b in zip(xs, ys))
    d = n * sxx - sx * sx
    m = (n * sxy - sx * sy) / d
    c = (sy - m * sx) / n
    return m, c


def main():
    toks = sys.stdin.read().split()
    m = int(toks[1])
    xs = []; ys = []
    idx = 2
    for _ in range(m):
        xs.append(float(toks[idx])); ys.append(float(toks[idx + 1]))
        idx += 2
    lx = [math.log(v) for v in xs]
    ly = [math.log(v) for v in ys]
    slope, inter = linfit(lx, ly)
    A = math.exp(inter)
    al = -slope
    sys.stdout.write("%.8g * x ** %.8g\n" % (A, -al))


if __name__ == "__main__":
    main()
