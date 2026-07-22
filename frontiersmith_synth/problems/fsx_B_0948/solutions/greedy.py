# TIER: greedy
# The obvious recipe: springback-vs-(radius/thickness) power laws are the
# textbook move in sheet-metal forming, so fit ONE global power law
#   S = a * (r/t)^b
# by ordinary least squares in log-log space over the whole training batch.
# This is an excellent fit to the (mostly single-regime) thin-sheet training
# cloud, but a single power law in x=r/t has no way to represent that the
# SAME x value is elastic-dominated on thin sheets and plastic-saturated on
# thick sheets -- it ignores t's separate role in shifting the regime
# boundary, so it extrapolates with the wrong exponent on the held-out
# thick-sheet batch.
import sys, math


def main():
    data = sys.stdin.read().split()
    if not data:
        print("1.2"); return
    n = int(data[0])
    vals = data[2:]
    xs = []
    ys = []
    for i in range(n):
        r = float(vals[3 * i])
        t = float(vals[3 * i + 1])
        s = float(vals[3 * i + 2])
        x = r / t
        if s > 1e-6 and x > 1e-9:
            xs.append(math.log(x))
            ys.append(math.log(s))

    if len(xs) < 2:
        print("1.2"); return

    n2 = len(xs)
    mx = sum(xs) / n2
    my = sum(ys) / n2
    sxx = sum((xv - mx) ** 2 for xv in xs)
    sxy = sum((xv - mx) * (yv - my) for xv, yv in zip(xs, ys))
    b = sxy / sxx if sxx > 1e-12 else 1.0
    loga = my - b * mx
    a = math.exp(loga)

    print("%.6f * ( r / t ) ** %.6f" % (a, b))


if __name__ == "__main__":
    main()
