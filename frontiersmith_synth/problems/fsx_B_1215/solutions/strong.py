# TIER: strong
# The insight: don't build a rule from the eruptive rows alone. Compare EACH
# signal's class-conditional distributions across BOTH outcomes -- eruptions
# AND the more numerous failed intrusions -- using a standardized mean-gap
# (Cohen's d) computed from the FULL catalogue. ACC and SEIS are confounded
# with the real driver only through shared unrest intensity, so once failed
# intrusions are weighed in they show only a modest gap; INFL (the actual
# magma-chamber inflation) separates the two outcomes far more cleanly. Fit a
# single-variable logistic regression on THAT feature by full-data maximum
# likelihood -- its intercept naturally encodes the true (low) base rate of
# eruption, rather than an alert threshold calibrated only on the salient
# positive cases. Because the recovered law tracks the physical driver
# (magma volume) rather than a confounded proxy, it keeps working once the
# held-out bout gets louder across the board.
import sys, math


def sigmoid(x):
    x = max(-60.0, min(60.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def cohend(xs, ys):
    pos = [x for x, y in zip(xs, ys) if y == 1]
    neg = [x for x, y in zip(xs, ys) if y == 0]
    if len(pos) < 2 or len(neg) < 2:
        return 0.0
    mp = sum(pos) / len(pos); mn = sum(neg) / len(neg)
    vp = sum((x - mp) ** 2 for x in pos) / len(pos)
    vn = sum((x - mn) ** 2 for x in neg) / len(neg)
    sp = ((vp + vn) / 2.0) ** 0.5 or 1.0
    return abs(mp - mn) / sp


def logreg_1d(xs, ys, iters=400, lr=0.5):
    n = len(xs)
    mx = sum(xs) / n
    sx = (sum((x - mx) ** 2 for x in xs) / n) ** 0.5 or 1.0
    zs = [(x - mx) / sx for x in xs]
    w = 0.0; b = 0.0
    for _ in range(iters):
        gw = 0.0; gb = 0.0
        for z, y in zip(zs, ys):
            p = sigmoid(w * z + b)
            e = p - y
            gw += e * z; gb += e
        w -= lr * gw / n
        b -= lr * gb / n
    return w, b, mx, sx


def main():
    data = sys.stdin.read().split()
    if not data:
        print("0.5"); return
    n = int(data[0])
    vals = data[2:]
    accs, infls, seiss, ys = [], [], [], []
    for i in range(n):
        accs.append(float(vals[4 * i]))
        infls.append(float(vals[4 * i + 1]))
        seiss.append(float(vals[4 * i + 2]))
        ys.append(int(vals[4 * i + 3]))

    feats = {"ACC": accs, "INFL": infls, "SEIS": seiss}
    ds = {name: cohend(xs, ys) for name, xs in feats.items()}
    best = max(ds, key=ds.get)
    w, b, mx, sx = logreg_1d(feats[best], ys)
    coef = w / sx
    intercept = b - w * mx / sx
    print("sig ( %.10g * %s + %.10g )" % (coef, best, intercept))


if __name__ == "__main__":
    main()
