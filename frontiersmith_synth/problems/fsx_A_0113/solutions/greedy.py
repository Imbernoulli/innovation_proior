# TIER: greedy
"""Weight-proportional treemap: tile the service region into n rectangles whose
AREAS are proportional to the weights w_i, then inscribe the largest disk in each
rectangle (radius = min(width,height)/2, centred). Higher-load substations get
larger cells and thus larger coverage disks -- beating the weight-blind grid on
skewed instances. (Area ~ w is a reasonable but sub-optimal allocation.)"""
import sys


def partition(x0, y0, w, h, items, out):
    if len(items) == 1:
        idx, _a = items[0]
        r = min(w, h) / 2.0
        out[idx] = (x0 + w / 2.0, y0 + h / 2.0, r)
        return
    items = sorted(items, key=lambda t: -t[1])
    total = sum(a for _, a in items)
    half = total / 2.0
    acc = 0.0
    cut = 0
    for i, (_, a) in enumerate(items):
        acc += a
        cut = i + 1
        if acc >= half:
            break
    g1, g2 = items[:cut], items[cut:]
    if not g2:
        g2 = [g1.pop()]
    a1 = sum(a for _, a in g1)
    a2 = sum(a for _, a in g2)
    if w >= h:
        w1 = w * a1 / (a1 + a2)
        partition(x0, y0, w1, h, g1, out)
        partition(x0 + w1, y0, w - w1, h, g2, out)
    else:
        h1 = h * a1 / (a1 + a2)
        partition(x0, y0, w, h1, g1, out)
        partition(x0, y0 + h1, w, h - h1, g2, out)


def area_of(weight):
    return float(weight)          # area proportional to weight


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    weights = [float(toks[1 + i]) for i in range(n)]
    areas = [area_of(w) for w in weights]
    s = sum(areas)
    items = [(i, areas[i] / s) for i in range(n)]   # normalise to unit area
    out = [None] * n
    partition(0.0, 0.0, 1.0, 1.0, items, out)
    lines = ["%.9f %.9f %.9f" % out[i] for i in range(n)]
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
