# TIER: strong
# The insight: the set of RATIOS reachable by any non-negative purchase combination is
# well APPROXIMATED by the convex hull, on the product simplex, of the raw types' own
# normalized FULL-yield directions (buying more of a single raw type stays at/near its
# own direction as long as it runs within its full-yield threshold; blending two raw
# types traces close to the segment between their two directions -- degradation past a
# threshold only nudges this by small integer per-product flooring). So:
#   1. Compute each raw type's own output direction dir[j] = yield[j] / sum(yield[j]).
#   2. For every PAIR of raw types (i, j) -- including a raw type paired with itself,
#      i.e. a single source -- find the point on segment [dir[i], dir[j]] closest to
#      the target under L1 distance. Because L1 distance is piecewise-linear along the
#      segment, the minimum is attained at one of its endpoints or at a breakpoint
#      where some coordinate of the interpolated point crosses the target's coordinate
#      -- both are cheap to enumerate exactly.
#   3. The best pair over the whole search is a good PROXY for the true reachable
#      ratio (exact when the chosen batch counts stay within threshold; close when
#      they don't) -- either close to the target (if it lies on/near some segment) or
#      close to the true nearest point on the reachable region's boundary.
#   4. Only THEN decide integer batch counts for the two chosen raw types: sweep whole
#      batch counts for each (bounded by their purchase caps) and directly simulate the
#      REAL admitted output -- respecting lot integrality and each raw type's full-yield
#      threshold/degradation -- picking whichever integer pair minimizes the true
#      dev + GAMMA*(spend/spend_cap) objective. This is what correctly trades off
#      "buy more of the cheap-but-degrading source" against "top up with a pricier,
#      non-degrading one" instead of assuming continuous full-rate yield forever.
import sys, json

GAMMA = 0.5


def simulate(P, lot, yld, thresh, dnum, dden, cost, order):
    output = [0, 0, 0]
    spend = 0.0
    for j in range(P):
        oj = order[j]
        nb = oj // lot[j]
        spend += oj * cost[j]
        th = thresh[j]
        full_b = min(nb, th)
        y = yld[j]
        for k in range(3):
            output[k] += full_b * y[k]
        deg_b = max(0, nb - th)
        if deg_b:
            dn, dd = dnum[j], dden[j]
            for k in range(3):
                output[k] += deg_b * ((y[k] * dn) // dd)
    return output, spend


def objective(output, spend, spend_cap, target):
    tot = sum(output)
    if tot <= 0:
        dev = 2.0
    else:
        ratio = [x / tot for x in output]
        dev = sum(abs(ratio[k] - target[k]) for k in range(3))
    return dev + GAMMA * (spend / spend_cap)


def norm(v):
    s = sum(v)
    if s <= 0:
        return [0.0, 0.0, 0.0]
    return [x / s for x in v]


def main():
    inst = json.load(sys.stdin)
    P = inst["P"]
    lot = inst["lot"]
    yld = inst["yield"]
    cost = inst["cost"]
    thresh = inst["thresh"]
    dnum = inst["degrade_num"]
    dden = inst["degrade_den"]
    maxOrder = inst["maxOrder"]
    target = inst["target"]
    spend_cap = sum(maxOrder[j] * cost[j] for j in range(P))

    dirs = [norm(yld[j]) for j in range(P)]

    best = None  # (dev, i, j, lam)
    for i in range(P):
        for j in range(i, P):
            di, dj = dirs[i], dirs[j]
            lambdas = {0.0, 1.0}
            for k in range(3):
                denom = di[k] - dj[k]
                if abs(denom) > 1e-12:
                    lam = (target[k] - dj[k]) / denom
                    if 0.0 <= lam <= 1.0:
                        lambdas.add(lam)
            for lam in lambdas:
                pt = [lam * di[k] + (1 - lam) * dj[k] for k in range(3)]
                dev = sum(abs(pt[k] - target[k]) for k in range(3))
                if best is None or dev < best[0] - 1e-12:
                    best = (dev, i, j, lam)

    _, bi, bj, _ = best

    order = [0] * P
    lot_i = lot[bi]
    cap_i = maxOrder[bi] // lot_i
    best_obj = None
    best_order = None

    if bi == bj:
        for nb in range(cap_i + 1):
            cand = [0] * P
            cand[bi] = nb * lot_i
            output, spend = simulate(P, lot, yld, thresh, dnum, dden, cost, cand)
            obj = objective(output, spend, spend_cap, target)
            if best_obj is None or obj < best_obj:
                best_obj, best_order = obj, cand
    else:
        lot_j = lot[bj]
        cap_j = maxOrder[bj] // lot_j
        for nbi in range(cap_i + 1):
            for nbj in range(cap_j + 1):
                cand = [0] * P
                cand[bi] = nbi * lot_i
                cand[bj] = nbj * lot_j
                output, spend = simulate(P, lot, yld, thresh, dnum, dden, cost, cand)
                obj = objective(output, spend, spend_cap, target)
                if best_obj is None or obj < best_obj:
                    best_obj, best_order = obj, cand

    order = best_order if best_order is not None else order
    print(json.dumps({"order": order}))


if __name__ == "__main__":
    main()
