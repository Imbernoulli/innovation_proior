# TIER: greedy
# The domain-obvious recipe: bid shading is naturally read as a RATIO
# price/v that depends on the rival count n. So average that ratio within
# each observed n in {2..6}, fit a plain straight line ratio ~= a + b*n
# through those 5 points (a sane coder then clamps the ratio to the physically
# plausible [0,1] range so it can never predict a price outside [0, v]), and
# multiply by v for any (n, v).
#
# This nails the quiet-day training range (it IS the empirical ratio there,
# clamping never triggers for n in 2..6), but it has two blind spots baked in
# by construction:
#   (1) the straight line was never told the shading law must reach EXACTLY
#       v as n -> infinity -- it only knows "cap at 1". Extrapolated to
#       n=25/60 the fitted line saturates at whatever level the n=2..6 trend
#       happened to hit the [0,1] clamp, which is rarely the true asymptote,
#       so it is systematically off in the high-competition regime;
#   (2) the model is purely MULTIPLICATIVE in v (ratio * v), so it cannot
#       express an ADDITIVE value-curvature term at all -- it silently
#       assumes the appraisal-risk surcharge doesn't exist, so it is blind
#       to the extreme-appraisal held-out rows regardless of n.
import sys


def main():
    data = sys.stdin.read().split()
    if not data:
        print("v")
        return
    num_rows = int(data[0])
    vals = data[3:]
    rows = []
    for i in range(num_rows):
        n = int(float(vals[3 * i]))
        v = float(vals[3 * i + 1])
        p = float(vals[3 * i + 2])
        rows.append((n, v, p))

    by_n = {}
    for n, v, p in rows:
        by_n.setdefault(n, []).append(p / v if v != 0 else 0.0)

    ns = sorted(by_n)
    ratios = [sum(by_n[n]) / len(by_n[n]) for n in ns]

    # plain 1-D least squares line: ratio ~= a + b*n
    m = len(ns)
    if m < 2:
        a, b = (ratios[0] if ratios else 0.5), 0.0
    else:
        mean_n = sum(ns) / m
        mean_r = sum(ratios) / m
        num = sum((n - mean_n) * (r - mean_r) for n, r in zip(ns, ratios))
        den = sum((n - mean_n) ** 2 for n in ns)
        b = num / den if den > 1e-12 else 0.0
        a = mean_r - b * mean_n

    print("max(0, min(1, %.10g + %.10g * n)) * v" % (a, b))


if __name__ == "__main__":
    main()
