# TIER: greedy
# The obvious recipe: estimate ONE average rate of change over the whole
# visible log by ordinary least squares (displacement vs elapsed time only --
# temperature is never used), then extrapolate that rate LINEARLY forward,
# anchored at the most recently logged reading (a very common real-world
# forecasting habit: "current value + historical average rate * time
# elapsed"). This never separates the reversible thermal wobble from the
# irreversible settlement trend -- both are baked into a single anchor value
# and a single slope. On logs whose visible window happens to end at a
# seasonal temperature peak, the anchor reading is inflated by the FULL
# reversible thermal swing; the naive method has no way to know that swing
# will reverse, so the offset rides unchanged across every prediction made
# several seasons later.
import sys


def main():
    data = sys.stdin.read().split()
    if not data:
        print("0.0"); return
    n = int(data[0])
    vals = data[2:]
    rows = []
    for i in range(n):
        tt = float(vals[3 * i])
        d = float(vals[3 * i + 2])
        rows.append((tt, d))

    sx = sum(r[0] for r in rows)
    sy = sum(r[1] for r in rows)
    sxx = sum(r[0] * r[0] for r in rows)
    sxy = sum(r[0] * r[1] for r in rows)
    denom = n * sxx - sx * sx
    b = (n * sxy - sx * sy) / denom if abs(denom) > 1e-9 else 0.0

    t_last, d_last = max(rows, key=lambda r: r[0])

    # d_hat(t) = d_last + b * (t - t_last)
    print("%.10g + %.10g * (t - %.10g)" % (d_last, b, t_last))


if __name__ == "__main__":
    main()
