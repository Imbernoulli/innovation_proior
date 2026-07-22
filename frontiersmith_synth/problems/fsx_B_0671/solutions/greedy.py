# TIER: greedy
"""
Greedy / textbook approach: fit a single power law directly to the raw
readings, as if they were unaliased measurements.

x -> reading is treated as an ordinary noisy power-law sample and fit with
log-log ordinary least squares: log(reading) = log(A') + p'*log(x). This
ignores the fact that every reading is a Nyquist FOLD of the true frequency
into [0, fs/2]. Where the sweep never crosses a fold this looks fine, but
once the true frequency starts wrapping the folded readings stop tracking
x monotonically (a zigzag), pulling the fitted exponent toward a much
SHALLOWER apparent growth rate than the true law -- and because every
reading is capped near fs/2, the fitted law badly UNDER-predicts the much
larger true frequencies on the held-out (higher drive-level) extrapolation
grid.
"""
import sys, math


def ols(xs, ys):
    n = len(xs)
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        slope = 0.0
    else:
        slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return slope, intercept


def main():
    data = sys.stdin.read().split()
    idx = 0
    t = int(data[idx]); idx += 1
    n = int(data[idx]); idx += 1
    fs = float(data[idx]); idx += 1
    fmax = float(data[idx]); idx += 1
    xs_raw, rs_raw = [], []
    for _ in range(n):
        x = float(data[idx]); idx += 1
        r = float(data[idx]); idx += 1
        xs_raw.append(x); rs_raw.append(max(1e-6, r))

    xs = [math.log(v) for v in xs_raw]
    ys = [math.log(v) for v in rs_raw]
    p, log_A = ols(xs, ys)
    A = math.exp(log_A)
    p = max(0.0, p)  # a real coder would at least sanity-clip an obviously
                      # wrong-signed (decaying) fit for a "speed law" -- but
                      # does not suspect the deeper aliasing structure

    print("%.10g * powv(x, %.10g)" % (A, p))


if __name__ == "__main__":
    main()
