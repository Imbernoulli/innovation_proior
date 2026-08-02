# TIER: greedy
"""
The obvious first attempt: throughput visibly climbs with batch size in the
training range, so fit an ordinary least-squares LINE  y = a + b*x  through
the training rows and extrapolate it. This ignores the given hardware/op
constants C, W, F, D entirely -- it only ever looks at (x, y). It fits the
sub-knee ramp very well (that region really is close to a straight line
starting near the origin), but a line never bends over: at the held-out
batch sizes, deep past the true bandwidth ceiling, this keeps predicting
unbounded growth and overshoots the flattened truth by several times.
"""
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    idx += 1  # test id
    idx += 4  # C W F D
    xs, ys = [], []
    for _ in range(n):
        x = float(data[idx]); idx += 1
        y = float(data[idx]); idx += 1
        xs.append(x); ys.append(y)

    m = len(xs)
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = m * sxx - sx * sx
    if abs(denom) < 1e-12:
        a, b = sy / m, 0.0
    else:
        b = (m * sxy - sx * sy) / denom
        a = (sy - b * sx) / m

    print(f"{a!r} + {b!r} * x")


if __name__ == "__main__":
    main()
