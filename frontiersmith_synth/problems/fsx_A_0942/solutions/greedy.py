# TIER: greedy
# The obvious recipe: treat every day's net level change as if it were pure
# decay (ignore the daily charge total Q_d entirely -- "surely the fitted
# curve can absorb whatever noise is in there"). Keep only days where the
# level net-decreased, take log(level) vs log(apparent rate) and fit a plain
# power law by ordinary least squares. This looks reasonable and even nails
# a handful of easy, low-charge instances -- but on charge-heavy instances
# the unmodelled charges masquerade as "unusually small decay", dragging the
# fitted exponent (and constant) far from the truth, which then blows up
# under extrapolation to the held-out high-level regime.
import sys, math

T_DAY = 1.0


def main():
    data = sys.stdin.read().split()
    if len(data) < 2:
        print("0.001*L")
        return
    D = int(data[0])
    L0 = float(data[2])
    rows = []
    idx = 3
    for _ in range(D):
        Q = float(data[idx]); E = float(data[idx + 1]); idx += 2
        rows.append((Q, E))

    xs, ys = [], []
    E_prev = L0
    for Q, E in rows:
        raw = (E_prev - E) / T_DAY
        Lref = (E_prev + E) / 2.0
        if raw > 1e-6 and Lref > 1e-6:
            xs.append(math.log(Lref))
            ys.append(math.log(raw))
        E_prev = E

    if len(xs) < 3:
        print("0.001*L")
        return

    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx < 1e-9:
        slope, intercept = 0.0, my
    else:
        slope = sxy / sxx
        intercept = my - slope * mx

    c = math.exp(intercept)
    print("%.8f * L ** %.6f" % (c, slope))


if __name__ == "__main__":
    main()
