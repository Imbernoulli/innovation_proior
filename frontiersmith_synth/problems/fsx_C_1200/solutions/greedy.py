# TIER: greedy
# The obvious first recipe: fit to the OBSERVED (uncensored) exits only --
# customers still active when the window closed are simply dropped, since
# "we don't know their tenure". Bucket by the cohort covariate x (that part
# is "obvious" too) and fit ONE global shape: assume exponential decay
# (hazard rate constant in time, kappa=1 always) with a per-cohort mean
# tenure that varies linearly in x. This looks like sound practice -- you
# only average tenures you actually measured -- but discarding censored
# customers silently over-represents the customers who churned EARLY (the
# longest-lived customers are disproportionately the censored ones), and
# assuming kappa=1 is the wrong SHAPE whenever the true hazard is rising or
# falling. Both errors compound when extrapolating past the visible window.
import sys


def main():
    data = sys.stdin.read().split()
    if len(data) < 3:
        print("0.5"); return
    N = int(data[0])
    vals = data[3:]
    xs, obs, cens = [], [], []
    for i in range(N):
        xs.append(float(vals[3 * i]))
        obs.append(float(vals[3 * i + 1]))
        cens.append(int(vals[3 * i + 2]))

    buckets = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    pooled_uncens = [o for o, c in zip(obs, cens) if c == 0]
    pooled_mean = (sum(pooled_uncens) / len(pooled_uncens)) if pooled_uncens else 10.0

    bx, by = [], []
    for b in buckets:
        vs = [o for xv, o, c in zip(xs, obs, cens) if abs(xv - b) < 1e-6 and c == 0]
        bx.append(b)
        by.append((sum(vs) / len(vs)) if vs else pooled_mean)

    if len(bx) < 2:
        A = by[0] if by else 10.0
        B = 0.0
    else:
        n = len(bx)
        sx = sum(bx); sy = sum(by)
        sxx = sum(v * v for v in bx); sxy = sum(u * v for u, v in zip(bx, by))
        denom = n * sxx - sx * sx
        if abs(denom) < 1e-9:
            A = sy / n; B = 0.0
        else:
            B = (n * sxy - sx * sy) / denom
            A = (sy - B * sx) / n

    print("exp ( - t / ( abs ( %.6f + %.6f * x ) + 1.0 ) )" % (A, B))


if __name__ == "__main__":
    main()
