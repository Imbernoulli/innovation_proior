# TIER: greedy
# The obvious textbook move: treat the REPORTED counts as if they directly
# were the epidemic curve. Fit a single log-linear trend r ~ exp(a+b*t) by
# ordinary least squares over ALL training rows (day-of-week rhythm and the
# scribe-capacity saturation are treated as "noise" and averaged away, not
# modelled). To extrapolate past the decree, apply the ONE textbook fact it
# knows -- the announced factor f multiplies the growth rate -- to its single
# fitted slope. This conflates the process and the reporting operator: the
# saturation flattens growth late in training (biasing b low) and the
# near-zero quiet-day counts drag the OLS trend around, so the mis-estimated
# slope, rescaled by f, misses badly once the regime actually changes.
import sys, math


def main():
    data = sys.stdin.read().split()
    if len(data) < 3:
        print("1.0"); return
    n = int(data[0])
    rows = data[3:]
    t_list = [float(rows[2 * i]) for i in range(n)]
    r_list = [float(rows[2 * i + 1]) for i in range(n)]
    y_list = [math.log(max(r, 1e-3)) for r in r_list]

    sx = sum(t_list); sy = sum(y_list)
    sxx = sum(x * x for x in t_list); sxy = sum(x * y for x, y in zip(t_list, y_list))
    m = n
    denom = m * sxx - sx * sx
    if abs(denom) < 1e-9:
        b_hat, a_hat = 0.0, sy / m
    else:
        b_hat = (m * sxy - sx * sy) / denom
        a_hat = (sy - b_hat * sx) / m

    # extrapolate: same fitted trend up to n, growth rate scaled by f after n
    print("exp ( %.10f + %.10f * n + %.10f * f * ( t - n ) )" % (a_hat, b_hat, b_hat))


if __name__ == "__main__":
    main()
