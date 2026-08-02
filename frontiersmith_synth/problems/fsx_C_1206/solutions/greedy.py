# TIER: greedy
# The obvious recipe: the target is a rate bounded in [0,1] that looks like it
# should transition from near-0 to near-1, so fit the textbook S-curve
# missrate ~ sig(a + b*log(N)) by ordinary least squares on the LOGIT-
# transformed training rows (clip to avoid log(0)), then extrapolate the same
# sigmoid arbitrarily far. This ignores the reuse-distance histogram and the
# public capacity formula entirely -- it tries to read the steepness and
# location of the cliff purely off the small-N log.
#
# The trap: every training row is deep in the sub-cliff regime. For workloads
# with a SHARP transition (large steepness), the true miss rate there is a
# tiny power-law tail that a finite-sample measurement mostly rounds to noise
# near 0 -- the visible log carries almost no information about how steep the
# real transition is, so the fitted slope badly underestimates it and the
# recipe keeps predicting near-zero long past where the true curve has
# already risen to nearly 1. (For gentler workloads the same recipe can
# accidentally do fine -- the point is it has no way to tell the two cases
# apart without the histogram.)
import sys, math


def main():
    data = sys.stdin.read().split()
    if not data:
        print("0.0"); return
    t, C, A, M, n_train = (int(x) for x in data[:5])
    idx = 5 + M
    EPS = 1e-4
    xs, ys = [], []
    for i in range(n_train):
        N = float(data[idx + 2 * i])
        mr = float(data[idx + 2 * i + 1])
        c = min(max(mr, EPS), 1.0 - EPS)
        xs.append(math.log(N))
        ys.append(math.log(c / (1.0 - c)))
    nrow = len(xs)
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    den = nrow * sxx - sx * sx
    if abs(den) < 1e-12:
        b = 0.0
        a = sy / nrow
    else:
        b = (nrow * sxy - sx * sy) / den
        a = (sy - b * sx) / nrow
    print("sig ( %.10g + %.10g * log ( n ) )" % (a, b))


if __name__ == "__main__":
    main()
