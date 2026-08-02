# TIER: greedy
# The obvious recipe: engagement looks like it is trending over time, so
# treat this as a plain popularity forecast -- fit engagement as a linear
# function of the period alone, e ~ a + b*t, by ordinary least squares on
# the visible log, and extrapolate the SAME line arbitrarily far. This
# ignores the exposure column entirely: it reads "is this item rising" off
# raw engagement, exactly the naive read a recommender's own dashboard would
# invite.
#
# The trap: across the logged window the recommender was ALSO reacting to
# the item's own recent engagement (an exposure-feedback loop), so exposure
# x(t) drifted upward together with t. The fitted slope b is therefore a
# blend of the item's genuine organic drift AND the loop's self-reinforcing
# amplification -- and extrapolating that blended, inflated slope keeps
# compounding long past the point (in the held-out period) where the
# recommender's adaptive policy is switched off and exposure is set by
# intervention instead. Depending on how strong the loop was, this recipe
# can wildly overshoot (or, if exposure happens to fall in the held-out
# period, undershoot) -- it has no way to tell genuine growth from
# recommender-induced growth apart, because it never looks at exposure.
import sys


def main():
    data = sys.stdin.read().split()
    if not data:
        print("0.0"); return
    tid, n_train = int(data[0]), int(data[1])
    idx = 2
    ts, es = [], []
    for i in range(n_train):
        tt = float(data[idx + 3 * i])
        e = float(data[idx + 3 * i + 2])
        ts.append(tt); es.append(e)

    n = len(ts)
    St = sum(ts); Se = sum(es)
    Stt = sum(v * v for v in ts)
    Ste = sum(u * v for u, v in zip(ts, es))
    den = n * Stt - St * St
    if abs(den) < 1e-9:
        b = 0.0
        a = Se / n
    else:
        b = (n * Ste - St * Se) / den
        a = (Se - b * St) / n
    print("%.10g + %.10g * t" % (a, b))


if __name__ == "__main__":
    main()
