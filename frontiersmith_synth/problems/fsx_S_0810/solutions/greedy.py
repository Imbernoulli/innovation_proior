# TIER: greedy
# The obvious recipe: this LOOKS like exponential growth, so just fit an
# exponential (log-linear regression of word length vs rewrite index) on the
# training window and build a toy morphism that reproduces that one scalar
# rate -- one letter self-repeats Lhat times, the other two letters are left
# as identity (irrelevant once the machine only ever emits the self-repeating
# letter). This nails the SHAPE (a growing word) but:
#   (1) the fitted rate is biased by sub-dominant-eigenvalue transients on the
#       short training window, so even the length prediction drifts at long
#       horizons, and
#   (2) the letter-frequency mix is simply wrong (collapsed to one letter),
#       since curve-fitting the length says nothing about the eigenvector.
import sys, math


def main():
    data = sys.stdin.read().split()
    if len(data) < 2:
        print("0"); print("1"); print("2"); return
    t = int(data[0]); nt = int(data[1])
    levels = data[2:2 + nt + 1]
    if len(levels) < 2:
        print("0"); print("1"); print("2"); return

    ys = [math.log(len(w)) for w in levels]
    xs = list(range(len(ys)))
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    b = num / den if den > 1e-12 else 0.0
    r_hat = math.exp(b)
    Lhat = max(1, min(12, round(r_hat)))

    print("0" * Lhat)
    print("1")
    print("2")


if __name__ == "__main__":
    main()
