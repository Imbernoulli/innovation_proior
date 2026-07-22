# TIER: greedy
"""The obvious first move: reach for ordinary least squares. Fit a memoryless
affine model y = a*x + b to the whole training sweep (no state, no notion of
"which side of the backlash band is engaged"). Since the training sweep is
dominated by ONE direction, this looks like an excellent fit on the logbook
(the fixed lag -r*D or +r*D just becomes the intercept). But the emitted
program has NO state at all: on the fast held-out drive, roughly half the
samples are on the OPPOSITE direction from calibration, where the true
output differs from this fixed line by about 2*r*D -- a big, systematic,
uncorrected error with every reversal."""
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    vals = list(map(float, data[2:2 + 2 * n]))
    xs = vals[0::2]
    ys = vals[1::2]

    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    a = sxy / sxx if sxx > 1e-12 else 0.0
    b = my - a * mx

    print("OUT %.6f * x + %.6f" % (a, b))


if __name__ == "__main__":
    main()
