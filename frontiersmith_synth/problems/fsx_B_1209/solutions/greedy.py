# TIER: greedy
# The obvious recipe: this is a forecasting problem, so fit a "temperature-
# based regression" of the observed ground index G against the visible
# history and persist it -- a plain trend/level fit sees only noise around a
# constant (because the plateau buffers the forcing entirely), so the honest
# best fit IS a flat continuation. This nails the (common) case where the
# station never leaves the plateau within the graded window, but has no way
# to see an energy accumulator building toward the hidden capacity -- so on
# stations whose graded window crosses into thaw, it confidently predicts
# continued stability and gets left far behind.
import sys


def main():
    data = sys.stdin.read().split()
    if not data:
        print("OUT 0.0"); return
    n = int(data[0])
    vals = data[2:]
    g = [float(vals[2 * i + 1]) for i in range(n)]
    if not g:
        print("OUT 0.0"); return
    mean_g = sum(g) / len(g)
    print("OUT %.6f" % mean_g)


if __name__ == "__main__":
    main()
