# TIER: greedy
# Single-resonance Lorentzian approximation:
#     P(f) ~ A / (1 + B*(f - c)^2)
# Locate the peak (c = frequency of max response), then fit A, B by linear
# regression of 1/P on [1, (f-c)^2] over the near-peak points (reciprocal
# linearisation).  This captures the resonance and a 1/f^2 roll-off, but the
# Lorentzian is symmetric in f and mis-sizes both skirts, so it extrapolates
# out-of-band better than a constant yet worse than the true rational form.
import sys


def main():
    data = sys.stdin.read().split("\n")
    n = int(data[0].split()[0])
    F = []; Y = []
    for ln in data[1:1 + n]:
        p = ln.split()
        if len(p) >= 2:
            F.append(float(p[0])); Y.append(float(p[1]))
    # peak location
    ymax = max(Y)
    c = F[Y.index(ymax)]
    # reciprocal linear fit on the strong (near-peak) points
    s0 = s1 = s2 = t0 = t1 = 0.0
    for fx, yv in zip(F, Y):
        if yv > 0.30 * ymax and yv > 1e-9:
            u = (fx - c) ** 2
            r = 1.0 / yv
            s0 += 1.0; s1 += u; s2 += u * u
            t0 += r;   t1 += r * u
    det = s0 * s2 - s1 * s1
    if abs(det) < 1e-12:
        a0 = 1.0 / max(ymax, 1e-9); a1 = 0.0
    else:
        a0 = (t0 * s2 - t1 * s1) / det      # = 1/A
        a1 = (s0 * t1 - s1 * t0) / det      # = B/A
    A = 1.0 / a0 if abs(a0) > 1e-12 else ymax
    B = a1 * A
    if not (B > 0.0):
        B = 1.0
    print("%r / ( 1 + %r * ( f - %r )**2 )" % (A, B, c))


if __name__ == "__main__":
    main()
