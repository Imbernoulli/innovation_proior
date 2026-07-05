# TIER: greedy
# Fit the SINGLE dominant tidal constituent by a periodogram / matching-pursuit
# scan: remove the train mean, then find the one frequency (over a tidal-band
# grid) whose cos/sin least-squares fit removes the most residual energy.
# Captures the strongest tide but ignores every other constituent, so on gauges
# with 2+ constituents it extrapolates only partially -> beats the constant but
# stays well under a full recovery.
import sys, math

TWO_PI = 2.0 * math.pi


def best_harmonic(ts, r):
    n = len(ts)
    best = None
    P = 6.0
    NP = 1400
    for k in range(NP):
        Pv = 5.0 + (35.0 - 5.0) * k / (NP - 1)
        w = TWO_PI / Pv
        Scc = Sss = Scs = Rc = Rs = 0.0
        for i in range(n):
            c = math.cos(w * ts[i]); s = math.sin(w * ts[i])
            Scc += c * c; Sss += s * s; Scs += c * s
            Rc += r[i] * c; Rs += r[i] * s
        det = Scc * Sss - Scs * Scs
        if abs(det) < 1e-9:
            continue
        c1 = (Rc * Sss - Rs * Scs) / det
        c2 = (Rs * Scc - Rc * Scs) / det
        red = c1 * Rc + c2 * Rs
        if best is None or red > best[0]:
            best = (red, w, c1, c2)
    return best


def main():
    data = sys.stdin.read().split("\n")
    n = int(data[0].split()[0])
    ts = []; ys = []
    for ln in data[1:1 + n]:
        p = ln.split()
        if len(p) >= 2:
            ts.append(float(p[0])); ys.append(float(p[1]))
    mean = sum(ys) / len(ys)
    r = [y - mean for y in ys]
    b = best_harmonic(ts, r)
    if b is None:
        sys.stdout.write(repr(mean) + "\n")
        return
    _, w, c1, c2 = b
    A = math.sqrt(c1 * c1 + c2 * c2)
    phi = math.atan2(-c2, c1)
    sys.stdout.write("%r + %r*cos(%r*t + %r)\n" % (mean, A, w, phi))


if __name__ == "__main__":
    main()
