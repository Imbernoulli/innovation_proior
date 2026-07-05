# TIER: strong
# Sparse harmonic recovery by iterated matching pursuit: subtract the train
# mean, then repeatedly find the tidal-band frequency whose cos/sin
# least-squares fit removes the most residual energy, subtract it, and continue
# until the marginal gain falls below a threshold (adaptive number of
# constituents, capped at 5).  Recovers the hidden Fourier-sparse law so it
# extrapolates the future window far better than a single harmonic, but the
# short window cannot perfectly resolve the off-grid frequencies and the
# instrument-noise floor leaves residual headroom below 1.0.
import sys, math

TWO_PI = 2.0 * math.pi


def fit_freq(ts, r, w):
    n = len(ts)
    Scc = Sss = Scs = Rc = Rs = 0.0
    for i in range(n):
        c = math.cos(w * ts[i]); s = math.sin(w * ts[i])
        Scc += c * c; Sss += s * s; Scs += c * s
        Rc += r[i] * c; Rs += r[i] * s
    det = Scc * Sss - Scs * Scs
    if abs(det) < 1e-9:
        return 0.0, 0.0, -1.0
    c1 = (Rc * Sss - Rs * Scs) / det
    c2 = (Rs * Scc - Rc * Scs) / det
    red = c1 * Rc + c2 * Rs
    return c1, c2, red


def best_harmonic(ts, r):
    best = None
    NP = 1600
    for k in range(NP):
        Pv = 5.0 + (35.0 - 5.0) * k / (NP - 1)
        w = TWO_PI / Pv
        c1, c2, red = fit_freq(ts, r, w)
        if best is None or red > best[0]:
            best = (red, w, c1, c2)
    # local refinement around the grid winner
    _, w0, _, _ = best
    dw = (TWO_PI / 5.0 - TWO_PI / 35.0) / (NP - 1)
    for _ in range(3):
        improved = False
        for w in (w0 - dw, w0 + dw):
            c1, c2, red = fit_freq(ts, r, w)
            if red > best[0]:
                best = (red, w, c1, c2); w0 = w; improved = True
        if improved:
            dw *= 0.5
        else:
            dw *= 0.5
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
    init_sse = sum(v * v for v in r)
    terms = []
    for _ in range(4):
        b = best_harmonic(ts, r)
        if b is None:
            break
        red, w, c1, c2 = b
        if red < 0.02 * init_sse:
            break
        A = math.sqrt(c1 * c1 + c2 * c2)
        phi = math.atan2(-c2, c1)
        terms.append((A, w, phi))
        for i in range(len(ts)):
            r[i] -= (c1 * math.cos(w * ts[i]) + c2 * math.sin(w * ts[i]))
    if not terms:
        sys.stdout.write(repr(mean) + "\n")
        return
    parts = [repr(mean)]
    for A, w, phi in terms:
        parts.append("%r*cos(%r*t + %r)" % (A, w, phi))
    sys.stdout.write(" + ".join(parts) + "\n")


if __name__ == "__main__":
    main()
