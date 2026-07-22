import sys, math, numpy as np

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def synth_env(freqs, amps, phis, N, W):
    """Synthesize the superposition and extract its sliding-RMS envelope."""
    n = np.arange(N)
    s = np.zeros(N)
    for f, a, p in zip(freqs, amps, phis):
        s += a * np.cos(2 * np.pi * f * n / N + p)
    s2 = s * s
    half = (W - 1) // 2
    acc = np.zeros(N)
    for d in range(-half, half + 1):
        acc += np.roll(s2, -d)          # acc[n] = sum_{d} s2[(n+d) mod N]
    power = acc / W
    power = np.clip(power, 0.0, None)
    return np.sqrt(power)

def main():
    inp = open(sys.argv[1]).read().split()
    out = open(sys.argv[2]).read().split()

    # ---- parse instance ----
    try:
        it = iter(inp)
        N = int(next(it)); W = int(next(it))
        F_min = int(next(it)); F_max = int(next(it))
        k = int(next(it)); na = int(next(it))
        allowed = [float(next(it)) for _ in range(na)]
        E = np.array([float(next(it)) for _ in range(N)], dtype=float)
    except Exception:
        fail("bad input")

    # ---- internal baseline B: best flat (constant) envelope = intrinsic variation of E ----
    B = float(np.sqrt(np.mean((E - E.mean()) ** 2)))
    B = max(B, 1e-9)

    # ---- parse participant output: up to k triples (freq amp phase) ----
    if len(out) == 0 or len(out) % 3 != 0:
        fail("output must be a multiple of 3 tokens")
    m = len(out) // 3
    if m < 1 or m > k:
        fail("need 1..%d oscillators, got %d" % (k, m))

    freqs, amps, phis = [], [], []
    for j in range(m):
        fs, as_, ps = out[3 * j], out[3 * j + 1], out[3 * j + 2]
        try:
            fv = float(fs); av = float(as_); pv = float(ps)
        except Exception:
            fail("non-numeric triple")
        if not (math.isfinite(fv) and math.isfinite(av) and math.isfinite(pv)):
            fail("non-finite triple")
        if abs(fv - round(fv)) > 1e-6:
            fail("frequency must be integer")
        fi = int(round(fv))
        if fi < F_min or fi > F_max:
            fail("frequency %d out of band [%d,%d]" % (fi, F_min, F_max))
        if min(abs(av - a) for a in allowed) > 1e-6:
            fail("amplitude %g not in allowed set" % av)
        freqs.append(fi); amps.append(av); phis.append(pv)

    # ---- score: L2 error between synthesized envelope and target ----
    env = synth_env(freqs, amps, phis, N, W)
    if not np.all(np.isfinite(env)):
        fail("non-finite envelope")
    F = float(np.sqrt(np.mean((env - E) ** 2)))     # objective (minimize)
    F = max(F, 1e-9)

    sc = min(1000.0, 100.0 * B / F)                 # trivial(flat)->0.1 ; 10x-better caps at 1
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))

if __name__ == "__main__":
    main()
