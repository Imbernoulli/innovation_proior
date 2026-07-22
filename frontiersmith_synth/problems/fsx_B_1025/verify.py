#!/usr/bin/env python3
# Deterministic checker for the feedback-delay-network (FDN) reverb-design problem
# (format C, minimize). CLI: python3 verify.py <in> <out> <ans> (ans ignored).
# Prints "... Ratio: <r>" with r in [0,1]. Any feasibility violation -> Ratio: 0.0.
import sys
import numpy as np

GAIN_MIN = 1e-3
GAIN_MAX = 0.99
STAB_EPS = 1e-6


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def householder(N):
    # M = I - (2/N) * ones(N,N): the fixed feedback matrix (symmetric, orthogonal,
    # eigenvalues -1 [multiplicity 1, eigvec = all-ones] and +1 [multiplicity N-1]).
    M = -(2.0 / N) * np.ones((N, N), dtype=np.float64)
    for i in range(N):
        M[i, i] += 1.0
    return M


def simulate(N, T, L, g, M):
    """Run the FDN for T samples with a unit impulse injected into line 0 at n=0.
    Per-sample recursion (gain applied at the write point of each delay line, i.e.
    every round trip through line i attenuates by g[i]):
        r_i          = buf_i[ptr_i]                      (read: this line's tap)
        fb           = M @ r                              (fixed mixing matrix)
        fb[0]       += impulse (n == 0 only)
        buf_i[ptr_i] = g_i * fb_i                          (write, attenuated)
        ptr_i        = (ptr_i + 1) mod L_i
        y[n]         = mean_i(r_i)                         (output tap)
    """
    bufs = [np.zeros(L[i], dtype=np.float64) for i in range(N)]
    ptr = [0] * N
    y = np.zeros(T, dtype=np.float64)
    r = np.zeros(N, dtype=np.float64)
    for n in range(T):
        for i in range(N):
            r[i] = bufs[i][ptr[i]]
        fb = M.dot(r)
        if n == 0:
            fb[0] += 1.0
        for i in range(N):
            bufs[i][ptr[i]] = g[i] * fb[i]
            ptr[i] += 1
            if ptr[i] == L[i]:
                ptr[i] = 0
        y[n] = r.sum() / N
    return y


def score_curves(y, T, ts, target_db, target_density):
    """Per-segment MEAN POWER decay curve (dB, relative to the first segment's own
    level -- this cancels any arbitrary overall output-amplitude scale, so only decay
    SHAPE is scored), and a per-segment participation-ratio echo-density curve
    (threshold-free: for a segment of samples, N_eff = (sum y^2)^2 / sum y^4 estimates
    the effective number of comparable-magnitude echoes present; density =
    min(1, N_eff / segment_len) is in (0,1] by Cauchy-Schwarz, ~1 only when the
    segment is fully "diffuse" (all samples comparable magnitude), and tiny when
    energy is concentrated in a few spikes with silence between (the sparse/metallic/
    flutter failure mode)."""
    K = len(ts)
    pow_meas = []
    dens_meas = []
    prev_t = 0
    for k in range(K):
        t = ts[k]
        seg = y[prev_t:t]
        L = len(seg)
        if L <= 0:
            pow_meas.append(0.0)
            dens_meas.append(0.0)
        else:
            s2 = float(np.sum(seg * seg))
            s4 = float(np.sum(seg * seg * seg * seg))
            pow_meas.append(s2 / L)
            if s4 < 1e-300:
                neff = 0.0
            else:
                neff = (s2 * s2) / s4
            dens_meas.append(min(1.0, neff / L))
        prev_t = t

    if not np.isfinite(pow_meas[0]) or pow_meas[0] < 1e-15:
        return None, None, "insufficient energy in first segment"

    db_meas = [10.0 * np.log10(max(p, 1e-300) / pow_meas[0]) for p in pow_meas]

    decay_err = float(np.mean(np.abs(np.array(db_meas) - np.array(target_db))))
    dens_err = float(np.mean(np.abs(np.array(dens_meas) - np.array(target_density))))
    return decay_err, dens_err, None


def baseline_construction(N, T, Lmin, Lmax, ts, target_db):
    """Trivial internal baseline: delay lengths are ascending multiples of Lmin
    (guaranteed common factor Lmin => sparse/aliased echo arrivals -- the density
    descriptor is simply not addressed). Gains are a single global value, fit with
    the crude "one exponential through the last checkpoint" recipe -- the decay
    descriptor gets a naive but not needlessly-bad recipe applied. Clamped into
    [Lmin, Lmax]. This is the reference a solver reaches for with zero insight into
    either descriptor beyond the most obvious one-line fit."""
    L = []
    for i in range(N):
        Li = Lmin * (i + 1)
        Li = min(Li, Lmax)
        Li = max(Li, Lmin)
        L.append(Li)
    Lavg = sum(L) / N
    slope = target_db[-1] / max(1, ts[-1])
    g0 = 10.0 ** (slope * Lavg / 20.0)
    g0 = min(GAIN_MAX, max(GAIN_MIN, g0))
    g = [g0] * N
    return L, g


def main():
    try:
        itoks = open(sys.argv[1]).read().split()
        p = 0
        N = int(itoks[p]); p += 1
        T = int(itoks[p]); p += 1
        Lmin = int(itoks[p]); p += 1
        Lmax = int(itoks[p]); p += 1
        K = int(itoks[p]); p += 1
        ts = [int(itoks[p + j]) for j in range(K)]; p += K
        target_db = [float(itoks[p + j]) for j in range(K)]; p += K
        target_density = [float(itoks[p + j]) for j in range(K)]; p += K
        w_decay = float(itoks[p]); p += 1
        w_density = float(itoks[p]); p += 1
    except Exception:
        fail("bad instance")

    try:
        otoks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")

    need = 2 * N
    if len(otoks) != need:
        fail("expected exactly %d numbers (%d lengths + %d gains), got %d" % (need, N, N, len(otoks)))

    L = []
    for i in range(N):
        try:
            v = float(otoks[i])
        except Exception:
            fail("bad delay length token %d" % i)
        if not np.isfinite(v) or v != round(v):
            fail("delay length %d not a finite integer" % i)
        Li = int(round(v))
        if Li < Lmin or Li > Lmax:
            fail("delay length %d = %d outside [%d,%d]" % (i, Li, Lmin, Lmax))
        L.append(Li)

    g = []
    for i in range(N):
        try:
            v = float(otoks[N + i])
        except Exception:
            fail("bad gain token %d" % i)
        if not np.isfinite(v):
            fail("non-finite gain %d" % i)
        if v < GAIN_MIN or v > GAIN_MAX:
            fail("gain %d = %g outside [%g,%g]" % (i, v, GAIN_MIN, GAIN_MAX))
        g.append(v)

    M = householder(N)
    Gm = np.diag(g).dot(M)
    try:
        eig = np.linalg.eigvals(Gm)
    except Exception:
        fail("eigendecomposition failed")
    rad = float(np.max(np.abs(eig)))
    if rad >= 1.0 - STAB_EPS:
        fail("unstable: spectral radius %.6f >= 1" % rad)

    y = simulate(N, T, L, g, M)
    decay_err, dens_err, why = score_curves(y, T, ts, target_db, target_density)
    if why is not None:
        fail(why)

    F = w_decay * decay_err + w_density * dens_err

    Lb, gb = baseline_construction(N, T, Lmin, Lmax, ts, target_db)
    yb = simulate(N, T, Lb, gb, M)
    decay_err_b, dens_err_b, why_b = score_curves(yb, T, ts, target_db, target_density)
    if why_b is not None:
        # Should not happen for the fixed baseline construction; guard anyway.
        B = 1.0
    else:
        B = w_decay * decay_err_b + w_density * dens_err_b

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%.6f B=%.6f decay_err=%.6f dens_err=%.6f Ratio: %.6f" %
          (F, B, decay_err, dens_err, sc / 1000.0))


if __name__ == "__main__":
    main()
