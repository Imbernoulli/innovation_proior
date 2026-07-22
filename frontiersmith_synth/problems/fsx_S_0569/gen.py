import sys, math, random
import numpy as np

# fir-cascade-precompensation:
#   A fixed cascade of K FIR filters (given). Craft an amplitude-bounded input pulse x
#   so that filtering x through the whole chain matches a target waveform y* while
#   spending little input energy.  J = ||conv(x,h)-y*||^2 + lam*||x||^2.
#
# The cascade planted here has NON-MINIMUM-PHASE zeros (radius > 1) and near-unit-circle
# resonances (radius ~0.95-0.98).  The textbook move -- causal inverse filtering
# (polynomial long division) -- runs an unstable recursion and blows up; the amplitude
# bound then forces a hard clip that wrecks both the match and the energy budget.
# The insight is regularized frequency-domain shaping (Tikhonov / normal-equations),
# which trades match against energy instead of inverting exactly.


def real_root(a):
    # first-order FIR with a single real zero at z = a
    return np.array([1.0, -a], dtype=np.float64)


def resonator(r, theta):
    # second-order FIR with conjugate zeros at r*e^{+-i theta}
    return np.array([1.0, -2.0 * r * math.cos(theta), r * r], dtype=np.float64)


def cascade(filters):
    h = np.array([1.0], dtype=np.float64)
    for f in filters:
        h = np.convolve(h, f)
    return h


def main():
    i = int(sys.argv[1])
    rng = random.Random(56900 + 977 * i)
    npr = np.random.RandomState(15690 + 131 * i)

    # ----- difficulty ladder -----
    N = [56, 72, 88, 104, 120, 136, 152, 168, 184, 196][i - 1]
    A = 1.0
    lam = 0.05

    filters = []
    # every cascade has a near-unit-circle resonance (ill-conditioning) + a mild smoother
    r1 = 0.86 + 0.012 * (i - 1)          # 0.86 .. 0.968
    th1 = 0.35 + 0.05 * (i % 4)
    filters.append(resonator(r1, th1))
    filters.append(real_root(0.55))       # stable smoother, zero inside

    hard = i >= 6                          # cases 6..10 are the deconvolution traps (5 of 10)
    if hard:
        # non-minimum-phase zero(s): |zero| > 1 -> causal inverse recursion diverges
        a_nmp = 1.03 + 0.013 * (i - 6)     # 1.03 .. 1.082
        filters.append(real_root(a_nmp))
        # a tighter resonance sharpens the conditioning cliff
        r2 = 0.955 + 0.004 * (i - 6)
        filters.append(resonator(r2, 1.15))
        if i >= 8:
            filters.append(real_root(1.02 + 0.007 * (i - 8)))  # a second NMP zero
    else:
        # easy regime: an extra minimum-phase zero, no blow-up
        filters.append(real_root(0.68 + 0.03 * i))

    h = cascade(filters)
    L = len(h)
    M = N + L - 1

    # ----- build a reachable target y* = conv(x_true,h) -----
    t = np.arange(N)
    x_true = np.zeros(N, dtype=np.float64)
    # a few smooth low-frequency components (a bounded pulse shape exists)
    for _ in range(3):
        w = npr.uniform(0.03, 0.22)
        ph = npr.uniform(0, 2 * math.pi)
        amp = npr.uniform(0.22, 0.40) * A
        x_true += amp * np.sin(2 * math.pi * w * t + ph)
    # planted spikes that EXCEED the amplitude bound -> the box constraint binds, so a
    # single clip of the Tikhonov solution is suboptimal vs a true box-constrained solve.
    n_spk = 3 + i // 3
    for _ in range(n_spk):
        pp = rng.randrange(N)
        x_true[pp] += rng.choice([-1.0, 1.0]) * (1.3 + 0.6 * rng.random()) * A

    base = np.convolve(x_true, h)
    S = float(np.dot(base, base))
    rms = math.sqrt(S / M) + 1e-12

    # ----- plant a genuinely UNREACHABLE component -----
    # The convolution operator H (M x N) has an (L-1)-dimensional left null space: no input
    # can ever produce those output directions.  Injecting energy there caps the achievable
    # score and forces every solver -- including the reference -- to leave a residual.  This
    # embodies the near-unit-circle-zero obstruction in exact, box-independent form.
    H = np.zeros((M, N), dtype=np.float64)
    for j in range(N):
        H[j:j + L, j] = h
    U, sv, Vt = np.linalg.svd(H, full_matrices=True)
    Z = U[:, N:]                                    # columns span the left null space
    coef = npr.standard_normal(Z.shape[1])
    v = Z @ coef
    v = v / (np.linalg.norm(v) + 1e-12)
    alpha = 0.13                                    # unreachable energy as a fraction of S
    unreach = math.sqrt(alpha * S) * v

    noise = npr.standard_normal(M) * (0.02 * rms)   # tiny broadband imperfection
    ystar = base + unreach + noise

    # ----- emit -----
    out = []
    out.append("%d %.10g %.10g %d" % (N, A, lam, len(filters)))
    for f in filters:
        out.append(str(len(f)))
        out.append(" ".join("%.17g" % c for c in f))
    out.append(str(M))
    out.append(" ".join("%.17g" % v for v in ystar))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
