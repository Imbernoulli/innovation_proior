#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0872 -- "Petal Press: Dispersion-Tuned Disk Fission"
(family: mode-k-symmetry-breaking; format B, quality-metric).

THEME.  A disk's rim is modeled as N=64 points around a ring, u_0..u_{N-1}.  The
rim starts near-uniform, perturbed only by tiny hidden random noise (never shown
to the candidate).  A fixed nonlinear growth process then runs for T steps; it
pointwise-saturates with a cubic term and keeps the field zero-mean (no bulk
growth mode -- only *shape*, i.e. symmetry breaking, can win).  The candidate
supplies the LINEAR COUPLING KERNEL of that process (a short-range convolution
on the ring) plus a tiny deterministic bias added to the initial state.  The
kernel's Fourier transform is the dispersion relation: it decides which
wavenumber m grows fastest before the cubic term saturates it.  After T steps we
count the sign changes of the final field around the ring; half that count is
the number of "petals" (domains).  The instance names a target k; the score is
the FRACTION of hidden noise realizations whose final petal count is EXACTLY k.

MECHANISM COMPOSITION.
  - wavelength-selection: score rewards resonance of the kernel's Fourier peak
    with wavenumber k, not merely "adding structure".
  - symmetry-breaking: the zero-mean constraint kills the trivial m=0 (no
    pattern) attractor, so *some* nonzero wavenumber must win every run.
  - mode-robustness: score is a FRACTION over M independent hidden noise draws
    -- a kernel that wins for one lucky draw but not the ensemble scores low.

INNOVATION HOOK.  A single strong, spatially localized bias (a "big seed") can
only ever drag the dynamics toward whatever wavenumber the UNSHAPED kernel
already favors (empirically: a plain local-averaging/diffusive kernel always
favors the LOWEST surviving wavenumber, m=1 -- a single domain).  Reliably
hitting an arbitrary target k instead requires shaping the kernel's dispersion
relation lambda(m) = w_0 + 2*sum_j w_j*cos(2*pi*j*m/N) so wavenumber k is the
one with (approximately) the largest lambda -- i.e. designing the OPERATOR, not
injecting a bigger perturbation.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "N": 64, "k": int, "L_max": int, "W_max": float,
             "B": float, "T": int, "M": int, "noise_amp": float}
          k is the target petal count.  L_max is the max kernel half-width the
          candidate may use (kernel has L_max+1 taps: w_0..w_{L_max}, applied
          symmetrically).  W_max bounds |w_j|.  B bounds the L2 norm of the
          bias vector (length N) and is DELIBERATELY tiny (~0.02 of the noise
          field's own L2 norm) -- small enough that even a bias aimed exactly
          at cos(2*pi*k*i/N) cannot noticeably move the score by itself; only
          genuine dispersion-shaping via the kernel can.  T, M, noise_amp
          describe the (hidden) noise ensemble the evaluator runs over.
  stdout: ONE JSON object:
            {"kernel": [w_0, ..., w_{L_max}], "bias": [b_0, ..., b_{N-1}]}

  VALID iff: kernel has exactly L_max+1 finite numbers with |w_j| <= W_max;
  bias has exactly N finite numbers with L2 norm <= B.  Any violation, a crash,
  a timeout, or non-JSON output makes that instance score 0.0.

DYNAMICS (run by the evaluator, never by the candidate).  Given kernel w and
bias b, for each of M hidden noise draws eta (i.i.d. Uniform(-noise_amp,
noise_amp) per point, seeded from the instance):
    u <- (b + eta) with its mean subtracted
    repeat T times:
        lin_i  <- w_0*u_i + sum_{j=1}^{L} w_j*(u_{i+j} + u_{i-j})   (indices mod N)
        u      <- u + dt*(lin - alpha*u^3)
        u      <- u with its mean subtracted, then clipped to [-8, 8]
    domains(u) <- (# indices i where sign(u_i) != sign(u_{i+1})) // 2
The instance's score contribution is the FRACTION of the M draws with
domains(u) == k EXACTLY.

SCORING (deterministic; no wall-time).  Per instance the evaluator also computes
two of its OWN reference kernels (never the candidate's):
  frac_naive  = match-fraction of the IDENTITY kernel (w_0=1, all w_j=0) with
                zero bias -- no coupling at all, a "did nothing" anchor.
  frac_ideal  = match-fraction of a fixed, WIDER reference kernel (more taps
                than any candidate is allowed: L=14, amplitude 0.5, shaped as
                cos(2*pi*k*j/N)) with zero bias -- an idealized, but not always
                reachable, dispersion-matched filter.
and normalizes:
    r = clamp( 0.1 + 0.82 * (frac_cand - frac_naive) / max(1e-9, frac_ideal - frac_naive), 0, 1 )
Doing nothing scores ~0.1; matching the idealized reference's fraction scores
up to 0.92 (never 1.0, by construction) -- deliberate headroom, since the
candidate is always capped at fewer taps (L_max <= 10) than the L=14 reference.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  The noise draws
and both reference kernels are computed by THIS parent process only.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math, hashlib
import numpy as np
import isorun

# ----------------------------- fixed constants ------------------------------
N = 64
DT = 0.1
ALPHA = 1.0
CLIP = 8.0
T_STEPS = 150
NOISE_AMP = 0.3
M_REAL = 60
W_MAX = 1.2
ORACLE_L = 14
ORACLE_AMP = 0.5
SCALE = 0.82
CAND_TIMEOUT = 10

BIAS_FACTOR = 0.02
B_BUDGET = round(BIAS_FACTOR * NOISE_AMP * math.sqrt(N / 3.0), 6)

# (target k, candidate's allowed kernel half-width L_max, seed offset)
INSTANCE_SPECS = [
    (1,  4, 0),
    (1,  8, 1),
    (6,  6, 2),
    (10, 6, 3),
    (3, 10, 4),
    (14, 8, 5),
    (8,  6, 6),
    (12, 8, 7),
    (20, 8, 8),
    (28, 10, 9),
]


# ----------------------------- deterministic RNG ----------------------------
def _rng_noise(seed, n, m, amp):
    rs = np.random.RandomState(seed)
    return rs.uniform(-amp, amp, size=(m, n))


# ----------------------------- dynamics core --------------------------------
def _run_dynamics(kernel, bias, noises, T=T_STEPS, dt=DT, alpha=ALPHA, clip=CLIP):
    """kernel: (L+1,) array; bias: (N,) array; noises: (M,N) array -> (M,N) final field."""
    L = len(kernel) - 1
    u = np.asarray(bias, dtype=float)[None, :] + noises
    u = u - u.mean(axis=1, keepdims=True)
    for _ in range(T):
        lin = kernel[0] * u
        for j in range(1, L + 1):
            lin = lin + kernel[j] * (np.roll(u, -j, axis=1) + np.roll(u, j, axis=1))
        du = dt * (lin - alpha * u ** 3)
        u = u + du
        u = u - u.mean(axis=1, keepdims=True)
        u = np.clip(u, -clip, clip)
    return u


def _domains(u):
    sign = np.where(u >= 0, 1, -1)
    changes = (sign != np.roll(sign, 1, axis=1)).sum(axis=1)
    return changes // 2


def _frac_match(k, kernel, bias, noises):
    u = _run_dynamics(kernel, bias, noises)
    d = _domains(u)
    return float((d == k).mean())


def _cos_kernel(k, L, amp, n=N):
    j = np.arange(0, L + 1)
    return amp * np.cos(2 * math.pi * k * j / n)


def _seed_for(off, k, Lmax):
    """Deterministic seed, but not a simple offset the candidate can read off the
    public instance (the public 'name' field carries no ordinal/seed information)."""
    digest = hashlib.sha256(f"fsx_A_0872:{off}:{k}:{Lmax}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


# ----------------------------- instance family -------------------------------
def make_instances():
    out = []
    for k, Lmax, off in INSTANCE_SPECS:
        seed = _seed_for(off, k, Lmax)
        noises = _rng_noise(seed, N, M_REAL, NOISE_AMP)
        public = {"name": "disk", "N": N, "k": k, "L_max": Lmax,
                  "W_max": W_MAX, "B": B_BUDGET, "T": T_STEPS, "M": M_REAL,
                  "noise_amp": NOISE_AMP}
        hidden = {"k": k, "Lmax": Lmax, "noises": noises}
        out.append({"public": public, "hidden": hidden})
    return out


# ----------------------------- answer validation ------------------------------
def _validate_answer(pub, answer):
    if not isinstance(answer, dict):
        return None
    Lmax = pub["L_max"]; n = pub["N"]; wmax = pub["W_max"]; bbud = pub["B"]
    kernel = answer.get("kernel")
    bias = answer.get("bias")
    if not isinstance(kernel, list) or len(kernel) != Lmax + 1:
        return None
    if not isinstance(bias, list) or len(bias) != n:
        return None
    kk = []
    for x in kernel:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return None
        xf = float(x)
        if not math.isfinite(xf) or abs(xf) > wmax + 1e-9:
            return None
        kk.append(xf)
    bb = []
    for x in bias:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return None
        xf = float(x)
        if not math.isfinite(xf):
            return None
        bb.append(xf)
    norm = math.sqrt(sum(x * x for x in bb))
    if norm > bbud + 1e-6:
        return None
    return np.array(kk, dtype=float), np.array(bb, dtype=float)


# ----------------------------- scoring driver ---------------------------------
def score(inst, answer):
    """Return (ok, frac_cand) validating `answer` against inst's public+hidden data."""
    pub = inst["public"]; hid = inst["hidden"]
    parsed = _validate_answer(pub, answer)
    if parsed is None:
        return False, 0.0
    kernel, bias = parsed
    frac = _frac_match(hid["k"], kernel, bias, hid["noises"])
    return True, frac


def baseline(inst):
    """Return (frac_naive, frac_ideal): the evaluator's own two reference fractions."""
    pub = inst["public"]; hid = inst["hidden"]
    n = pub["N"]; k = hid["k"]; Lmax = hid["Lmax"]; noises = hid["noises"]
    triv_kernel = np.zeros(Lmax + 1)
    triv_kernel[0] = 1.0
    frac_naive = _frac_match(k, triv_kernel, np.zeros(n), noises)
    oracle_kernel = _cos_kernel(k, ORACLE_L, ORACLE_AMP, n=n)
    frac_ideal = _frac_match(k, oracle_kernel, np.zeros(n), noises)
    return frac_naive, frac_ideal


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = make_instances()

    vec = []
    for inst in instances:
        pub = inst["public"]
        ans, st = isorun.run_candidate(cand, pub, timeout=CAND_TIMEOUT)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, frac_cand = score(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0)
            continue
        frac_naive, frac_ideal = baseline(inst)
        denom = max(1e-9, frac_ideal - frac_naive)
        r = 0.1 + SCALE * (frac_cand - frac_naive) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        vec.append(max(0.0, min(1.0, r)))

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
