#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_0195 -- "Interstellar Relay: Beam-Alignment Saddle Optimizer"
(tier C; family: optimization-convergence-analysis; eval_form: flops;
inspired by MLS-Bench optimization-* tasks).

STORY.  A chain of deep-space relay stations must jointly steer their phased-array
beams.  The transmit side (variables x) wants to MINIMISE a coupling cost while the
adversarial channel / interference side (variables y) wants to MAXIMISE it.  The
equilibrium is the saddle point of a convex-concave objective

        f(x, y) = 1/2 x^T A x  +  x^T B y  -  1/2 y^T C y  +  b^T x  +  c^T y ,

with A, C symmetric positive semidefinite (convex in x, concave in y) and B the
cross-coupling ("rotation") between the two sides.  The natural monotone operator is

        F(z) = ( grad_x f , -grad_y f ) = M z + q ,   z = (x, y),
        M = [[ A ,  B ],
             [-B^T,  C ]]  = S + K   (S = blockdiag(A,C) symmetric PSD, K skew).

At the equilibrium z* we have F(z*) = 0, so the RESIDUAL GRADIENT NORM  ||F(z_T)||
measures how far a run is from equilibrium after a FIXED ITERATION BUDGET of T steps
(op-count budget, NOT wall-time -- fully deterministic).

THE DESIGN TASK.  The candidate does NOT run the optimiser and never sees the hidden
matrices.  It designs the *update schedule* of a FROZEN generalised first-order
saddle-point method and submits the schedule as data.  The evaluator then executes
that schedule ITSELF on several HIDDEN problems drawn from the same relay class and
recomputes ||F(z_T)|| deterministically.  Because the actual matrices are hidden, the
candidate must design a schedule that GENERALISES from the class descriptors
(dimension, budget T, Lipschitz bound L, strong-monotonicity mu, class type) -- this
is convergence-rate analysis, not offline solving.

FROZEN UPDATE TEMPLATE (executed by the evaluator; the candidate only picks the
coefficient schedules eta_t, alpha_t, beta_t for t = 0..T-1):

        m = 0
        for t in range(T):
            g      = F(z)                       # gradient at current iterate
            z_look = z - alpha_t * eta_t * g    # extrapolation / look-ahead
            g_look = F(z_look)                   # gradient at the look-ahead point
            m      = beta_t * m + g_look          # (heavy-ball) momentum buffer
            z      = z - eta_t * m
    final objective = || F(z_T) ||

    alpha=1, beta=0  -> classic extragradient (EG); alpha=0,beta=0 -> plain gradient
    descent-ascent (GDA, which DIVERGES on rotation-dominated instances); other
    settings give optimistic / momentum / accelerated variants.

REFERENCE METHOD (the evaluator's baseline).  Extragradient with the constant step
size eta = 1/(2L) for all t (alpha=1, beta=0).  This is stable on every relay class
(EG converges for eta < 1/L on monotone L-Lipschitz operators).  A candidate that
reproduces this exact schedule normalises to ~0.1 on every instance.

ISOLATION.  The candidate is untrusted model output run as an ISOLATED subprocess
(isorun): it reads ONE JSON public view from stdin and writes ONE JSON schedule to
stdout.  The hidden matrices, the ground-truth saddle points, and this evaluator's
state live only in the parent process.

Public instance JSON (stdin):
    {
      "type":       str,     # relay class: "strong"|"mixed"|"bilinear"|"illcond"
      "n":          int,     # block dimension (dim x = dim y = n; total dim = 2n)
      "dim":        int,     # 2n
      "T":          int,     # iteration budget (number of update steps)
      "L":          float,   # Lipschitz upper bound on ||M|| (safe: 1/L step is stable-ish)
      "mu":         float,   # strong-monotonicity lower bound (min eig of symmetric part); 0 if none
      "num_hidden": int,     # how many hidden problems your schedule is scored on
      "seed":       int      # advisory seed
    }

Answer JSON (stdout):
    {"eta": <float | [T floats]>,          # step sizes (REQUIRED)
     "alpha": <float | [T floats]>,         # look-ahead fraction (optional, default 1.0)
     "beta":  <float | [T floats]>}         # momentum coefficient (optional, default 0.0)
    A scalar is broadcast to all T steps.  Every value must be finite.

Objective: MINIMISE the mean final residual gradient norm across the hidden problems.
Per-instance normalisation (minimisation form):
        r = min(1.0, 0.1 * B / max(obj, 1e-12))
where B is the reference (EG 1/(2L)) mean final norm and obj is the candidate's mean
final norm.  So reproducing the reference -> 0.1, and being 10x better -> 1.0.  An
invalid schedule (wrong shape / non-finite / missing eta), a crash, or a timeout
scores exactly 0 for that instance.  The reported Ratio is the arithmetic mean of the
per-instance r.

CLI:  python3 evaluator.py <candidate.py>
Prints:
    Ratio: <mean r in [0,1]>
    Vector: [r_1, ..., r_k]
"""
import os
# Pin BLAS/OMP threads so numpy is deterministic and imports cleanly inside the
# RLIMIT-capped candidate child (isorun copies this env into the subprocess).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys, json, math
import numpy as np
import isorun

CAND_TIMEOUT = 20
BASE_FRAC = 0.5          # reference EG step = BASE_FRAC / L
DIVERGE_CAP = 1.0e18     # objective assigned to a run that blows up


# ============================ relay-class problem generator ================
def _sym_psd(rng, n, lo, hi):
    """Random symmetric matrix with eigenvalues drawn uniformly in [lo, hi]."""
    if hi <= 0.0:
        return np.zeros((n, n))
    G = rng.normal(0.0, 1.0, size=(n, n))
    Q, _ = np.linalg.qr(G)
    ev = rng.uniform(lo, hi, size=n)
    return (Q * ev) @ Q.T


def _make_problem(seed, n, ptype, mu_s, L_s, L_b):
    """Build one hidden saddle instance of a given relay class.

    Returns (M, q, z0, Lhat, mu) where M z* + q = 0 at the saddle, z0 is the start,
    Lhat is a spectral upper bound (>= ||M||) and mu is a lower bound on the min
    eigenvalue of the symmetric part.  The public view exposes only Lhat and mu.
    """
    rng = np.random.default_rng(seed)
    A = _sym_psd(rng, n, mu_s, L_s)
    C = _sym_psd(rng, n, mu_s, L_s)
    if L_b > 0.0:
        B = rng.normal(0.0, 1.0, size=(n, n))
        nrm = np.linalg.norm(B, 2)
        B = B * (L_b / (nrm + 1e-12))          # spectral norm == L_b exactly
    else:
        B = np.zeros((n, n))
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = A
    M[:n, n:] = B
    M[n:, :n] = -B.T
    M[n:, n:] = C
    # ground-truth saddle z*: pick it, set q = -M z*
    zstar = rng.normal(0.0, 1.0, size=2 * n)
    q = -M @ zstar
    # start a fixed-scale displacement away from z*
    delta = rng.normal(0.0, 1.0, size=2 * n)
    delta = delta / (np.linalg.norm(delta) + 1e-12) * 3.0
    z0 = zstar + delta
    Lhat = float(L_s + L_b) if L_s > 0.0 else float(L_b)
    mu = float(mu_s)
    return M, q, z0, Lhat, mu


# ============================ frozen update template =======================
def _run_schedule(M, q, z0, eta, alpha, beta, T):
    """Execute the frozen generalised EG+momentum update; return ||F(z_T)||."""
    z = z0.astype(np.float64).copy()
    m = np.zeros_like(z)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for t in range(T):
            g = M @ z + q
            z_look = z - (alpha[t] * eta[t]) * g
            g_look = M @ z_look + q
            m = beta[t] * m + g_look
            z = z - eta[t] * m
            nz = np.linalg.norm(z)
            if not np.isfinite(nz) or nz > 1.0e12:
                return DIVERGE_CAP
        final = float(np.linalg.norm(M @ z + q))
    if not np.isfinite(final):
        return DIVERGE_CAP
    return final


def _class_problems(spec):
    """Deterministically materialise the hidden problems for one instance class."""
    probs = []
    for h in range(spec["num_hidden"]):
        seed = spec["seed"] * 100003 + h * 97
        M, q, z0, Lhat, mu = _make_problem(
            seed, spec["n"], spec["type"], spec["mu_s"], spec["L_s"], spec["L_b"])
        probs.append((M, q, z0, Lhat, mu))
    return probs


def _build_specs():
    # Diverse relay classes: strongly-monotone (easy) ... rotation/ill-conditioned (hard).
    return [
        dict(name="strong_easy", type="strong",   seed=101, n=8, T=50, num_hidden=4,
             mu_s=0.80, L_s=1.6, L_b=0.30),
        dict(name="strong_mod",  type="strong",   seed=102, n=8, T=50, num_hidden=4,
             mu_s=0.30, L_s=3.0, L_b=1.00),
        dict(name="mixed_a",     type="mixed",    seed=103, n=8, T=50, num_hidden=4,
             mu_s=0.20, L_s=2.0, L_b=2.00),
        dict(name="bilinear_a",  type="bilinear", seed=104, n=8, T=50, num_hidden=4,
             mu_s=0.00, L_s=0.0, L_b=2.00),
        dict(name="bilinear_b",  type="bilinear", seed=105, n=8, T=50, num_hidden=4,
             mu_s=0.00, L_s=0.0, L_b=4.00),
        dict(name="illcond_a",   type="illcond",  seed=106, n=8, T=50, num_hidden=4,
             mu_s=0.05, L_s=8.0, L_b=0.50),
        dict(name="illcond_b",   type="illcond",  seed=107, n=8, T=50, num_hidden=4,
             mu_s=0.02, L_s=12.0, L_b=1.00),
        dict(name="mixed_ill",   type="mixed",    seed=108, n=8, T=50, num_hidden=4,
             mu_s=0.05, L_s=6.0, L_b=3.00),
    ]


# ============================ candidate answer handling ====================
def _coerce(v, T):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        arr = np.full(T, float(v), dtype=np.float64)
    elif isinstance(v, list):
        if len(v) != T:
            return None
        try:
            arr = np.asarray(v, dtype=np.float64)
        except Exception:
            return None
        if arr.ndim != 1 or arr.shape[0] != T:
            return None
    else:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _valid_schedule(ans, T):
    if not isinstance(ans, dict):
        return None
    if "eta" not in ans or ans["eta"] is None:
        return None
    eta = _coerce(ans["eta"], T)
    alpha = _coerce(ans.get("alpha", 1.0), T)
    beta = _coerce(ans.get("beta", 0.0), T)
    if eta is None or alpha is None or beta is None:
        return None
    return eta, alpha, beta


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <candidate.py>")
        sys.exit(2)
    cand = sys.argv[1]
    specs = _build_specs()

    vec = []
    for spec in specs:
        T = spec["T"]
        probs = _class_problems(spec)

        # reference (baseline) mean final norm on the hidden problems
        base_vals = []
        for (M, q, z0, Lhat, mu) in probs:
            eta_b = np.full(T, BASE_FRAC / Lhat)
            al_b = np.full(T, 1.0)
            be_b = np.zeros(T)
            base_vals.append(_run_schedule(M, q, z0, eta_b, al_b, be_b, T))
        B = float(np.mean(base_vals))

        # public view (exposes class descriptors only -- NOT the matrices)
        Lpub = float(spec["L_s"] + spec["L_b"]) if spec["L_s"] > 0.0 else float(spec["L_b"])
        public = {
            "type": spec["type"],
            "n": int(spec["n"]),
            "dim": int(2 * spec["n"]),
            "T": int(T),
            "L": Lpub,
            "mu": float(spec["mu_s"]),
            "num_hidden": int(spec["num_hidden"]),
            "seed": int(20240195 + spec["seed"]),
        }

        ans, st = isorun.run_candidate(cand, public, timeout=CAND_TIMEOUT)
        if st != "OK":
            vec.append(0.0)
            continue
        sched = _valid_schedule(ans, T)
        if sched is None:
            vec.append(0.0)
            continue
        eta, alpha, beta = sched

        try:
            cand_vals = []
            for (M, q, z0, Lhat, mu) in probs:
                cand_vals.append(_run_schedule(M, q, z0, eta, alpha, beta, T))
            obj = float(np.mean(cand_vals))
        except Exception:
            vec.append(0.0)
            continue
        if not np.isfinite(obj):
            vec.append(0.0)
            continue

        r = 0.1 * B / max(obj, 1e-12)
        if not np.isfinite(r):
            r = 0.0
        r = max(0.0, min(1.0, r))
        vec.append(float(r))

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(v, 6) for v in vec]))


if __name__ == "__main__":
    main()
