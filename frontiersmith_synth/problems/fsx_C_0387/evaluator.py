#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_0387 -- "Coral Reef Survey: Robust Transect Balancing"
(family: optimization-convergence-analysis; format B, eval_form flops).

THEME.  A marine lab runs an autonomous reef-survey vehicle.  Two teams argue over
each dive plan: the SURVEY planner picks a control vector x (how long to linger on
each transect) to minimise a cost, while an adversarial CURRENT/estimator picks a
disturbance vector y that maximises the same cost (worst-case drift, sensor bias).
The equilibrium dive plan is the saddle point of a convex-concave objective

    f(x, y) = 1/2 x^T A x  +  x^T B y  -  1/2 y^T C y  +  a^T x  -  c^T y ,

with A, C symmetric positive-definite (each team's own curvature) and B the
survey/current coupling.  The joint stationarity condition is V(z) = 0 where
z = (x, y) and V is the (monotone) saddle vector field

    V(z) = ( A x + B y + a ,  -B^T x + C y + c ) = H z + q ,
    H = [[A, B], [-B^T, C]]  (positive-definite: z^T H z = x^T A x + y^T C y > 0).

You get a FIXED BUDGET of K extragradient-style iterations (2 gradient/field
evaluations each, NO wall-time) from a fixed start z0.  Each iteration k applies

    z_half   = z_k - alpha_k * V(z_k)
    z_{k+1}  = z_k - beta_k  * V(z_half)          # extragradient template

Setting alpha_k = 0 recovers a plain descent-ascent step of size beta_k; a positive
alpha_k adds an extrapolation (extragradient) correction that tames the rotational
part of the saddle field.  YOUR JOB is to DESIGN THE SCHEDULE {alpha_k, beta_k}
that drives the FINAL survey residual ||V(z_K)|| as small as possible.  Because the
state dimension exceeds 2K, no schedule can zero the residual -- this is a genuine
min-residual (Chebyshev-type) design problem with no closed-form optimum.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "N": int, "K": int,
             "H": [[float]*N]*N,          # the saddle operator
             "q": [float]*N,              # affine term (V(z)=H z + q)
             "z0": [float]*N,             # fixed start point
             "Lspec": float,              # spectral-norm bound ||H||_2
             "ref_alpha": float, "ref_beta": float}   # the reference (weak) step
  stdout: ONE JSON object:
            {"alpha": [float]*K, "beta": [float]*K}   # the update schedule

  VALID iff alpha and beta are each a list of exactly K finite real numbers with
  |value| <= 1e8.  Any other shape, a non-finite entry, a crash, a timeout, or a
  schedule whose iterates blow up (non-finite / norm > 1e12) -> that instance 0.0.

SCORING (deterministic; no wall-time).  For each instance the parent computes the
final residual by RE-RUNNING the extragradient template with the candidate schedule:
    g0     = ||V(z0)||                                   (initial residual)
    g_base = ||V(z_K)|| under the REFERENCE schedule      (weak constant step)
    g_cand = ||V(z_K)|| under the candidate schedule
    g_ideal = g0 * IDEAL_RATIO                            (unreachable ideal floor)
  and normalises on a LOG anchor (reference -> 0.1, ideal floor -> 1.0):
    r = clamp( 0.1 + 0.9 * (log10 g_base - log10 g_cand)
                          / (log10 g_base - log10 g_ideal),  0, 1 )
  Reproducing the reference schedule scores ~0.1; beating it drives r up, but the
  ideal floor is unreachable in K steps so even a Chebyshev-optimal design stays
  below 1.0 (headroom).  Doing worse than the reference scores < 0.1.

ISOLATION.  The candidate is untrusted and runs OS-sandboxed in a fresh subprocess
via `isorun.run_candidate`; it only ever sees the PUBLIC instance.  All references
(g0, g_base, g_ideal) are computed by THIS parent process from data the sandbox
cannot reach, so a frame-walking / introspecting candidate learns nothing useful --
the only way to score is to submit a genuinely better schedule.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import numpy as np
import isorun

IDEAL_RATIO = 1e-4          # unreachable residual floor (relative to g0)
MAXVAL = 1e8                # per-coefficient magnitude cap


# ----------------------------- instance family -----------------------------
# (seed, d, kappa, mu): d = per-block dim (state N = 2d), kappa = condition number
# of each team's curvature, mu = survey/current coupling strength.
_SPECS = [
    (101, 20,   8, 0.15),
    (102, 20,  20, 0.25),
    (103, 20,  40, 0.20),
    (104, 20,  12, 0.30),
    (105, 20,  60, 0.10),
    (106, 20,  30, 0.20),
    (107, 20,  50, 0.15),
    (108, 20,  16, 0.25),
    # harder / larger held-out reefs
    (211, 24,  80, 0.20),
    (212, 24, 100, 0.25),
    (213, 24,  25, 0.15),
    (214, 24,  64, 0.30),
]
_K = 12


def _spd(rng, d, kappa):
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    lam = np.exp(np.linspace(0.0, math.log(kappa), d))     # spectrum in [1, kappa]
    return (Q * lam) @ Q.T


def _build(seed, d, kappa, mu):
    rng = np.random.default_rng(seed)
    A = _spd(rng, d, kappa)
    C = _spd(rng, d, kappa)
    B = mu * rng.standard_normal((d, d))
    H = np.block([[A, B], [-B.T, C]])
    N = 2 * d
    zstar = rng.standard_normal(N)
    q = -H @ zstar
    z0 = zstar + rng.standard_normal(N)
    return H, q, z0, N


def _run_method(H, q, z0, alpha, beta, K):
    """Run the extragradient template. Return final residual norm, or None if it
    produces a non-finite iterate or blows up."""
    z = np.array(z0, dtype=float)
    for k in range(K):
        v = H @ z + q
        zh = z - alpha[k] * v
        vh = H @ zh + q
        z = z - beta[k] * vh
        if not np.all(np.isfinite(z)):
            return None
        if np.linalg.norm(z) > 1e12:
            return None
    g = H @ z + q
    if not np.all(np.isfinite(g)):
        return None
    return float(np.linalg.norm(g))


def _build_instances():
    out = []
    for seed, d, kappa, mu in _SPECS:
        H, q, z0, N = _build(seed, d, kappa, mu)
        L = float(np.linalg.norm(H, 2))
        g0 = float(np.linalg.norm(H @ z0 + q))
        ref = 0.4 / L
        g_base = _run_method(H, q, z0, [ref] * _K, [ref] * _K, _K)
        out.append({"name": f"reef{seed}", "N": N, "K": _K,
                    "H": H, "q": q, "z0": z0, "Lspec": L,
                    "ref_alpha": ref, "ref_beta": ref,
                    "g0": g0, "g_base": g_base})
    return out


def _public(inst):
    return {"name": inst["name"], "N": inst["N"], "K": inst["K"],
            "H": inst["H"].tolist(), "q": inst["q"].tolist(),
            "z0": inst["z0"].tolist(), "Lspec": inst["Lspec"],
            "ref_alpha": inst["ref_alpha"], "ref_beta": inst["ref_beta"]}


# ----------------------------- validation ----------------------------------
def _valid_schedule(answer, K):
    if not isinstance(answer, dict):
        return None
    a = answer.get("alpha")
    b = answer.get("beta")
    if not isinstance(a, list) or not isinstance(b, list):
        return None
    if len(a) != K or len(b) != K:
        return None
    for seq in (a, b):
        for v in seq:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return None
            fv = float(v)
            if not math.isfinite(fv) or abs(fv) > MAXVAL:
                return None
    return [float(v) for v in a], [float(v) for v in b]


def _score_instance(inst, answer):
    chk = _valid_schedule(answer, inst["K"])
    if chk is None:
        return 0.0
    alpha, beta = chk
    g_cand = _run_method(inst["H"], inst["q"], inst["z0"], alpha, beta, inst["K"])
    if g_cand is None or g_cand <= 0.0:
        # non-finite / blow-up -> 0 ; an exact zero residual is impossible here so
        # a non-positive value indicates degenerate output.
        return 0.0 if g_cand is None else 1.0
    g_base = inst["g_base"]
    g0 = inst["g0"]
    if g_base is None or g_base <= 0.0:
        return 0.0
    g_ideal = g0 * IDEAL_RATIO
    denom = math.log10(g_base) - math.log10(g_ideal)
    if denom <= 1e-9:
        return 0.0
    num = math.log10(g_base) - math.log10(max(g_cand, 1e-300))
    r = 0.1 + 0.9 * num / denom
    if not math.isfinite(r):
        return 0.0
    return max(0.0, min(1.0, r))


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        ans, st = isorun.run_candidate(cand, _public(inst), timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            r = _score_instance(inst, ans)
        except Exception:
            r = 0.0
        if not (isinstance(r, float) and r == r and 0.0 <= r <= 1.0):
            r = 0.0
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
