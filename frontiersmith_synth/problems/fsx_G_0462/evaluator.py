#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_G_0462 -- "Spectral Trainer: Learning-Rate Schedule Design"
(family: ml-lr-schedule; format B, quality-metric).

THEME.  A convex trainer runs a FIXED number of gradient-descent steps on a FIXED
convex quadratic loss and you must hand it a learning-rate SCHEDULE -- one step size
per iteration -- that drives the final gradient as close to zero as possible.

The loss is a diagonal (separable) convex quadratic
        L(w) = sum_i  0.5 * h_i * (w_i - w*_i)^2 ,   h_i > 0
so coordinate i has curvature h_i and gradient component  g_i = h_i*(w_i - w*_i).
Plain gradient descent with schedule (eta_0,...,eta_{N-1}) evolves each coordinate
independently, so the gradient component after all N steps is
        g_i(N) = g_i(0) * PROD_{t=0}^{N-1} (1 - eta_t * h_i).
The trainer's final gradient NORM is  ||g(N)||_2 = sqrt( sum_i g_i(N)^2 ),  and we
MINIMIZE it.  (Working in the Hessian's eigenbasis is WLOG for GD, so the diagonal
view is fully general -- it is exactly the spectral picture of quadratic optimization.)

The schedule is FIXED in advance (a function of the step index, not of the running
iterate): the candidate sees the whole spectrum h and the initial gradient g(0) and
must design the N step sizes offline.  Because the product over t is order-independent,
the schedule is really a choice of a degree-N polynomial p(x)=PROD_t(1-eta_t*x) with
p(0)=1, and the score measures how small p makes  sqrt( sum_i (g_i(0)*p(h_i))^2 ).

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str,
             "n_steps": N (int),
             "curv":  [h_0, ..., h_{d-1}],   # positive curvatures (Hessian diagonal)
             "grad0": [g_0, ..., g_{d-1}]}    # initial gradient components (real)
  stdout: ONE JSON object:
            {"lr": [eta_0, ..., eta_{N-1}]}   # exactly N finite real step sizes

  A schedule is VALID iff `lr` is a list of exactly N finite real numbers (int/float,
  no bool, no nan/inf).  Wrong length, non-number, non-finite entry, a crash, a
  timeout, or non-JSON -> that instance scores 0.0.  (Divergent-but-finite schedules
  are allowed; they simply produce a huge gradient norm and score near 0.)

SCORING (deterministic; no wall-time).  Per instance the parent computes, in
log10(gradient-norm) space, two anchors and the candidate's value:
    G_base = log10 || grad after the WEAK constant schedule eta_t = 1/L ||     # 0.1
    G_cheb = log10 || grad after the N-step CHEBYSHEV schedule on [mu, L] ||    # near-optimal
    G_ideal = G_cheb - MARGIN                                                   # unreachable, 1.0
    G_cand  = log10 || grad after the candidate schedule ||
where mu=min(curv), L=max(curv), and the Chebyshev schedule uses step sizes 1/rho_k
with rho_k the roots of the degree-N minimax residual polynomial on [mu, L].  Then
    r = clamp( 0.1 + 0.9 * (G_base - G_cand) / max(G_base - G_ideal, 1e-6), 0, 1 ).
A candidate reproducing the weak constant schedule scores ~0.1; the (near-optimal)
Chebyshev schedule scores well below 1.0; only a schedule an order of magnitude
better than Chebyshev could reach 1.0, and the discrete/weighted spectrum makes that
unreachable -> headroom.  Doing worse than the weak baseline scores < 0.1 (down to 0).

ISOLATION.  The candidate is untrusted and runs OS-sandboxed in a FRESH SUBPROCESS
via `isorun.run_candidate`; it only ever sees the PUBLIC instance.  Both anchors are
computed by THIS parent process, so a frame-walking / introspecting candidate learns
nothing it was not already handed.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun

MARGIN = 1.0          # orders of magnitude of headroom beyond Chebyshev
_TINY = 1e-300        # floor so gradient norm never underflows to log(0)


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def u():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return ((state >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    return u


# ----------------------------- instance family -----------------------------
def _build(seed, d, N, kappa, deig, dg):
    """Deterministic diagonal convex quadratic. mu=1, L=kappa (condition number)."""
    u = _rng(seed)
    mu = 1.0
    L = float(kappa)
    curv = []
    for _ in range(d):
        t = u()
        if deig == "uniform":                       # eigenvalues spread linearly
            lam = mu + (L - mu) * t
        elif deig == "geom":                        # log-uniform spacing
            lam = mu * (L / mu) ** t
        elif deig == "edges":                       # arcsine: mass piled near mu and L (hard)
            lam = (L + mu) / 2 - (L - mu) / 2 * math.cos(math.pi * t)
        else:
            lam = mu + (L - mu) * t
        curv.append(lam)
    curv[0] = mu                                     # pin the spectrum endpoints
    curv[1] = L
    if dg == "flat":
        grad0 = [1.0] * d
    elif dg == "rand":
        grad0 = [0.5 + u() for _ in range(d)]
    elif dg == "lowmode":                            # weight low-curvature (slow) modes
        grad0 = [1.0 / math.sqrt(curv[i]) for i in range(d)]
    else:
        grad0 = [1.0] * d
    return {"name": f"trainer{seed}", "n_steps": N,
            "curv": curv, "grad0": grad0, "mu": mu, "L": L}


def _build_instances():
    specs = [
        (101, 50, 24, 50,  "uniform", "flat"),
        (102, 50, 24, 100, "uniform", "rand"),
        (103, 60, 24, 200, "geom",    "flat"),
        (104, 50, 20, 80,  "edges",   "flat"),
        (105, 60, 28, 300, "geom",    "rand"),
        (106, 50, 24, 150, "uniform", "lowmode"),
        # harder / larger held-out instances
        (107, 70, 24, 500,  "geom",  "flat"),
        (108, 50, 20, 60,   "uniform", "rand"),
        (109, 64, 30, 250,  "edges", "rand"),
        (110, 80, 24, 400,  "geom",  "lowmode"),
        (111, 60, 26, 120,  "uniform", "flat"),
        (112, 50, 22, 1000, "geom",  "flat"),
    ]
    return [_build(*s) for s in specs]


# ----------------------------- physics / references ------------------------
def _gradnorm(curv, grad0, eta):
    """Final gradient L2 norm after running GD with schedule eta."""
    s = 0.0
    for i in range(len(curv)):
        f = 1.0
        li = curv[i]
        for e in eta:
            f *= (1.0 - e * li)
        v = grad0[i] * f
        s += v * v
    return math.sqrt(max(s, _TINY))


def _cheb_schedule(mu, L, N):
    """N step sizes = 1/root_k of the degree-N minimax residual polynomial on [mu,L]."""
    eta = []
    for k in range(1, N + 1):
        rho = (L + mu) / 2 - (L - mu) / 2 * math.cos((2 * k - 1) * math.pi / (2 * N))
        eta.append(1.0 / rho)
    return eta


# ----------------------------- validation ----------------------------------
def _validate_schedule(inst, answer):
    """Return list of N finite floats, or None if the answer is malformed."""
    if not isinstance(answer, dict):
        return None
    lr = answer.get("lr")
    if not isinstance(lr, list) or len(lr) != inst["n_steps"]:
        return None
    out = []
    for e in lr:
        if isinstance(e, bool) or not isinstance(e, (int, float)):
            return None
        e = float(e)
        if e != e or e in (float("inf"), float("-inf")):
            return None
        out.append(e)
    return out


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        curv = inst["curv"]
        grad0 = inst["grad0"]
        mu, L, N = inst["mu"], inst["L"], inst["n_steps"]

        base = _gradnorm(curv, grad0, [1.0 / L] * N)
        cheb = _gradnorm(curv, grad0, _cheb_schedule(mu, L, N))
        G_base = math.log10(base)
        G_cheb = math.log10(cheb)
        G_ideal = G_cheb - MARGIN
        denom = G_base - G_ideal
        if denom < 1e-6:
            denom = 1e-6

        public = {"name": inst["name"], "n_steps": N,
                  "curv": list(curv), "grad0": list(grad0)}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            eta = _validate_schedule(inst, ans)
        except Exception:
            eta = None
        if eta is None:
            vec.append(0.0)
            continue
        try:
            G_cand = math.log10(_gradnorm(curv, grad0, eta))
        except Exception:
            vec.append(0.0)
            continue

        r = 0.1 + 0.9 * (G_base - G_cand) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
