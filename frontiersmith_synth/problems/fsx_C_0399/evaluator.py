#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_0399 -- "Rooftop Garden Equilibrium: Blind Saddle Schedule"
(family: optimization-convergence-analysis; format B, eval_form flops).

THEME.  A city runs a network of interconnected ROOFTOP GARDENS.  A horticulture
controller sets continuous action levels  x  (irrigation, shade, nutrient dosing per
plot) to keep the gardens healthy, while an adversarial environment  y  (heat waves,
pests, wind stress) pushes them out of balance.  Their conflict is a convex-concave
saddle game

    min_x  max_y   f(x,y) = 1/2 x^T P x  +  x^T A y  -  1/2 y^T Q y  +  b^T x  -  c^T y

with P, Q symmetric positive semidefinite (the controller's cost is convex in x, the
environment's payoff is concave in y) and A the cross-coupling between the two.  The
system is "at rest" (a saddle equilibrium) exactly when the joint disequilibrium
vector

    F(z) = [ grad_x f ;  -grad_y f ] = M z + q ,   z = (x, y),
    M = [[ P ,  A ],
         [ -A^T, Q ]] ,           q = [ b ; c ]

vanishes.  ||F(z)|| is the residual imbalance of the whole rooftop network.

THE TASK (blind, op-budgeted schedule design).  Starting from z0 the controller may
apply exactly T update steps.  At step k the controller pays for ONE disequilibrium
probe g_k = F(z_k) (one operator application -- the "flop budget") and then moves

    z_{k+1} = z_k - a_k * g_k + m_k * (z_k - z_{k-1}) - o_k * (g_k - g_{k-1})

    (z_{-1} := z0 ,  g_{-1} := g_0)

The scalar schedule (a_k step, m_k heavy-ball momentum, o_k optimistic/negative-probe
correction) is fixed IN ADVANCE for all T steps.  Special cases: (m=o=0) = plain
gradient descent-ascent; (o=a, m=0) = optimistic GDA; (m>0) = heavy-ball acceleration.
Because the coefficients are committed up front, the iterate residual is exactly a
degree-<=T matrix polynomial in M applied to g_0, so this is a pure
convergence-schedule-design problem: pick the schedule that drives the FINAL imbalance
||F(z_T)|| as low as possible within the T-step budget.  Lower is better (minimize).

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "dx": int, "dy": int, "d": int, "T": int,
             "M": [[...]] (d x d, the saddle operator),
             "q": [...]   (length d),
             "z0": [...]  (length d, start iterate),
             "P":..., "Q":..., "A":..., "b":..., "c":...}   # theme blocks (M,q authoritative)
  stdout: ONE JSON object:
            {"a": [T floats], "m": [T floats] (optional), "o": [T floats] (optional)}
          "a" REQUIRED (length T, finite).  "m","o" optional, default all zeros;
          if present each must be length T and finite.

  Invalid output, wrong length, non-finite coefficients, a crash, a timeout, or a
  schedule that makes the iterate blow up to inf/nan -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  Per instance the evaluator computes:
    base_res = ||F(z0)||                      # "do nothing" reference  -> normalized 0.1
    cand_res = ||F(z_T)|| from the candidate schedule
    ref_res  = the OPTIMAL degree-T residual  = min over degree-<=T polynomials p with
               p(0)=1 of ||p(M) g_0||, computed by truncated GMRES(T) in THIS process.
               This is the information-theoretic floor for ANY committed T-step schedule
               (our update family is a subset of these polynomials), so no candidate can
               beat it -> the r=1.0 anchor is unreachable, guaranteeing headroom.
  Normalize affinely in log-residual (residuals decay geometrically):
    r = clamp( 0.1 + 0.9 * (ln base_res - ln cand_res) / (ln base_res - ln ref_res), 0, 1 )
  do-nothing -> 0.1 ; matching the (unreachable) GMRES optimum -> 1.0 ; worse than doing
  nothing (an unstable/divergent schedule) -> < 0.1, floored at 0.

  Because the budget is strictly smaller than the dimension (T < d), a degree-T
  polynomial cannot annihilate all d modes -> even the optimal schedule leaves residual
  well above 0, so strong hand-tuned schedules stay comfortably below 1.0.

ISOLATION.  The candidate is untrusted and runs in a FRESH bwrap SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  The references
(base_res, GMRES optimum) are computed by THIS parent process, so a frame-walking /
introspecting candidate learns nothing it did not already receive.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import numpy as np
import isorun


# ----------------------------- instance family -----------------------------
def _spd(rng, n, kappa):
    """Random symmetric positive-definite n x n with eigenvalues log-spread in [1, kappa]."""
    # random orthogonal via QR of a gaussian
    G = rng.standard_normal((n, n))
    Qm, _ = np.linalg.qr(G)
    if n == 1:
        eigs = np.array([1.0])
    else:
        t = np.linspace(0.0, 1.0, n)
        eigs = np.exp(t * math.log(kappa))          # 1 .. kappa, log-spaced
    return (Qm * eigs) @ Qm.T


def _build_instance(seed, dx, dy, T, kappa, rot):
    rng = np.random.default_rng(seed)
    P = _spd(rng, dx, kappa)
    Q = _spd(rng, dy, kappa)
    A = rot * rng.standard_normal((dx, dy)) / math.sqrt(max(dx, dy))
    b = rng.standard_normal(dx)
    c = rng.standard_normal(dy)
    d = dx + dy
    M = np.zeros((d, d))
    M[:dx, :dx] = P
    M[:dx, dx:] = A
    M[dx:, :dx] = -A.T
    M[dx:, dx:] = Q
    q = np.concatenate([b, c])
    z0 = rng.standard_normal(d)
    return {"name": f"rooftop{seed}", "dx": dx, "dy": dy, "d": d, "T": T,
            "M": M, "q": q, "z0": z0, "P": P, "Q": Q, "A": A, "b": b, "c": c}


def _specs():
    #  (seed, dx, dy, T, kappa, rot)
    return [
        (101,  8,  8,  6,  8.0,  0.6),   # easy: well-conditioned, mild rotation
        (102, 10, 10,  8, 12.0,  0.8),
        (103, 10, 10,  8, 20.0,  1.2),   # rotation-heavy
        (104, 12, 12, 10, 15.0,  0.9),
        (105,  9,  9,  7, 30.0,  0.7),   # ill-conditioned
        (106, 12, 12, 10, 25.0,  1.6),   # ill-conditioned + rotation
        (107, 11, 11,  8, 10.0,  1.4),
        (108, 14, 14, 10, 18.0,  1.0),
        # harder / held-out
        (211, 13, 13, 10, 40.0,  1.2),   # very ill-conditioned
        (212, 12, 12,  8, 22.0,  2.2),   # strong rotation, tight budget
        (213, 15, 15, 11, 28.0,  1.5),
        (214, 11, 11,  7, 35.0,  1.8),   # tight budget, hard both ways
    ]


def _build_instances():
    return [_build_instance(*s) for s in _specs()]


# ----------------------------- references ----------------------------------
def _base_res(M, q, z0):
    return float(np.linalg.norm(M @ z0 + q))


def _gmres_opt_res(M, q, z0, T):
    """Optimal residual achievable by a degree-<=T polynomial with p(0)=1 applied to
    g0 = M z0 + q, i.e. truncated GMRES(T) residual for solving M x = -q from x0=z0.
    This is a lower bound for ANY committed T-step schedule in the candidate's family."""
    r0 = -(M @ z0 + q)
    beta = float(np.linalg.norm(r0))
    if beta < 1e-300:
        return beta
    d = M.shape[0]
    m = min(T, d)
    V = np.zeros((d, m + 1))
    H = np.zeros((m + 1, m))
    V[:, 0] = r0 / beta
    keff = m
    for j in range(m):
        w = M @ V[:, j]
        for i in range(j + 1):
            H[i, j] = V[:, i] @ w
            w = w - H[i, j] * V[:, i]
        h = float(np.linalg.norm(w))
        H[j + 1, j] = h
        if h < 1e-12:                    # happy breakdown -> exact within j+1 dims
            keff = j + 1
            break
        V[:, j + 1] = w / h
    e1 = np.zeros(keff + 1)
    e1[0] = beta
    Hk = H[:keff + 1, :keff]
    y, *_ = np.linalg.lstsq(Hk, e1, rcond=None)
    res = float(np.linalg.norm(Hk @ y - e1))
    return res


# ----------------------------- schedule simulation -------------------------
def _run_schedule(M, q, z0, T, a, m, o):
    """Run the committed T-step update. Return final residual or None if it blows up."""
    z = z0.astype(float).copy()
    z_prev = z0.astype(float).copy()
    g_prev = M @ z + q
    for k in range(T):
        g = M @ z + q
        z_next = z - a[k] * g + m[k] * (z - z_prev) - o[k] * (g - g_prev)
        z_prev = z
        g_prev = g
        z = z_next
        if not np.all(np.isfinite(z)):
            return None
    res_vec = M @ z + q
    if not np.all(np.isfinite(res_vec)):
        return None
    return float(np.linalg.norm(res_vec))


def _parse_schedule(answer, T):
    """Validate candidate answer; return (a, m, o) numpy arrays or None."""
    if not isinstance(answer, dict):
        return None
    a = answer.get("a")
    if not isinstance(a, list) or len(a) != T:
        return None

    def _vec(v, name):
        if v is None:
            return np.zeros(T)
        if not isinstance(v, list) or len(v) != T:
            return "BAD"
        out = np.zeros(T)
        for i, x in enumerate(v):
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                return "BAD"
            xf = float(x)
            if not math.isfinite(xf):
                return "BAD"
            out[i] = xf
        return out

    av = _vec(a, "a")
    mv = _vec(answer.get("m"), "m")
    ov = _vec(answer.get("o"), "o")
    if isinstance(av, str) or isinstance(mv, str) or isinstance(ov, str):
        return None
    return av, mv, ov


# ----------------------------- scoring driver ------------------------------
def _public(inst):
    return {"name": inst["name"], "dx": inst["dx"], "dy": inst["dy"], "d": inst["d"],
            "T": inst["T"], "M": inst["M"].tolist(), "q": inst["q"].tolist(),
            "z0": inst["z0"].tolist(), "P": inst["P"].tolist(), "Q": inst["Q"].tolist(),
            "A": inst["A"].tolist(), "b": inst["b"].tolist(), "c": inst["c"].tolist()}


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        M, q, z0, T = inst["M"], inst["q"], inst["z0"], inst["T"]
        base_res = _base_res(M, q, z0)
        ref_res = _gmres_opt_res(M, q, z0, T)
        ref_res = max(ref_res, base_res * 1e-10)     # floor to keep log finite
        L_base = math.log(max(base_res, 1e-300))
        L_ref = math.log(max(ref_res, 1e-300))
        denom = L_base - L_ref
        if denom < 1e-9:
            denom = 1e-9

        ans, st = isorun.run_candidate(cand, _public(inst), timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        parsed = _parse_schedule(ans, T)
        if parsed is None:
            vec.append(0.0)
            continue
        a, m, o = parsed
        cand_res = _run_schedule(M, q, z0, T, a, m, o)
        if cand_res is None or not math.isfinite(cand_res) or cand_res <= 0.0:
            # cand_res==0 impossible for T<d; treat as reject if it happens
            if cand_res is None or not math.isfinite(cand_res):
                vec.append(0.0)
                continue
            cand_res = max(cand_res, 1e-300)
        L_cand = math.log(max(cand_res, 1e-300))
        r = 0.1 + 0.9 * (L_base - L_cand) / denom
        if not math.isfinite(r):
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
