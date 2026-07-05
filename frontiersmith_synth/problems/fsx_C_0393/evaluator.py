#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_0393 -- "Alpine Relay: Convergence-Budget Saddle Descent"
(family: optimization-convergence-analysis; format B, eval_form flops).

THEME.  A mountain-rescue command post coordinates a chain of relay stations that pass a
casualty-triage estimate up a ridge (variables x, the ascent team's plan) while adverse
conditions push back (variables y, the storm/terrain adversary).  Balancing the two is a
convex-concave SADDLE problem: the rescue plan minimizes a cost that the environment
maximizes.  Each relay "leg" is ONE optimizer iteration -- fuel, radio, and daylight cap the
number of legs at a fixed budget T.  There is NO wall-clock scoring: the relay is graded ONLY
by how small the residual imbalance (the saddle operator norm) is after exactly T legs.

MATH.  For each instance we fix a monotone affine saddle operator
      F(z) = M z + q,     z = (x, y) in R^d,   d = nx + ny,
      M = [[ P ,  A ],            P = P^T >= 0   (cost is convex in x)
           [-A^T, Q ]],           Q = Q^T >= 0   (cost is concave in y)
coming from L(x,y) = 1/2 x^T P x + x^T A y - 1/2 y^T Q y + b^T x - c^T y, with
F = (grad_x L, -grad_y L).  The unique saddle point z* solves F(z*) = 0; we MINIMIZE the
final operator norm ||F(z_T)||_2 after T legs.

THE OPTIMIZER THE CANDIDATE DESIGNS.  Every leg follows one FROZEN "optimistic / momentum"
relay template with per-leg gains a[t], b[t] that the candidate supplies:
      g_t   = F(z_t)
      z_{t+1} = z_t - a[t]*g_t - b[t]*(g_t - g_{t-1})       (g_{-1} := g_0)
b[t]=0 recovers plain gradient descent-ascent; a[t]=b[t]=const recovers optimistic GDA;
a well-chosen SCHEDULE (Chebyshev-style acceleration, per-leg line search, ...) drives the
residual far lower under the same budget.  The candidate has the full (M, q, z0, T), so it may
simulate the trajectory itself and output any numeric schedule.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "nx": int, "ny": int, "d": int, "T": int,
             "M": [[float]*d]*d, "q": [float]*d, "z0": [float]*d,
             "baseline_step": float}        # the step used by the weak reference leg
  stdout: ONE JSON object:
            {"a": [float]*T, "b": [float]*T}   # per-leg relay gains

  VALID iff a and b are lists of exactly T finite real numbers with |value| <= 1e4.  Anything
  else -- wrong length, non-finite, |gain|>1e4, a crash, a timeout, non-JSON, or a schedule that
  makes the residual overflow to inf/nan -- scores 0.0 on that instance.

SCORING (deterministic; NO wall-time).  Per instance:
    q_base = ||F(z_T)|| from the weak reference: plain GDA (a=baseline_step, b=0) for T legs
    q_cand = ||F(z_T)|| from the candidate's schedule
    r = clamp( 0.1 + 0.9 * log10(q_base / q_cand) / K , 0, 1 ),   K = 2.5
  So reproducing the weak reference -> 0.1; reducing the residual by K=2.5 orders of magnitude
  under budget -> 1.0.  K is deliberately below what a good line-search reaches on hard,
  ill-conditioned, strongly-coupled instances, so excellent schedules stay strictly < 1.0
  (headroom); doing WORSE than the reference scores < 0.1.

ISOLATION.  The candidate is untrusted and runs OS-sandboxed via isorun.run_candidate; it only
sees the PUBLIC instance.  q_base, K, and the recomputation of q_cand happen in THIS parent
process, so introspection/frame-walking buys the candidate nothing -- the score is a pure
function of the residual the parent recomputes from the candidate's numbers.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import numpy as np
import isorun

K_ORDERS = 2.5          # target orders-of-magnitude reduction that maps to full score
GAIN_CAP = 1e4          # reject absurd gains outright


# ----------------------------- instance family -----------------------------
def _spd(rng, n, mu, L):
    """Symmetric PSD matrix with eigenvalues linearly spaced in [mu, L]."""
    Qm, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eig = np.linspace(mu, L, n)
    return (Qm * eig) @ Qm.T


def _build(seed, nx, ny, Lx, mux, Ly, muy, kA):
    """Deterministic monotone affine saddle operator F(z)=M z + q with start z0."""
    rng = np.random.default_rng(seed)
    P = _spd(rng, nx, mux, Lx)
    Q = _spd(rng, ny, muy, Ly)
    Araw = rng.standard_normal((nx, ny))
    Araw = Araw / np.linalg.norm(Araw, 2) * kA           # fix coupling strength
    d = nx + ny
    M = np.zeros((d, d))
    M[:nx, :nx] = P
    M[:nx, nx:] = Araw
    M[nx:, :nx] = -Araw.T
    M[nx:, nx:] = Q
    q = rng.standard_normal(d)
    z0 = np.ones(d)
    return M, q, z0


# (seed, nx, ny, Lx, mux, Ly, muy, kA): rising dimension / conditioning / coupling
_SPECS = [
    (101, 12, 12, 8.0, 0.40, 8.0, 0.40, 3.0),
    (102, 15, 15, 12.0, 0.30, 12.0, 0.30, 4.0),
    (103, 18, 14, 15.0, 0.25, 14.0, 0.25, 5.0),
    (104, 20, 20, 18.0, 0.20, 18.0, 0.20, 6.0),
    (205, 16, 16, 12.0, 0.20, 12.0, 0.20, 8.0),
    (206, 22, 18, 20.0, 0.15, 18.0, 0.15, 8.0),
    # harder / larger held-out relays (bigger d, worse conditioning, stronger coupling)
    (207, 25, 25, 28.0, 0.12, 26.0, 0.12, 11.0),
    (208, 24, 20, 22.0, 0.15, 20.0, 0.15, 10.0),
    (309, 28, 24, 30.0, 0.10, 28.0, 0.10, 12.0),
    (310, 30, 26, 35.0, 0.10, 32.0, 0.10, 14.0),
]


def _instances():
    out = []
    for (seed, nx, ny, Lx, mux, Ly, muy, kA) in _SPECS:
        M, q, z0 = _build(seed, nx, ny, Lx, mux, Ly, muy, kA)
        d = nx + ny
        T = max(8, d // 3)
        L = float(np.linalg.norm(M, 2))
        eta0 = 0.15 / L                       # deliberately conservative reference step
        out.append({"seed": seed, "nx": nx, "ny": ny, "d": d, "T": T,
                    "M": M, "q": q, "z0": z0, "eta0": eta0})
    return out


def _public(inst):
    return {"name": f"relay{inst['seed']}", "nx": inst["nx"], "ny": inst["ny"],
            "d": inst["d"], "T": inst["T"],
            "M": inst["M"].tolist(), "q": inst["q"].tolist(),
            "z0": inst["z0"].tolist(), "baseline_step": inst["eta0"]}


# ----------------------------- relay simulator -----------------------------
def _run(M, q, z0, a, b):
    """Run the frozen optimistic-relay template; return ||F(z_T)|| or None on blow-up."""
    z = z0.copy()
    g = M @ z + q
    gprev = g.copy()
    for t in range(len(a)):
        z = z - a[t] * g - b[t] * (g - gprev)
        gprev = g
        g = M @ z + q
        if not np.all(np.isfinite(g)):
            return None
    val = float(np.linalg.norm(g))
    if not math.isfinite(val):
        return None
    return val


# ----------------------------- answer validation ---------------------------
def _extract_schedule(inst, answer):
    """Validate candidate answer -> (a, b) float arrays of length T, or None."""
    if not isinstance(answer, dict):
        return None
    T = inst["T"]
    out = []
    for key in ("a", "b"):
        seq = answer.get(key)
        if not isinstance(seq, list) or len(seq) != T:
            return None
        arr = []
        for v in seq:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return None
            fv = float(v)
            if not math.isfinite(fv) or abs(fv) > GAIN_CAP:
                return None
            arr.append(fv)
        out.append(arr)
    return out[0], out[1]


# ----------------------------- scoring driver ------------------------------
def _score_one(inst, answer):
    sched = _extract_schedule(inst, answer)
    if sched is None:
        return 0.0
    a, b = sched
    M, q, z0 = inst["M"], inst["q"], inst["z0"]
    q_cand = _run(M, q, z0, a, b)
    if q_cand is None:
        return 0.0
    q_base = _run(M, q, z0, [inst["eta0"]] * inst["T"], [0.0] * inst["T"])
    if q_base is None or q_base <= 0.0:
        return 0.0
    q_cand = max(q_cand, 1e-15)
    r = 0.1 + 0.9 * math.log10(q_base / q_cand) / K_ORDERS
    if not math.isfinite(r):
        return 0.0
    return max(0.0, min(1.0, r))


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _instances()
    vec = []
    for inst in instances:
        ans, st = isorun.run_candidate(cand, _public(inst), timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            r = _score_one(inst, ans)
        except Exception:
            r = 0.0
        vec.append(r)
    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
