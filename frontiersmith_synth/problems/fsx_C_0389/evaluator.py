#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_0389 -- "Solar-Farm Inverter Bench: Design a Drop-in
Activation for the Inverter-Response Surrogate"
(family: modular-component-cross-setting; format B, quality-metric).

THEME.  A solar-farm operator fits cheap CPU surrogate models that map a handful of
inverter telemetry channels (irradiance, cell temperature, DC voltage/current, phase,
etc.) onto a scalar response of the inverter -- efficiency, clipped AC power, thermal
derating, MPP-tracking loss, harmonic distortion.  Each response obeys a DIFFERENT
nonlinear physical law, so a single linear model is hopeless.  The surrogate is a
fixed random-feature model (an "extreme learning machine"): the raw channels are
standardized, hit with a FROZEN random projection, passed through ONE elementwise
activation g(.), the resulting features are standardized, and a ridge head is solved
in closed form.  The ONLY thing the operator gets to design is the activation g --
the atomic, transferable component shared across every surrogate on the farm.

THE TASK (modular component, cross setting).  Invent ONE drop-in scalar activation
g(x) that works well as the surrogate nonlinearity ACROSS several inverter settings
at once, PLUS a held-out setting the designer never sees.  g is specified as a small
sum of whitelisted basis functions:

    g(x) = sum_k  w_k * base_k( a_k * x + b_k )

The candidate never sees any telemetry, any target, the random projection, or the
held-out setting -- it designs g blind, purely as a general-purpose nonlinearity, the
way one invents ReLU / GELU / Swish rather than tuning to one dataset.  There is no
single dominant answer: saturating (tanh/sigmoid), rectifying (relu/swish/softplus),
even/localized (abs/gauss) and periodic (sin) shapes each help some settings and hurt
others, and the held-out setting rewards designs that generalize instead of overfitting
the visible mix.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str,
             "input_dim": D,            # raw telemetry channels fed to the surrogate
             "hidden_units": H,         # width of the frozen random-feature projection
             "ridge_lambda": float,     # L2 on the closed-form ridge head
             "n_public_settings": int,  # inverter responses in the visible mix
             "n_hidden_settings": int,  # additional held-out response(s), never revealed
             "activation_bases": [ ... names ... ],   # the whitelist you may combine
             "spec": str}               # human-readable reminder of g's form
  stdout: ONE JSON object:
            {"components": [ {"base": <name>, "a": <float>, "b": <float>, "w": <float>},
                             ... ]}       # 1..8 entries, defining g(x) above

  VALID iff `components` is a list of 1..8 dicts; each has base in the whitelist and
  finite floats a,b,w with |a|<=10, |b|<=10, |w|<=10 (bools rejected).  Wrong shape,
  unknown base, out-of-range / non-finite number, a crash, a timeout, or non-JSON ->
  that whole run scores 0.0 on every instance.

SCORING (deterministic; no wall-time).  For each instance we hold >=3 public inverter
settings plus >=1 HELD-OUT setting.  For a setting we standardize the D channels on the
train split, apply the FROZEN random projection (seeded per setting), apply g, standardize
the H feature columns (train stats), append an intercept, and solve ridge in closed form.
The per-setting metric is normalized test MSE:  nmse = mean((y-yhat)^2) / var(y).  The
per-instance objective is the GEOMETRIC MEAN of nmse over ALL settings (public + held-out)
-- lower is better.  We normalize against the evaluator's own trivial component:
    b     = per-instance gmean-nmse using the IDENTITY activation g(x)=x (a linear head).
    obj   = per-instance gmean-nmse using the candidate's g.
    r     = min(1, 0.1 * b / max(obj, 1e-12))
Reproducing the linear baseline scores ~0.1; a worse-than-linear g scores <0.1; a good
nonlinearity that fits every response scores higher, and the irreducible noise floor keeps
even excellent designs below 1.0 -> headroom.

ISOLATION.  The candidate is untrusted and runs OS-sandboxed in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  All telemetry, targets,
random projections, the held-out setting, the baseline and every validation live ONLY in
this parent process, so a frame-walking / introspecting candidate learns nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import os
# Force single-threaded BLAS BEFORE importing numpy so the closed-form solves are
# bit-reproducible across re-runs (the harness re-runs and compares Ratio+Vector).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import sys, json
import numpy as np
import isorun

# ------------------------------- configuration -----------------------------
D = 6            # telemetry channels
H = 50           # random-feature width
N = 400          # train == test size per setting
NOISE = 0.15     # observation noise std (sets the headroom / irreducible floor)
LAM = 0.5        # ridge regularization on the feature head
TEACHER_UNITS = 4
BASE_SCALE = 0.6 # scale of the random projection bias

WHITELIST = ["identity", "relu", "tanh", "sigmoid", "gauss",
             "sin", "abs", "swish", "softplus", "cube"]

# public inverter responses (visible mix) and the held-out pool for the hidden setting
PUBLIC_KINDS = ["efficiency", "clip", "derate", "mppt"]
HIDDEN_POOL = ["efficiency", "clip", "derate", "mppt", "harmonic"]


# ------------------------------- basis functions ---------------------------
def _base_eval(name, u):
    u = np.clip(u, -20.0, 20.0)   # keep exp/cube finite regardless of a,b
    if name == "identity":  return u
    if name == "relu":      return np.maximum(0.0, u)
    if name == "tanh":      return np.tanh(u)
    if name == "sigmoid":   return 1.0 / (1.0 + np.exp(-u))
    if name == "gauss":     return np.exp(-u * u)
    if name == "sin":       return np.sin(u)
    if name == "abs":       return np.abs(u)
    if name == "swish":     return u / (1.0 + np.exp(-u))
    if name == "softplus":  return np.log1p(np.exp(u))
    if name == "cube":      return u ** 3
    raise ValueError("bad base")


def _apply_activation(components, t):
    out = np.zeros_like(t)
    for c in components:
        out = out + c["w"] * _base_eval(c["base"], c["a"] * t + c["b"])
    return out


# ------------------------------- teacher (hidden physics) ------------------
def _teacher(kind, u):
    if kind == "efficiency": return np.exp(-((u - 0.4) ** 2))   # bell efficiency curve
    if kind == "clip":       return np.abs(u)                    # rectified clip magnitude
    if kind == "derate":     return -np.maximum(0.0, u - 0.3)    # thermal derating kink
    if kind == "mppt":       return -(u * u)                     # MPP-tracking loss
    if kind == "harmonic":   return np.sin(2.3 * u)              # harmonic distortion
    raise ValueError("bad kind")


def _make_setting(seed, kind):
    """Deterministic train/test split for one inverter response (a teacher net)."""
    rng = np.random.RandomState(seed)
    X = rng.randn(2 * N, D)
    U = rng.randn(D, TEACHER_UNITS) / np.sqrt(D)
    cf = rng.randn(TEACHER_UNITS)
    Z = X @ U
    y = np.zeros(2 * N)
    for j in range(TEACHER_UNITS):
        y += cf[j] * _teacher(kind, Z[:, j])
    y = y / (y.std() + 1e-9)
    y = y + NOISE * rng.randn(2 * N)
    return X[:N], y[:N], X[N:], y[N:]


# ------------------------------- surrogate nMSE ----------------------------
def _setting_nmse(components, Xtr, ytr, Xte, yte, proj_seed):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-9
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    rng = np.random.RandomState(proj_seed)
    W = rng.randn(D, H)
    b = rng.randn(H) * BASE_SCALE
    Ptr = _apply_activation(components, Xtr @ W + b)
    Pte = _apply_activation(components, Xte @ W + b)
    if not (np.all(np.isfinite(Ptr)) and np.all(np.isfinite(Pte))):
        return 1e6
    pmu = Ptr.mean(0); psd = Ptr.std(0)
    psd = np.where(psd < 1e-9, 1.0, psd)
    Ptr = (Ptr - pmu) / psd
    Pte = (Pte - pmu) / psd
    Ptr = np.hstack([Ptr, np.ones((Ptr.shape[0], 1))])
    Pte = np.hstack([Pte, np.ones((Pte.shape[0], 1))])
    A = Ptr.T @ Ptr + LAM * np.eye(Ptr.shape[1])
    try:
        beta = np.linalg.solve(A, Ptr.T @ ytr)
    except Exception:
        return 1e6
    pred = Pte @ beta
    nmse = float(np.mean((yte - pred) ** 2) / (np.var(yte) + 1e-12))
    if not np.isfinite(nmse):
        return 1e6
    return nmse


def _gmean(vals):
    v = np.clip(np.asarray(vals, dtype=float), 1e-9, None)
    return float(np.exp(np.mean(np.log(v))))


# ------------------------------- instance family ---------------------------
def _build_instances():
    insts = []
    for iseed in range(8):
        rng = np.random.RandomState(1000 + iseed)
        pubs = list(rng.choice(PUBLIC_KINDS, 3, replace=False))
        hidden = str(rng.choice(HIDDEN_POOL))
        settings = pubs + [hidden]      # 3 public + 1 held-out
        # per-setting data seeds and projection seeds (deterministic, distinct)
        data_seeds = [2000 + iseed * 10 + si for si in range(len(settings))]
        proj_seeds = [3000 + iseed * 10 + si for si in range(len(settings))]
        insts.append({
            "name": f"farm{iseed:02d}",
            "settings": settings,           # HELD IN PARENT ONLY (kinds are hidden)
            "data_seeds": data_seeds,
            "proj_seeds": proj_seeds,
            "n_public": 3,
        })
    return insts


def _public_view(inst):
    return {
        "name": inst["name"],
        "input_dim": D,
        "hidden_units": H,
        "ridge_lambda": LAM,
        "n_public_settings": inst["n_public"],
        "n_hidden_settings": len(inst["settings"]) - inst["n_public"],
        "activation_bases": list(WHITELIST),
        "spec": "g(x)=sum_k w_k*base_k(a_k*x+b_k); return {'components':[{base,a,b,w},...]}, 1..8 entries.",
    }


# ------------------------------- validation --------------------------------
def _validate(answer):
    if not isinstance(answer, dict):
        return None
    comps = answer.get("components")
    if not isinstance(comps, list) or not (1 <= len(comps) <= 8):
        return None
    out = []
    for c in comps:
        if not isinstance(c, dict):
            return None
        base = c.get("base")
        if base not in WHITELIST:
            return None
        vals = {}
        for key, lim in (("a", 10.0), ("b", 10.0), ("w", 10.0)):
            v = c.get(key)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return None
            v = float(v)
            if not np.isfinite(v) or abs(v) > lim + 1e-9:
                return None
            vals[key] = v
        out.append({"base": base, "a": vals["a"], "b": vals["b"], "w": vals["w"]})
    return out


# ------------------------------- objective ---------------------------------
def _instance_objective(components, inst):
    nmses = []
    for kind, dseed, pseed in zip(inst["settings"], inst["data_seeds"], inst["proj_seeds"]):
        Xtr, ytr, Xte, yte = _make_setting(dseed, kind)
        nmses.append(_setting_nmse(components, Xtr, ytr, Xte, yte, pseed))
    return _gmean(nmses)


IDENTITY = [{"base": "identity", "a": 1.0, "b": 0.0, "w": 1.0}]


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        b_obj = _instance_objective(IDENTITY, inst)   # linear baseline, parent-only
        ans, st = isorun.run_candidate(cand, _public_view(inst), timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            comps = _validate(ans)
        except Exception:
            comps = None
        if comps is None:
            vec.append(0.0)
            continue
        try:
            obj = _instance_objective(comps, inst)
        except Exception:
            vec.append(0.0)
            continue
        if not np.isfinite(obj) or obj <= 0:
            vec.append(0.0)
            continue
        r = min(1.0, 0.1 * b_obj / max(obj, 1e-12))
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        vec.append(max(0.0, min(1.0, r)))

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
