#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the silly-putty memory-kernel recovery task.

- Reads the test id from <in>'s header, then regenerates the hidden kernel
  (power law x small discrete-scale correction) EXACTLY as gen.py does --
  the ground truth lives only here and in gen.py, never in the training data.
- Parses the participant's single-line kernel expression G(t) (arithmetic
  over + - * / ** parentheses, numeric constants, the unary functions
  exp/log/sqrt/sin/cos/absv, and the single variable t).
- Builds FOUR held-out strain histories purely from the training window
  bounds [t_min, t_max] read out of <in> (never from hidden params):
    * osc_fast  -- oscillation much FASTER than the observed window
    * osc_slow  -- oscillation much SLOWER than the observed window
    * osc_mid   -- oscillation inside the observed window (sanity probe)
    * ramp      -- a strain ramp run far longer than the observed window
  For each, the true and predicted stress are obtained by EXACT numerical
  convolution (fixed midpoint quadrature, deterministic) of the respective
  kernel against the strain-rate history, then compared by a normalised RMS
  error. Errors are combined in a fixed weighted average (extrapolation
  probes weighted most) plus a small parsimony penalty on expression size.
- F = weighted_error(submitted) + LAMBDA * max(0, nodes - 3)
  B = weighted_error(single-relaxation-time baseline fit)      [internal]
  Ratio = min(1000, 100*B / F) / 1000
A flat/constant-ish guess reproduces roughly the baseline (~0.1). A kernel
that gets the FUNCTIONAL FORM right (not just the observed-window fit)
predicts all four held-out histories well and scores much higher, but the
planted discrete-scale correction and measurement noise keep even a perfect
power-law fit short of the ceiling.
"""
import sys
import math
import ast
import random

import numpy as np

LAMBDA = 0.006
N_QUAD = 1500
MAX_OUT_BYTES = 4000
MAX_EXPR_LEN = 300
MAX_NODES = 40
WEIGHTS = {"osc_fast": 0.05, "osc_slow": 0.30, "osc_mid": 0.25, "ramp": 0.40}

ALLOWED_FUNCS = {
    "exp": lambda x: np.exp(np.clip(x, -700.0, 700.0)),
    "log": lambda x: np.log(x),
    "sqrt": lambda x: np.sqrt(x),
    "sin": lambda x: np.sin(x),
    "cos": lambda x: np.cos(x),
    "absv": lambda x: np.abs(x),
}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden kernel (identical to gen.py) ----------
def truth(t):
    rng = random.Random(20260726 + 97 * t)
    alpha_true = round(rng.uniform(0.15, 0.85), 4)
    A_true = round(rng.uniform(0.6, 4.0), 4)
    window_choices = [(0.1, 10.0), (1.0, 100.0), (0.05, 5.0), (2.0, 200.0), (0.5, 50.0)]
    t_min, t_max = window_choices[t % len(window_choices)]
    noise_amp = 0.01 + 0.0025 * (t - 1)
    delta = round(rng.uniform(0.125, 0.165), 4)
    plog = round(rng.uniform(0.9, 1.6), 4)
    phase = round(rng.uniform(0.0, 2 * math.pi), 4)
    return alpha_true, A_true, t_min, t_max, noise_amp, delta, plog, phase


def kernel_true_fn(alpha, A, delta, plog, phase):
    wlog = 2.0 * math.pi / (plog * math.log(10.0))

    def f(u):
        u = np.asarray(u, dtype=float)
        base = A * np.power(u, -alpha)
        corr = 1.0 + delta * np.sin(wlog * np.log(u) + phase)
        return base * corr
    return f


# ---------- held-out history design (derived only from t_min, t_max) ----------
def held_out_configs(t_min, t_max):
    p_fast = t_min / 8.0
    p_slow = t_max * 8.0
    p_mid = math.sqrt(t_min * t_max)
    cfgs = []
    for name, p in [("osc_fast", p_fast), ("osc_slow", p_slow), ("osc_mid", p_mid)]:
        t_evals = [(4.0 + 0.25 * i) * p for i in range(8)]
        cfgs.append((name, "osc", p, t_evals))
    t_evals_ramp = [(4.0 + 0.25 * i) * t_max for i in range(8)]
    cfgs.append(("ramp", "ramp", None, t_evals_ramp))
    return cfgs


def stress_series(kernel_fn, hist_type, p, t_evals):
    out = []
    for t_eval in t_evals:
        h = t_eval / N_QUAD
        idx = np.arange(N_QUAD)
        s_mid = (idx + 0.5) * h
        u = t_eval - s_mid          # always > 0
        gv = kernel_fn(u)
        if not np.all(np.isfinite(gv)):
            return None
        if hist_type == "osc":
            omega = 2.0 * math.pi / p
            gdot = omega * np.cos(omega * s_mid)
        else:
            gdot = np.ones_like(s_mid)
        out.append(h * float(np.sum(gv * gdot)))
    return np.array(out)


def rel_rms(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)) / (np.sqrt(np.mean(true ** 2)) + 1e-9))


def weighted_error(kernel_fn, t_min, t_max, truevals):
    cfgs = held_out_configs(t_min, t_max)
    total = 0.0
    for name, htype, p, t_evals in cfgs:
        pred = stress_series(kernel_fn, htype, p, t_evals)
        if pred is None or not np.all(np.isfinite(pred)):
            return None
        total += WEIGHTS[name] * rel_rms(pred, truevals[name])
    return total


def baseline_kernel_fn(rows):
    """Internal trivial construction: single fixed relaxation time (geometric
    mean of the observed lags), amplitude by 1-D least squares. Never sees
    the hidden alpha/A."""
    ts = [r[1] for r in rows]
    tau = math.sqrt(min(ts) * max(ts))
    num = 0.0
    den = 0.0
    for g0, t, s in rows:
        x = math.exp(-t / tau)
        y = s / g0
        num += x * y
        den += x * x
    a = num / den if den > 1e-12 else 0.0

    def f(u):
        u = np.asarray(u, dtype=float)
        return a * np.exp(-u / tau)
    return f


# ---------- safe expression parsing ----------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
)


def _count_nodes(tree):
    return sum(1 for nd in ast.walk(tree)
               if isinstance(nd, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant)))


def parse_expression(raw):
    if len(raw.encode("utf-8", "replace")) > MAX_OUT_BYTES:
        fail("output too large")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if len(lines) != 1:
        fail("expected exactly one non-empty output line")
    text = lines[0]
    if len(text) > MAX_EXPR_LEN:
        fail("expression too long")
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("parse error")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            fail("disallowed syntax %s" % type(node).__name__)
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCS):
                fail("disallowed function call")
            if node.keywords or len(node.args) != 1:
                fail("bad function arity")
        if isinstance(node, ast.Name) and node.id not in ALLOWED_FUNCS:
            if node.id != "t":
                fail("unknown name %s" % node.id)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                fail("non-numeric constant")
            v = float(node.value)
            if v != v or v in (float("inf"), float("-inf")):
                fail("non-finite constant")
    nodes = _count_nodes(tree)
    if nodes > MAX_NODES:
        fail("expression too large (%d nodes)" % nodes)
    try:
        code = compile(tree, "<kernel>", "eval")
    except Exception:
        fail("compile error")
    return code, nodes


def make_submitted_kernel_fn(code):
    def f(u):
        env = dict(ALLOWED_FUNCS)
        env["t"] = np.asarray(u, dtype=float)
        try:
            val = eval(code, {"__builtins__": {}}, env)
        except Exception:
            return None
        val = np.asarray(val, dtype=float)
        if val.shape == ():
            val = np.full_like(env["t"], float(val))
        return val
    return f


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            lines_in = fh.read().splitlines()
        n_rows, t = (int(x) for x in lines_in[0].split())
        rows = []
        for ln in lines_in[1:1 + n_rows]:
            g0, tt, sig = (float(x) for x in ln.split())
            rows.append((g0, tt, sig))
        if len(rows) != n_rows or n_rows < 4:
            raise ValueError
    except Exception:
        fail("bad instance file")

    if t < 1 or t > 100000:
        fail("bad test id")

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    text = raw.decode("utf-8", "replace")

    code, nodes = parse_expression(text)
    submitted_fn = make_submitted_kernel_fn(code)

    ts = [r[1] for r in rows]
    t_min_obs, t_max_obs = min(ts), max(ts)

    # regenerate hidden truth (only place the real kernel exists)
    alpha_true, A_true, t_min, t_max, noise_amp, delta, plog, phase = truth(t)
    true_fn = kernel_true_fn(alpha_true, A_true, delta, plog, phase)

    cfgs = held_out_configs(t_min_obs, t_max_obs)
    truevals = {}
    for name, htype, p, t_evals in cfgs:
        truevals[name] = stress_series(true_fn, htype, p, t_evals)

    f_err = weighted_error(submitted_fn, t_min_obs, t_max_obs, truevals)
    if f_err is None:
        fail("non-finite prediction")

    baseline_fn = baseline_kernel_fn(rows)
    b_err = weighted_error(baseline_fn, t_min_obs, t_max_obs, truevals)
    if b_err is None or b_err <= 1e-9:
        fail("internal baseline degenerate")

    f_total = f_err + LAMBDA * max(0, nodes - 3)
    b_total = b_err + LAMBDA * max(0, 1 - 3)

    sc = min(1000.0, 100.0 * b_total / max(1e-9, f_total))
    print("held_out_err=%.6f baseline_err=%.6f nodes=%d  Ratio: %.6f"
          % (f_err, b_err, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
