#!/usr/bin/env python3
"""Deterministic checker for fsx_G_0418 (Format E, scientific-law-extrapolation).

Usage: python3 verify.py <in> <out> <ans>
  <in>  : the train sample (as printed by gen.py)  -- rows the solver saw
  <out> : the participant artifact -- ONE closed-form expression over {T, C}
  <ans> : ignored (empty placeholder)

The checker regenerates the HELD-OUT EXTRAPOLATION split (hot, concentrated operating
points, far above any bench row) from the hidden Arrhenius law with a FIXED seed, then:
  * strictly validates the submitted expression (safe AST whitelist, finite reals),
  * scores it by relative RMSE on the held-out split (+ a mild complexity penalty),
  * normalises against an internal constant-mean baseline B.

Any violation -> `Ratio: 0.0`. Deterministic and O(size).
"""
import sys, ast, math, random

# ---- hidden ground-truth Arrhenius law (identical to gen.py; server-side only) ----
A_PRE, E_ACT, ORD, M_EXP = 2.0e5, 6500.0, 0.7, 1.0
def true_k(T, C):
    return A_PRE * (T ** M_EXP) * math.exp(-E_ACT / T) * (C ** ORD)

# ---- held-out EXTRAPOLATION split: HOTTER and MORE CONCENTRATED than any train row ----
HELD_T = [520.0, 560.0, 600.0, 640.0, 680.0, 720.0]   # train T max was 440
HELD_C = [5.0, 6.0, 7.0, 8.0]                          # train C max was 4.0
HELD_SEED = 90218
HELD_NOISE = 0.12

MAX_EXPR_CHARS = 5000
MAX_NODES = 500

ALLOWED_FUNCS = {
    'exp': math.exp, 'log': math.log, 'sqrt': math.sqrt,
    'sin': math.sin, 'cos': math.cos, 'tan': math.tan, 'tanh': math.tanh,
    'abs': abs, 'pow': pow, 'log2': math.log2, 'log10': math.log10,
}
ALLOWED_CONSTS = {'pi': math.pi, 'e': math.e}
VARS = {'T', 'C'}

ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Name, ast.Load, ast.Call,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.USub, ast.UAdd,
)


def fail(msg):
    sys.stdout.write("reason: %s\n" % msg)
    sys.stdout.write("Ratio: 0.0\n")
    sys.exit(0)


def gen_heldout():
    rng = random.Random(HELD_SEED)
    pts = []
    for T in HELD_T:
        for C in HELD_C:
            k = true_k(T, C) * (1.0 + rng.uniform(-HELD_NOISE, HELD_NOISE))
            pts.append((float(T), float(C), k))
    return pts


def read_train_mean(path):
    try:
        toks = open(path).read().split()
        M = int(toks[0])
        ks = [float(toks[3 * i + 3]) for i in range(M)]
        if not ks:
            return None
        return sum(ks) / len(ks)
    except Exception:
        return None


def validate_ast(tree):
    n_nodes = 0
    for node in ast.walk(tree):
        n_nodes += 1
        if n_nodes > MAX_NODES:
            return None, "expression too large"
        if not isinstance(node, ALLOWED_NODES):
            return None, "disallowed syntax: %s" % type(node).__name__
        if isinstance(node, ast.Name):
            if node.id not in VARS and node.id not in ALLOWED_FUNCS \
                    and node.id not in ALLOWED_CONSTS:
                return None, "unknown name: %s" % node.id
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCS:
                return None, "disallowed call"
            if node.keywords:
                return None, "keyword args not allowed"
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                return None, "non-numeric constant"
    return n_nodes, None


def rrmse(preds, obs):
    return math.sqrt(sum(((p - o) / o) ** 2 for p, o in zip(preds, obs)) / len(obs))


def main():
    if len(sys.argv) < 3:
        fail("usage")
    in_path, out_path = sys.argv[1], sys.argv[2]

    # ---- read participant expression: first non-empty line ----
    try:
        raw = open(out_path).read()
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_EXPR_CHARS:
        fail("output too long")
    expr = None
    for line in raw.splitlines():
        if line.strip():
            expr = line.strip()
            break
    if expr is None:
        fail("empty output")

    # ---- parse + whitelist ----
    try:
        tree = ast.parse(expr, mode="eval")
    except Exception as ex:
        fail("parse error: %s" % ex)
    n_nodes, err = validate_ast(tree)
    if err:
        fail(err)
    try:
        code = compile(tree, "<expr>", "eval")
    except Exception as ex:
        fail("compile error: %s" % ex)

    # ---- held-out extrapolation split + baseline ----
    held = gen_heldout()
    obs = [k for (_, _, k) in held]
    mean_k = read_train_mean(in_path)
    if mean_k is None or mean_k <= 0:
        fail("bad instance")
    B = rrmse([mean_k] * len(obs), obs)          # internal constant-mean baseline

    # ---- evaluate participant expression on held-out points ----
    base_env = dict(ALLOWED_CONSTS)
    base_env.update(ALLOWED_FUNCS)
    preds = []
    for (T, C, _) in held:
        env = dict(base_env)
        env['T'] = T
        env['C'] = C
        try:
            v = eval(code, {"__builtins__": {}}, env)
        except Exception:
            fail("evaluation error on held-out point")
        if isinstance(v, complex):
            fail("complex value on held-out point")
        try:
            fv = float(v)
        except Exception:
            fail("non-numeric result")
        if not math.isfinite(fv):
            fail("non-finite value on held-out point")
        preds.append(fv)

    # ---- score: relative RMSE + mild complexity penalty, normalised by B ----
    F = rrmse(preds, obs)
    complexity_factor = 1.0 + 0.01 * max(0, n_nodes - 8)
    F *= complexity_factor

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    ratio = sc / 1000.0
    sys.stdout.write("baseline_B=%.6f held_rrmse=%.6f complexity=%.4f nodes=%d\n"
                     % (B, F, complexity_factor, n_nodes))
    sys.stdout.write("Ratio: %.6f\n" % ratio)


if __name__ == "__main__":
    main()
