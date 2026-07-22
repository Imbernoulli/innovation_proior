#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the retired-traffic-engineer controller recovery task.

- Reads the test id from <in>'s header, then regenerates the hidden controller
  and a HELD-OUT set of 5-way and 6-way junction episodes (topologies never
  present in the training log) entirely from that id.  The hidden decision
  rule -- weighted-queue priority, a starvation override, and a clockwise
  tie-break -- lives ONLY here (and, necessarily duplicated, inside gen.py's
  own log-construction code); it is never read from any file.
- Parses the participant's ONE-LINE priority expression:
      PRIORITY <expr>
  `expr` is arithmetic over the per-approach variables q, w, a, cw, n,
  constants, + - * /, parentheses, and the unary functions sig, step, relu,
  tanh, absv.  For every held-out junction STATE the expression is evaluated
  once per approach (substituting that approach's own q,w,a,cw and the
  junction's n); the approach with the LARGEST value is the prediction
  (ties broken by the lower approach index).
- Score = held-out prediction accuracy, with a light expression-size penalty,
  normalised against the checker's own baseline predictor (priority = 0, a
  degenerate constant that always ties and so always "predicts" approach 0):
      F = accuracy / (1 + LAMBDA * nodes)
      B = baseline_accuracy / (1 + LAMBDA * 1)
      Ratio = min(1000, 100*F/B) / 1000
  Reproducing the baseline scores ~0.1.  Sensor-free but genuinely
  irreducible cases (exact weighted-queue ties whose true winner depends on
  the clockwise tie-break, which the expression must encode itself as a tiny
  numeric perturbation) keep even a fully-correct rule below the ceiling.
"""
import sys, math, ast, random

STARVE_LIMIT = 6
W_MIN, W_MAX = 1, 3
ARRIVAL_MAX = 3
TESTID_SALT = 900001
HELD_SALT = 41221801

LAMBDA = 0.006
MAX_NODES = 60
MAX_OUT_BYTES = 20000

ALLOWED_FUNCS = {
    "sig": lambda x: 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x)))),
    "step": lambda x: 1.0 if x > 0 else 0.0,
    "relu": lambda x: x if x > 0 else 0.0,
    "tanh": math.tanh,
    "absv": abs,
}
ALLOWED_VARS = {"q", "w", "a", "cw", "n"}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden controller (identical decision logic to gen.py) ----------
def decide(q, w, a, last_green, N):
    def cw(i):
        return (i - last_green - 1) % N
    overriders = [i for i in range(N) if a[i] >= STARVE_LIMIT]
    if overriders:
        maxage = max(a[i] for i in overriders)
        cands = [i for i in overriders if a[i] == maxage]
    else:
        maxscore = max(w[i] * q[i] for i in range(N))
        cands = [i for i in range(N) if w[i] * q[i] == maxscore]
    winner = min(cands, key=cw)
    return winner, [cw(i) for i in range(N)]


def held_out_states(t):
    """Regenerate the extrapolation split: 5-way / 6-way junctions, never
    trained on, with a seed disjoint from gen.py's training-log seed."""
    rng = random.Random(HELD_SALT + t * 15485863)
    degrees = [5, 6] if t % 3 else [5, 6, 6]
    n_ep = 6 + (t % 5)
    T = 26 + (t % 7) * 3
    states = []
    for _ in range(n_ep):
        N = rng.choice(degrees)
        w = [rng.randint(W_MIN, W_MAX) for _ in range(N)]
        q = [0] * N
        a = [0] * N
        lg = -1
        for _ in range(T):
            for i in range(N):
                q[i] += rng.randint(0, ARRIVAL_MAX)
            winner, cwlist = decide(q, w, a, lg, N)
            states.append((N, lg, winner, list(q), list(w), list(a), cwlist))
            q[winner] = 0
            a[winner] = 0
            for i in range(N):
                if i != winner:
                    a[i] += 1
            lg = winner
    return states


# ---------- expression parsing / validation ----------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd,
)


def _validate_ast(tree):
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return "disallowed syntax %s" % type(node).__name__
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCS):
                return "disallowed call"
            if node.keywords or len(node.args) != 1:
                return "bad function arity"
        if isinstance(node, ast.Name):
            nm = node.id
            if nm in ALLOWED_FUNCS:
                continue
            if nm not in ALLOWED_VARS:
                return "unknown name %s" % nm
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                return "non-numeric constant"
            v = float(node.value)
            if v != v or v in (float("inf"), float("-inf")):
                return "non-finite constant"
    return None


def _count_nodes(tree):
    return sum(1 for nd in ast.walk(tree)
               if isinstance(nd, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant)))


def parse_program(raw):
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        fail("empty output")
    ln = lines[0]
    head = ln.split(None, 1)
    if len(head) != 2 or head[0].upper() != "PRIORITY":
        fail("missing PRIORITY line")
    text = head[1].strip()
    if not text:
        fail("empty expression")
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("parse error")
    err = _validate_ast(tree)
    if err:
        fail(err)
    nodes = _count_nodes(tree)
    if nodes > MAX_NODES:
        fail("expression too large (%d nodes)" % nodes)
    try:
        code = compile(tree, "<expr>", "eval")
    except Exception:
        fail("compile error")
    return code, nodes


def eval_priority(code, q, w, a, cw, n):
    env = dict(ALLOWED_FUNCS)
    env["q"], env["w"], env["a"], env["cw"], env["n"] = float(q), float(w), float(a), float(cw), float(n)
    try:
        v = eval(code, {"__builtins__": {}}, env)
    except Exception:
        fail("evaluation error")
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        fail("non-numeric priority value")
    v = float(v)
    if v != v or v in (float("inf"), float("-inf")):
        fail("non-finite priority value")
    return v


def predict(code, N, q, w, a, cw):
    vals = [eval_priority(code, q[i], w[i], a[i], cw[i], N) for i in range(N)]
    best_i, best_v = 0, vals[0]
    for i in range(1, N):
        if vals[i] > best_v:
            best_i, best_v = i, vals[i]
    return best_i


def score(code, states):
    correct = 0
    for (N, lg, winner, q, w, a, cw) in states:
        pred = predict(code, N, q, w, a, cw)
        if pred == winner:
            correct += 1
    return correct / len(states)


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            header = fh.readline().split()
        t = int(header[1])
    except Exception:
        fail("bad instance header")
    if t < 1 or t > 100000:
        fail("bad test id")

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    text = raw.decode("utf-8", "replace")

    code, nodes = parse_program(text)

    states = held_out_states(t)
    acc = score(code, states)

    base_code, base_nodes = parse_program("PRIORITY 0")
    base_acc = score(base_code, states)

    F = acc / (1.0 + LAMBDA * nodes)
    B = base_acc / (1.0 + LAMBDA * base_nodes)
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("heldout_acc=%.6f baseline_acc=%.6f nodes=%d n_states=%d  Ratio: %.6f"
          % (acc, base_acc, nodes, len(states), sc / 1000.0))


if __name__ == "__main__":
    main()
