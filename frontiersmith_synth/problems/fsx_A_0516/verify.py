#!/usr/bin/env python3
"""verify.py <in> <out> <ans>   -- deterministic scorer for Morphogen-Golf (format C).

Instance (<in>):   line1 = L ; line2 = target transcript T (space-separated tokens over F + -).
Submission (<out>): an L-system that GROWS to a creature with the SAME FORM as T.
    line1: n            (iterations, integer 0..NMAX)
    line2: axiom        (space-separated tokens)
    line3: k            (number of rules)
    next k lines: LHS RHS...   (LHS one token; RHS space-separated tokens, may be empty)

Terminals with turtle meaning (90 deg): F = draw forward, + = turn left, - = turn right,
[ = push state, ] = pop state.  Any other token is a NON-drawing variable (rewritten if it
has a rule).  A symbol with NO rule is unchanged by iteration (F/+/- may still be rewritten
if the solver gives them a rule -- classic L-system semantics).

FEASIBILITY (exact-match constraint): expand n times, interpret as a turtle, collect the set
of drawn unit edges; the creature must match T's edge set (up to translation and the 8 grid
symmetries) with IoU >= 0.99.

OBJECTIVE (Kolmogorov / gene length, MINIMIZE): F = #axiom-tokens + sum(1 + #RHS-tokens)
over rules + 1 (for n).  score is decreasing in F, calibrated so the literal transcript
(the trivial baseline) ~ 0.1 and short recursive genes climb toward -- but never reach -- 1.
"""
import sys, math

DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
NMAX = 256
LEN_CAP = 4_000_000
OP_CAP = 8_000_000
TOK_CAP = 64          # max chars per token
RULES_CAP = 200_000

# scoring constants (hidden from the statement; tune the ceiling / floor here)
C_FLOOR = 2.0
BASE = 0.10
SPAN = 0.90
CAP = 0.98


def out_ratio(v, reason=""):
    if reason:
        sys.stdout.write("# %s\n" % reason)
    sys.stdout.write("Ratio: %.6f\n" % v)
    sys.exit(0)


def draw_edges(tokens):
    x = y = 0
    h = 0
    stack = []
    edges = set()
    ne = 0
    for t in tokens:
        if t == 'F':
            dx, dy = DIRS[h]
            nx, ny = x + dx, y + dy
            edges.add(frozenset(((x, y), (nx, ny))))
            x, y = nx, ny
            ne += 1
            if ne > LEN_CAP:
                return None
        elif t == '+':
            h = (h + 1) % 4
        elif t == '-':
            h = (h + 3) % 4
        elif t == '[':
            stack.append((x, y, h))
        elif t == ']':
            if stack:
                x, y, h = stack.pop()
    return edges


def expand(axiom, rules, n):
    cur = list(axiom)
    ops = 0
    for _ in range(n):
        if not any(s in rules for s in cur):
            break
        nxt = []
        for s in cur:
            r = rules.get(s)
            if r is None:
                nxt.append(s)
            else:
                nxt.extend(r)
            ops += 1
            if len(nxt) > LEN_CAP or ops > OP_CAP:
                return None
        cur = nxt
    return cur


# ---- the 8 dihedral maps of the square lattice ----
def _transforms():
    fns = []
    for a, b, c, d in [(1, 0, 0, 1), (0, -1, 1, 0), (-1, 0, 0, -1), (0, 1, -1, 0),
                       (-1, 0, 0, 1), (1, 0, 0, -1), (0, 1, 1, 0), (0, -1, -1, 0)]:
        fns.append(lambda p, a=a, b=b, c=c, d=d: (a * p[0] + b * p[1], c * p[0] + d * p[1]))
    return fns


def normalize(edges, fn):
    pts = []
    for e in edges:
        pts.extend(fn(p) for p in e)
    mnx = min(p[0] for p in pts)
    mny = min(p[1] for p in pts)
    out = set()
    for e in edges:
        (p, q) = tuple(e)
        tp = fn(p)
        tq = fn(q)
        out.add(frozenset(((tp[0] - mnx, tp[1] - mny), (tq[0] - mnx, tq[1] - mny))))
    return out


def best_iou(sub_edges, target_norm):
    best = 0.0
    for fn in _transforms():
        se = normalize(sub_edges, fn)
        inter = len(se & target_norm)
        union = len(se | target_norm)
        if union == 0:
            continue
        iou = inter / union
        if iou > best:
            best = iou
            if best >= 0.999999:
                break
    return best


def main():
    inf, outf = sys.argv[1], sys.argv[2]

    # -- read instance --
    with open(inf) as f:
        idata = f.read().split("\n")
    try:
        L = int(idata[0].strip())
    except Exception:
        out_ratio(0.0, "bad instance")
    T = idata[1].split() if len(idata) > 1 else []
    if len(T) != L or L <= 0:
        out_ratio(0.0, "bad instance body")
    target = draw_edges(T)
    if not target:
        out_ratio(0.0, "empty target")
    target_norm = normalize(target, _transforms()[0])

    # -- read submission --
    try:
        with open(outf) as f:
            odata = f.read().split("\n")
    except Exception:
        out_ratio(0.0, "no output")
    if len(odata) < 3:
        out_ratio(0.0, "too few lines")
    try:
        n = int(odata[0].strip())
    except Exception:
        out_ratio(0.0, "bad n")
    if n < 0 or n > NMAX:
        out_ratio(0.0, "n out of range")
    axiom = odata[1].split()
    try:
        k = int(odata[2].strip())
    except Exception:
        out_ratio(0.0, "bad rule count")
    if k < 0 or k > RULES_CAP:
        out_ratio(0.0, "k out of range")
    if len(odata) < 3 + k:
        out_ratio(0.0, "missing rules")

    rules = {}
    obj = len(axiom) + 1  # axiom tokens + 1 for n
    for tok in axiom:
        if len(tok) == 0 or len(tok) > TOK_CAP:
            out_ratio(0.0, "bad axiom token")
    for i in range(k):
        parts = odata[3 + i].split()
        if len(parts) < 1:
            out_ratio(0.0, "empty rule line")
        lhs = parts[0]
        rhs = parts[1:]
        if len(lhs) > TOK_CAP or any(len(t) > TOK_CAP for t in rhs):
            out_ratio(0.0, "token too long")
        if lhs in rules:
            out_ratio(0.0, "duplicate rule LHS")
        rules[lhs] = rhs
        obj += 1 + len(rhs)

    if not axiom:
        out_ratio(0.0, "empty axiom")

    expanded = expand(axiom, rules, n)
    if expanded is None:
        out_ratio(0.0, "expansion exceeds cap")
    sub_edges = draw_edges(expanded)
    if not sub_edges:
        out_ratio(0.0, "creature draws nothing")

    iou = best_iou(sub_edges, target_norm)
    if iou < 0.99:
        out_ratio(0.0, "form mismatch iou=%.4f" % iou)

    # -- objective -> score (minimize gene length) --
    lit = float(L + 1)          # literal-program baseline (trivial reference)
    F = float(max(obj, 1))
    a = math.log(max(lit, math.e))
    x = math.log(max(F, 1.0))
    denom = a - math.log(C_FLOOR)
    if denom <= 1e-9:
        frac = 0.0
    else:
        frac = (a - x) / denom
    score = BASE + SPAN * frac
    if score < 0.0:
        score = 0.0
    if score > CAP:
        score = CAP
    out_ratio(score)


if __name__ == "__main__":
    main()
