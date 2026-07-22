#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for symmetry-generator-discovery.

The participant's <out> must contain exactly 6 finite numbers a11 a12 a21 a22 t1 t2
declaring an affine plane generator V(p) = A*p + t (A=[[a11,a12],[a21,a22]]).
We regenerate the FULL hidden instance (including secrets never shown to the
solver) from testId alone -- an independent copy of gen.py's construction --
then:
  (1) invariance-testing: apply the solver's (normalized) generator's finite
      flow g_s to held-out, far-outside-the-training-slab points and check
      whether the TRUE hidden label is preserved.
  (2) label-propagation-holdout: use the same flow to pull each held-out point
      back toward the training slab, look up the nearest training label there,
      and check that prediction against the true held-out label.
  (3) a mild parsimony penalty on non-minimal generators.
Never leaks the hidden law; always deterministic; O(input).
"""
import sys, math, random

TYPES = ["rotation", "scaling", "translation", "rotation", "scaling",
         "rotation", "scaling", "translation", "rotation", "scaling"]
GCOUNT = [14, 13, 12, 11, 10, 9, 9, 8, 7, 6]
ARCW   = [1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.75, 0.65, 0.55]
QW     = [1.3, 1.2, 1.1, 1.0, 0.9, 0.85, 0.75, 0.7, 0.6, 0.5]
EPS = 0.04
NHELD = 6

S_INV = [1.5, 3.0, 5.0, -1.5, -3.0]
S_PROP = [i * 0.25 for i in range(-32, 33)]
SCALE_INV = 0.8
SCALE_PROP = 0.8
PROP_MAXDIST = 0.6
MAXTOK = 1e4


def _make_group(ax, ay, truelabel):
    pts = [(ax, ay), (ax + EPS, ay), (ax - EPS, ay), (ax, ay + EPS), (ax, ay - EPS)]
    return [(x, y, truelabel(x, y)) for (x, y) in pts]


def build_instance(tid):
    tid = int(tid)
    idx = (tid - 1) % 10
    typ = TYPES[idx]
    rnd = random.Random(90000 + 131 * tid)
    groups = []
    heldout = []

    if typ == "rotation":
        hx = rnd.uniform(-2.0, 2.0); hy = rnd.uniform(-2.0, 2.0)
        b0 = rnd.uniform(-1.0, 1.0); b1 = rnd.uniform(0.8, 1.6)
        b2 = rnd.uniform(-0.15, 0.15); b3 = rnd.uniform(-0.6, 0.6)

        def hfun(r):
            return b0 + b1 * r + b2 * r * r + b3 * math.sin(2.0 * r)

        def truelabel(x, y):
            r = math.hypot(x - hx, y - hy)
            return hfun(r)

        arc = ARCW[idx]
        for _ in range(GCOUNT[idx]):
            r = rnd.uniform(3.0, 4.0); th = rnd.uniform(0.0, arc)
            ax = hx + r * math.cos(th); ay = hy + r * math.sin(th)
            groups.append(_make_group(ax, ay, truelabel))
        for _ in range(NHELD):
            # SAME radius band as training (the invariant coordinate must stay
            # reachable), but an angle far OUTSIDE the training arc: this point
            # is unreachable by interpolation yet on a training-covered orbit.
            r = rnd.uniform(2.9, 4.1)
            th = rnd.uniform(arc + 0.5, 2.0 * math.pi - 0.15)
            x = hx + r * math.cos(th); y = hy + r * math.sin(th)
            heldout.append((x, y, truelabel(x, y)))

    elif typ == "scaling":
        hx = rnd.uniform(-2.0, 2.0); hy = rnd.uniform(-2.0, 2.0)
        b0 = rnd.uniform(-1.0, 1.0); b1 = rnd.uniform(-1.3, 1.3)
        b2 = rnd.uniform(-1.3, 1.3); b3 = rnd.uniform(-0.5, 0.5)

        def hfun(th):
            return b0 + b1 * math.cos(th) + b2 * math.sin(th) + b3 * math.cos(2.0 * th)

        def truelabel(x, y):
            th = math.atan2(y - hy, x - hx)
            return hfun(th)

        arc = ARCW[idx]
        for _ in range(GCOUNT[idx]):
            r = rnd.uniform(1.0, 2.0); th = rnd.uniform(0.0, arc)
            ax = hx + r * math.cos(th); ay = hy + r * math.sin(th)
            groups.append(_make_group(ax, ay, truelabel))
        for _ in range(NHELD):
            # SAME angular sector as training (the invariant coordinate stays
            # reachable), but a radius far OUTSIDE the training band.
            r = rnd.uniform(4.0, 8.0)
            th = rnd.uniform(0.0, arc)
            x = hx + r * math.cos(th); y = hy + r * math.sin(th)
            heldout.append((x, y, truelabel(x, y)))

    else:  # translation
        phi = rnd.uniform(0.0, 2.0 * math.pi)
        ux, uy = math.cos(phi), math.sin(phi)
        vx, vy = -math.sin(phi), math.cos(phi)
        b0 = rnd.uniform(-1.0, 1.0)
        sgn = rnd.choice([-1.0, 1.0]); b1 = sgn * rnd.uniform(0.5, 1.3)
        b2 = rnd.uniform(-0.1, 0.1); b3 = rnd.uniform(-0.5, 0.5)

        def hfun(q):
            return b0 + b1 * q + b2 * q * q + b3 * math.sin(0.6 * q)

        def truelabel(x, y):
            q = x * vx + y * vy
            return hfun(q)

        qw = QW[idx]
        for _ in range(GCOUNT[idx]):
            p = rnd.uniform(-1.5, 1.5); q = rnd.uniform(0.3, 0.3 + qw)
            ax = p * ux + q * vx; ay = p * uy + q * vy
            groups.append(_make_group(ax, ay, truelabel))
        for _ in range(NHELD):
            # SAME q-band as training (the invariant coordinate stays
            # reachable), but a p_par far OUTSIDE the training range.
            p = rnd.choice([-1.0, 1.0]) * rnd.uniform(4.0, 9.0)
            q = rnd.uniform(0.3, 0.3 + qw)
            x = p * ux + q * vx; y = p * uy + q * vy
            heldout.append((x, y, truelabel(x, y)))

    return {"type": typ, "groups": groups, "heldout": heldout, "truelabel": truelabel}



# ---------- pure-python 3x3 affine-flow machinery ----------
def mat3_mul(A, B):
    return [[A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j]
              for j in range(3)] for i in range(3)]


def matexp3(M, s):
    N = [[M[i][j] * s for j in range(3)] for i in range(3)]
    norm = sum(abs(N[i][j]) for i in range(3) for j in range(3))
    k = 0
    while norm > 0.5:
        norm /= 2.0
        k += 1
    scale = 2.0 ** k
    Ns = [[N[i][j] / scale for j in range(3)] for i in range(3)]
    result = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    term = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    for n in range(1, 18):
        term = mat3_mul(term, Ns)
        term = [[v / n for v in row] for row in term]
        result = [[result[i][j] + term[i][j] for j in range(3)] for i in range(3)]
    for _ in range(k):
        result = mat3_mul(result, result)
    return result


def apply_generator(a, b, c, d, t1, t2, s, x, y):
    M = [[a, b, t1], [c, d, t2], [0.0, 0.0, 0.0]]
    E = matexp3(M, s)
    x2 = E[0][0] * x + E[0][1] * y + E[0][2]
    y2 = E[1][0] * x + E[1][1] * y + E[1][2]
    return x2, y2


def compute_F(a, b, c, d, t1, t2, inst):
    truelabel = inst["truelabel"]
    heldout = inst["heldout"]
    groups = inst["groups"]
    train_pts = [pt for g in groups for pt in g]

    inv_acc = 0.0
    inv_n = 0
    for (hx0, hy0, hl0) in heldout:
        for s in S_INV:
            x2, y2 = apply_generator(a, b, c, d, t1, t2, s, hx0, hy0)
            l2 = truelabel(x2, y2)
            diff = abs(hl0 - l2)
            inv_acc += math.exp(-diff / SCALE_INV)
            inv_n += 1
    invariance_score = inv_acc / inv_n

    prop_acc = 0.0
    for (hx0, hy0, hl0) in heldout:
        best_dist = None
        best_label = None
        for s in S_PROP:
            x2, y2 = apply_generator(a, b, c, d, t1, t2, s, hx0, hy0)
            for (tx, ty, tl) in train_pts:
                dist = math.hypot(x2 - tx, y2 - ty)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_label = tl
        if best_dist is not None and best_dist <= PROP_MAXDIST:
            diff2 = abs(best_label - hl0)
            prop_acc += math.exp(-diff2 / SCALE_PROP)
    propagation_score = prop_acc / len(heldout)

    active = sum(1 for v in (a, b, c, d, t1, t2) if abs(v) > 0.05)
    complexity_factor = max(0.6, 1.0 - 0.03 * max(0, active - 2))

    return (0.5 * invariance_score + 0.5 * propagation_score) * complexity_factor


def fail(reason):
    print("Infeasible: %s Ratio: 0.0" % reason)
    sys.exit(0)


def main():
    inf, outf = sys.argv[1], sys.argv[2]
    with open(inf) as f:
        first_tok = f.readline().split()
    if not first_tok:
        fail("empty instance")
    try:
        tid = int(first_tok[0])
    except ValueError:
        fail("bad instance header")

    try:
        raw = open(outf).read()
    except Exception:
        fail("cannot read output")
    toks = raw.split()
    if len(toks) != 6:
        fail("expected exactly 6 numbers, got %d" % len(toks))
    vals = []
    for t in toks:
        try:
            v = float(t)
        except ValueError:
            fail("non-numeric token %r" % t)
        if not math.isfinite(v):
            fail("non-finite value")
        if abs(v) > MAXTOK:
            fail("value out of range")
        vals.append(v)
    a, b, c, d, t1, t2 = vals
    norm = math.sqrt(a * a + b * b + c * c + d * d + t1 * t1 + t2 * t2)
    if norm < 1e-9:
        fail("degenerate zero generator")
    a, b, c, d, t1, t2 = (v / norm for v in vals)

    inst = build_instance(tid)
    F = compute_F(a, b, c, d, t1, t2, inst)
    B = compute_F(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, inst)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
