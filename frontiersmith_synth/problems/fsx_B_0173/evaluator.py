"""
Evaluator for fsx_B_0173 -- "Glacier Sensor Net: budgeted Pareto hypervolume on a DTLZ2 surrogate".

FROZEN scaffold. The candidate is UNTRUSTED and runs in an isolated subprocess (isorun):
it reads ONE public-instance JSON on stdin and writes ONE answer JSON on stdout. The DTLZ2
surrogate, the reference point and the exact hypervolume oracle live ONLY in this parent
process, so the candidate cannot reach evaluator internals.

Family: multiobjective-hypervolume. Objective: MAX exact Pareto hypervolume of the M=3
objective vectors produced by evaluating a batch of >=1 and <=budget decision vectors on the
DTLZ2 surface, w.r.t. a fixed reference point (minimization convention).

Normalization (maximization analog of the brief's F/B rule):
    baseline(inst) = HV of the trivial one-shot construction (a single centre sensor).
    r = min(1.0, 0.1 * obj / baseline)   -> a trivial batch scores ~0.1, better batches climb.
"""
import sys, json, math, isorun

M = 3  # three conflicting glacier objectives: (coverage-loss, energy, latency)


def make_instances():
    # Deterministic, seeded distribution. Each instance = a glacier deployment with a decision
    # dimension n, an evaluation budget, and a reference (nadir) point. No randomness needed for
    # the surrogate itself (DTLZ2 is a closed form); the "seed" is exposed for candidate use.
    specs = [
        (6, 40, 1.1), (7, 30, 1.1), (8, 50, 1.2), (9, 25, 1.1),
        (10, 60, 1.2), (8, 20, 1.1), (9, 36, 1.1), (10, 49, 1.2),
    ]
    out = []
    for s, (n, budget, r) in enumerate(specs):
        pub = {"problem": "DTLZ2", "M": M, "n": n, "budget": budget,
               "ref": [r] * M, "seed": 1700 + s}
        out.append({"public": pub, "hidden": {}})
    return out


# ---------- DTLZ2 surrogate (closed form) ----------
def dtlz2(x, M):
    dist = x[M - 1:]
    g = sum((xi - 0.5) ** 2 for xi in dist)
    f = []
    for i in range(M):
        v = 1.0 + g
        for j in range(M - 1 - i):
            v *= math.cos(x[j] * math.pi / 2.0)
        if i > 0:
            v *= math.sin(x[M - 1 - i] * math.pi / 2.0)
        f.append(v)
    return f


# ---------- exact 3-D hypervolume (minimization) ----------
def _area2d(xy, r0, r1):
    pts = [(x, y) for (x, y) in xy if x < r0 and y < r1]
    if not pts:
        return 0.0
    pts.sort(key=lambda p: (p[0], p[1]))
    nd = []
    best = float("inf")
    for x, y in pts:            # keep the strictly-decreasing staircase (non-dominated)
        if y < best:
            nd.append((x, y)); best = y
    area = 0.0
    for j in range(len(nd)):
        nx = nd[j + 1][0] if j + 1 < len(nd) else r0
        area += (nx - nd[j][0]) * (r1 - nd[j][1])
    return area


def hypervolume3(objs, ref):
    r0, r1, r2 = ref
    P = [p for p in objs if p[0] < r0 and p[1] < r1 and p[2] < r2]
    if not P:
        return 0.0
    P.sort(key=lambda p: p[2])
    vol = 0.0
    active = []
    for i, p in enumerate(P):
        active.append((p[0], p[1]))
        nz = P[i + 1][2] if i + 1 < len(P) else r2
        vol += _area2d(active, r0, r1) * (nz - p[2])
    return vol


def baseline(inst):
    pub = inst["public"]
    n = pub["n"]
    centre = [0.5] * n           # a single sensor placed at the domain centre
    return hypervolume3([dtlz2(centre, M)], pub["ref"])


def _finite01(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and \
        v == v and v not in (float("inf"), float("-inf")) and -1e-9 <= v <= 1.0 + 1e-9


def score(inst, answer):
    pub = inst["public"]
    n = pub["n"]; budget = pub["budget"]; ref = pub["ref"]
    if not isinstance(answer, dict) or "points" not in answer:
        return False, 0.0
    pts = answer["points"]
    if not isinstance(pts, list) or not (1 <= len(pts) <= budget):
        return False, 0.0
    objs = []
    for row in pts:
        if not isinstance(row, list) or len(row) != n:
            return False, 0.0
        if not all(_finite01(v) for v in row):
            return False, 0.0
        x = [min(1.0, max(0.0, float(v))) for v in row]
        objs.append(dtlz2(x, M))
    obj = hypervolume3(objs, ref)
    if not (obj == obj and obj != float("inf")):
        return False, 0.0
    return True, obj


def main():
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=20)
        if st != "OK":
            vec.append(0.0); continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0); continue
        b = baseline(inst)
        r = min(1.0, 0.1 * obj / max(b, 1e-12))
        vec.append(r if (r == r and 0.0 <= r <= 1.0) else 0.0)
    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


main()
