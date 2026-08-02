#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for chromatogram-peak-unmix.

Reads testId from <in>'s header and regenerates the hidden truth
(which reference slots are truly present, their areas, and the shared column
tailing constant tau) with the EXACT same seeded recipe as gen.py's
instance_spec() -- the ground truth is never taken from <in> or <out>.

Participant output: M non-negative finite numbers x_1..x_M, one estimated
area per reference slot (0 => "absent").

Score = compound-identification F1 (slot i counted "predicted present" iff
x_i >= thr) times a per-matched-compound area-accuracy factor
exp(-|x_i-a_i|/a_i / lam), averaged over true positives.  A phantom compound
(false positive) or a badly wrong area on a real compound both cost score;
Ratio = min(1, SCALE * F1 * AreaAcc).
"""
import sys, math, random

D = 20.0
SIGMA = 3.0
BASE_AREA = 40.0
THR = 0.25 * BASE_AREA
LAM = 0.15
SCALE = 0.90
MAX_AREA_TOKEN = 1.0e9


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


# ---------- hidden ground truth (identical to gen.py's instance_spec) ----------
def instance_spec(testId):
    rng = random.Random(90210 + testId * 7919)
    M = 8 + (testId - 1)
    if testId == 1:
        tau_true = 0.0
    else:
        tau_true = min(10.0, 2.0 + 1.0 * (testId - 2))
    r = [int(D * (i + 1)) for i in range(M)]
    T = int(M * D + 8 * SIGMA + 4 * tau_true + 20)

    K = max(2, int(round(M * 0.5)))
    order = list(range(M))
    rng.shuffle(order)
    present = sorted(order[:K])
    areas = {i: rng.uniform(0.6, 1.6) * BASE_AREA for i in present}

    if testId >= 4 and tau_true > 0:
        candidates = [i for i in present if i + 1 < M]
        rng.shuffle(candidates)
        placed = False
        for j in candidates:
            if (j + 1) not in present:
                areas[j] = rng.uniform(1.8, 2.6) * BASE_AREA
                placed = True
                break
        if not placed and candidates:
            j = candidates[0]
            present = sorted(set(present) - {j + 1})
            areas.pop(j + 1, None)
            areas[j] = rng.uniform(1.8, 2.6) * BASE_AREA

    return dict(T=T, M=M, r=r, sigma=SIGMA, thr=THR, lam=LAM,
                tau_true=tau_true, present=set(present), areas=areas)


def main():
    if len(sys.argv) < 3:
        fail("usage")

    try:
        header_tokens = open(sys.argv[1]).read().split()
        testId = int(header_tokens[0])
        M_declared = int(header_tokens[2])
    except Exception:
        fail("bad input header")

    if testId < 1 or testId > 100000:
        fail("bad test id")

    spec = instance_spec(testId)
    M = spec['M']
    if M_declared != M:
        fail("internal generation mismatch")

    present = spec['present']
    areas = spec['areas']
    thr = spec['thr']
    lam = spec['lam']

    try:
        raw = open(sys.argv[2]).read().split()
    except Exception:
        fail("cannot read output")

    if len(raw) != M:
        fail("expected %d numbers, got %d" % (M, len(raw)))

    x = []
    for tok in raw:
        try:
            v = float(tok)
        except Exception:
            fail("non-numeric token %r" % tok)
        if not math.isfinite(v):
            fail("non-finite value %r" % tok)
        if v < 0.0:
            fail("negative area %r" % tok)
        if v > MAX_AREA_TOKEN:
            fail("area out of range %r" % tok)
        x.append(v)

    pred = set(i for i in range(M) if x[i] >= thr)
    TP = pred & present
    FP = pred - present
    FN = present - pred
    tp, fp, fn = len(TP), len(FP), len(FN)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 0.0 if tp == 0 else 2.0 * prec * rec / (prec + rec)

    if TP:
        accs = []
        for i in TP:
            relerr = abs(x[i] - areas[i]) / areas[i]
            accs.append(math.exp(-relerr / lam))
        area_acc = sum(accs) / len(accs)
    else:
        area_acc = 0.0

    F = f1 * area_acc
    ratio = min(1.0, SCALE * F)

    print("tp=%d fp=%d fn=%d f1=%.4f area_acc=%.4f F=%.4f  Ratio: %.6f"
          % (tp, fp, fn, f1, area_acc, F, ratio))


if __name__ == "__main__":
    main()
