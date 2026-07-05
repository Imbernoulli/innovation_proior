#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_G_0477 -- "Fleet-Telemetry Unsupervised Anomaly Scorer".

Family: ml-anomaly-scorer (MLS-Bench flavored). Theme: FLEET TELEMETRY.

A fleet of machines (say, long-haul trucks / turbines) each streams a vector of
sensor readings per operating snapshot: engine temperature, vibration RMS, oil
pressure, coolant flow, rpm, voltage, ... The features are CORRELATED under normal
operation (rpm drives fuel flow, temperature tracks load, etc.). A small fraction of
snapshots are FAULTS. The candidate is an UNSUPERVISED anomaly scorer: it sees only
the unlabeled telemetry matrix X (one row per snapshot) and must return a real-valued
anomaly score per row (HIGHER = more anomalous). It never sees the fault labels.

The catch that makes this open-ended: the faults come from DIVERSE regimes, and no
single simple detector wins them all:
  * global/point faults  -- a sensor pegged far into its tail (a univariate z-score
    already catches these);
  * correlation-break faults -- every marginal looks normal, but the joint structure
    is destroyed (features that should move together do not) -> needs a MULTIVARIATE
    view (Mahalanobis / whitening);
  * contextual faults -- one sensor reflected against its mean so it contradicts its
    correlated partners; marginals still look normal, a very thin off-manifold shift;
  * density faults -- normal operation is BIMODAL (two operating modes); the fault is a
    scattered, ISOLATED low-density outlier, offset from a mode by a few marginal-std in
    a RANDOM full-dimensional direction, so NO single sensor is extreme (~offset/sqrt(d)
    per sensor) -> a univariate view is weak; the point only stands out in the JOINT /
    LOCAL-DENSITY view (Mahalanobis / kNN / LOF);
  * mixtures of the above.
A strong entry must therefore combine a marginal, a global multivariate, and a
local-density view. There is no closed-form optimum and many viable strategies
(Mahalanobis / whitening, PCA reconstruction error, kNN/LOF, iForest-style random
splits, one-class boundaries, and rank ensembles of these).

Scoring (deterministic; NO wall-time / GPU):
  For each instance the evaluator computes ROC-AUC of the candidate's scores against
  the HIDDEN fault labels (Mann-Whitney form, exact tie handling -> a constant score
  gives AUC exactly 0.5). Per-instance reward is an affine lift over chance:
        r = clip( (AUC - 0.45) / 0.5 , 0, 1 )
  so AUC 0.50 (chance / constant scorer) -> 0.10, AUC 0.95 -> 1.00, AUC 0.90 -> 0.90.
  The reported Ratio is the mean reward over all (public-diverse + held-out-hard)
  instances -- the family's "geometric-mean AUC across regimes" specialized to a
  bounded, chance-anchored reward. Malformed / non-finite / wrong-length answers -> 0.

The candidate is UNTRUSTED model output: it is run OS-sandboxed (bwrap) via `isorun`,
sees ONLY inst["public"] on stdin, and returns ONLY its score list on stdout, so it can
never reach the hidden labels, the scorer, or the evaluator frames.

CLI:  python3 evaluator.py <candidate.py>
Prints:
  Ratio: <mean reward in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun

_MASK64 = (1 << 64) - 1


# ----------------------------- deterministic RNG ---------------------------
class Rng:
    """splitmix64 -> uniform (0,1) and standard normals (Box-Muller). Fully seeded."""
    def __init__(self, seed):
        self.state = seed & _MASK64
        self._spare = None

    def _u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        z ^= z >> 31
        return z & _MASK64

    def uniform(self):
        # in (0,1)
        return (self._u64() + 0.5) / 18446744073709551616.0

    def randint(self, lo, hi):        # inclusive
        return lo + self._u64() % (hi - lo + 1)

    def normal(self):
        if self._spare is not None:
            v = self._spare
            self._spare = None
            return v
        u1 = self.uniform()
        u2 = self.uniform()
        r = math.sqrt(-2.0 * math.log(u1))
        self._spare = r * math.sin(2.0 * math.pi * u2)
        return r * math.cos(2.0 * math.pi * u2)


# ----------------------------- instance family -----------------------------
def _make_one(seed, n, d, regime, contam, mag):
    """Deterministically build one telemetry dataset.
    Returns (X, y, feature_names). Normal snapshots share a correlated Gaussian
    structure (mixing matrix A); faults are injected per `regime`."""
    r = Rng(seed)
    # feature base levels (sensor operating points) and a dense mixing matrix A
    mu = [50.0 + 15.0 * r.normal() for _ in range(d)]
    A = [[(1.0 if i == j else 0.0) + 0.55 * r.normal() for j in range(d)] for i in range(d)]
    s = [math.sqrt(sum(A[j][k] ** 2 for k in range(d))) for j in range(d)]  # marginal std

    # second operating mode (normal telemetry is bimodal: two operating modes)
    mu2 = [mu[j] + (3.0 * s[j]) * (1 if r.uniform() < 0.5 else -1) for j in range(d)]

    def normal_point(center):
        z = [r.normal() for _ in range(d)]
        return [center[j] + sum(A[j][k] * z[k] for k in range(d)) for j in range(d)]

    n_anom = max(3, int(round(contam * n)))
    n_norm = n - n_anom
    X = []
    y = []

    bimodal = (regime == "density")
    for _ in range(n_norm):
        center = mu
        if bimodal:
            center = mu if r.uniform() < 0.5 else mu2
        X.append(normal_point(center))
        y.append(0)

    for _ in range(n_anom):
        reg = regime
        if regime == "mixA":            # mixture: half global, half correlation-break
            reg = "global" if r.uniform() < 0.5 else "corrbreak"
        elif regime == "mixB":          # mixture: half correlation-break, half contextual
            reg = "corrbreak" if r.uniform() < 0.5 else "contextual"

        if reg == "global":
            p = normal_point(mu)
            kf = 1 + (1 if r.uniform() < 0.5 else 0)          # 1-2 pegged sensors
            for _ in range(kf):
                j = r.randint(0, d - 1)
                sign = 1.0 if r.uniform() < 0.5 else -1.0
                p[j] += sign * mag * s[j]
        elif reg == "corrbreak":
            # marginals identical to normal, but sampled INDEPENDENTLY -> joint broken
            p = [mu[j] + s[j] * r.normal() for j in range(d)]
        elif reg == "contextual":
            # a normal point with ONE correlated sensor reflected about its mean:
            # it now contradicts its partners (thin off-manifold shift; marginals ok)
            p = normal_point(mu)
            j = r.randint(0, d - 1)
            p[j] = 2.0 * mu[j] - p[j]
        elif reg == "density":
            # a scattered, ISOLATED low-density outlier: from an operating mode, shifted
            # by `mag` marginal-std in a RANDOM full-dimensional direction. No single
            # sensor is extreme (per-feature ~ mag/sqrt(d)), so a univariate view is
            # weak, but the point sits in a sparse joint region (kNN / multivariate).
            base = mu if r.uniform() < 0.5 else mu2
            gv = [r.normal() for _ in range(d)]
            nrm = math.sqrt(sum(g * g for g in gv)) or 1.0
            # joint offset of `mag` marginal-std spread over ALL sensors, so each single
            # sensor moves only ~mag/sqrt(d) (univariate-invisible), joint sits in a
            # sparse region (kNN / multivariate catch it).
            p = [base[j] + (gv[j] / nrm) * mag * s[j] for j in range(d)]
        else:
            p = normal_point(mu)
        X.append(p)
        y.append(1)

    # deterministic shuffle (Fisher-Yates with the same RNG stream)
    for i in range(n - 1, 0, -1):
        jj = r.randint(0, i)
        X[i], X[jj] = X[jj], X[i]
        y[i], y[jj] = y[jj], y[i]

    names = ["egt", "vib_rms", "oil_prs", "coolant", "rpm", "volt",
             "fuel_flow", "boost", "torque", "ambient"][:d]
    while len(names) < d:
        names.append("s%d" % len(names))
    return X, y, names


def make_instances():
    """Deterministic, seeded. A spread of regimes and difficulties; the last few are
    harder/held-out (higher dimension, subtler shifts, lower contamination) so a strong
    entry keeps headroom (does not saturate to 1.0)."""
    specs = [
        # seed,  n,   d, regime,       contam, mag
        (1101, 420, 6, "global",     0.10, 5.0),
        (1102, 420, 6, "global",     0.08, 3.2),
        (1103, 440, 6, "corrbreak",  0.10, 0.0),
        (1104, 440, 7, "corrbreak",  0.06, 0.0),
        (1105, 420, 6, "contextual", 0.10, 0.0),
        (1106, 460, 7, "contextual", 0.06, 0.0),
        (1107, 440, 6, "density",    0.10, 3.0),
        (1108, 480, 7, "density",    0.06, 3.0),
        (1109, 460, 6, "mixA",       0.10, 4.0),
        (1110, 480, 7, "mixB",       0.08, 0.0),
        # ---- harder / held-out ----
        (1111, 520, 10, "corrbreak", 0.05, 0.0),
        (1112, 540, 8, "density",    0.04, 2.8),
    ]
    out = []
    for seed, n, d, regime, contam, mag in specs:
        X, y, names = _make_one(seed, n, d, regime, contam, mag)
        pub = {
            "instance_id": seed,
            "n": n, "d": d,
            "X": X,
            "feature_names": names,
            "contamination": round(contam, 3),   # domain prior; scorer may use or ignore
            "note": "unsupervised: labels are hidden; higher score = more anomalous",
        }
        out.append({"public": pub, "hidden": {"y": y}})
    return out


# ----------------------------- ROC-AUC (exact, tie-safe) -------------------
def _auc(scores, y):
    """Mann-Whitney AUC with average ranks for ties. Constant scores -> exactly 0.5."""
    n = len(scores)
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0        # 1-based average rank over the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    npos = sum(1 for v in y if v == 1)
    nneg = n - npos
    if npos == 0 or nneg == 0:
        return 0.5
    sum_pos = sum(ranks[i] for i in range(n) if y[i] == 1)
    return (sum_pos - npos * (npos + 1) / 2.0) / (npos * nneg)


# ----------------------------- scoring -------------------------------------
def score(inst, answer):
    """Strictly validate; return (ok, auc)."""
    pub = inst["public"]
    y = inst["hidden"]["y"]
    n = pub["n"]
    if not isinstance(answer, dict):
        return False, None
    sc = answer.get("scores", None)
    if not isinstance(sc, list) or len(sc) != n:
        return False, None
    try:
        sc = [float(x) for x in sc]
    except (TypeError, ValueError):
        return False, None
    for v in sc:
        if not math.isfinite(v):
            return False, None
    auc = _auc(sc, y)
    if not math.isfinite(auc):
        return False, None
    return True, float(auc)


def _reward(auc):
    r = (auc - 0.45) / 0.5
    if r < 0.0:
        return 0.0
    if r > 1.0:
        return 1.0
    return r


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <candidate.py>")
        sys.exit(2)
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=20)
        if st != "OK":
            vec.append(0.0); continue
        try:
            ok, auc = score(inst, ans)
        except Exception:
            ok, auc = False, None
        if not ok or auc is None:
            vec.append(0.0); continue
        r = _reward(auc)
        vec.append(r if (r == r and 0.0 <= r <= 1.0) else 0.0)
    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
