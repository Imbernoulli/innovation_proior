#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_G_0457 -- "Forecast Committee: Fusing a Panel of Base Forecasters"
(family: ml-ensemble-combiner; format B, quality-metric).

THEME.  A risk desk runs a COMMITTEE of k probabilistic forecasters (base models).
For each upcoming binary event ("did the flagged shipment actually spoil?",
"did the borrower default?", ...) every committee member emits a probability
forecast p_ij in [0,1].  The desk must FUSE the k member forecasts into a single
committee forecast q_i for every event, and it is judged by how sharp AND
calibrated that committee forecast is against the realized outcomes (Brier score).

The members are heterogeneous: some are genuine experts (their forecasts track the
truth), some are near-noise, some are systematically over/under-confident
(miscalibrated), and some are biased.  A naive equal-weight average lets the weak
members drag the committee down.  A good fusion rule figures out -- from a labelled
VALIDATION history of past forecasts+outcomes -- which members to trust, how to
re-weight them, and how to recalibrate, WITHOUT ever seeing the future (test)
outcomes.  There is no single dominant strategy: skill-weighting, logistic/linear
stacking, trimming, rank-fusion and calibration all buy different amounts on
different panels, and none reaches the (unreachable) test-fitted oracle.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str,
             "k": K,                 # number of committee members
             "n_val": Nv,            # labelled validation events
             "n_test": Nt,           # test events to forecast (outcomes HIDDEN)
             "val_pred":  [[p_0,...,p_{K-1}], ... Nv rows],   # member forecasts, in [0,1]
             "val_y":     [y_0, ..., y_{Nv-1}],               # validation OUTCOMES (0/1) -- given
             "test_pred": [[p_0,...,p_{K-1}], ... Nt rows]}   # member forecasts on test events
  stdout: ONE JSON object:
            {"forecast": [q_0, ..., q_{Nt-1}]}
          the fused committee forecast for every test event.  Each q_i must be a
          finite real in [0,1].  Wrong length, a non-finite value (nan/inf), a value
          outside [0,1], a crash, a timeout, or non-JSON -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  Per instance, with test outcomes y (HIDDEN):
    BS(q)   = mean_i (q_i - y_i)^2                              # Brier, lower is better
    BS_base = BS of the equal-weight committee mean            # weak reference (-> 0.1)
    BS_orac = residual MSE of the least-squares linear fuser    # UNREACHABLE ideal (-> 1.0)
              fitted on the TEST outcomes (intercept + K member columns, tiny ridge)
    BS_cand = BS of the candidate's fused forecast
  and we normalize with an affine anchor (equal-weight mean -> 0.1, oracle -> 1.0):
    r = clamp( 0.1 + 0.9 * (BS_base - BS_cand) / max(1e-9, BS_base - BS_orac), 0, 1 )
  Matching the equal-weight mean scores ~0.1; reaching the test-fitted oracle scores
  1.0; doing worse than the mean scores < 0.1.  The oracle is fitted on the HELD-OUT
  test outcomes the candidate never sees, so it is a genuinely unreachable upper
  anchor and every honest fuser stays strictly below 1.0 -- real headroom.

ISOLATION.  The candidate is untrusted and runs OS-sandboxed in a FRESH SUBPROCESS
via `isorun.run_candidate`; it only ever sees the PUBLIC instance (validation
labels + member forecasts, NO test outcomes).  BS_base / BS_orac / test outcomes
are computed by THIS parent process, so a frame-walking / introspecting candidate
learns nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]        # per-instance normalized scores
"""
import sys, json, math
import isorun


# ----------------------------- deterministic RNG ---------------------------
class Rng:
    """Seeded 64-bit LCG -> uniform / gaussian, fully deterministic."""
    def __init__(self, seed):
        self.s = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        self._spare = None

    def _u64(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return self.s

    def uniform(self):
        # 53-bit mantissa uniform in [0,1)
        return (self._u64() >> 11) / float(1 << 53)

    def urange(self, lo, hi):
        return lo + (hi - lo) * self.uniform()

    def gauss(self):
        if self._spare is not None:
            g = self._spare
            self._spare = None
            return g
        # Box-Muller (guard u1 away from 0)
        u1 = self.uniform()
        if u1 < 1e-12:
            u1 = 1e-12
        u2 = self.uniform()
        rmag = math.sqrt(-2.0 * math.log(u1))
        self._spare = rmag * math.sin(2.0 * math.pi * u2)
        return rmag * math.cos(2.0 * math.pi * u2)


def _sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# ----------------------------- instance family -----------------------------
def _build_committee(seed, k, n_val, n_test, sigma_z, profile):
    """
    Build a heterogeneous committee.  Returns
        val_pred (n_val x k), val_y (n_val), test_pred (n_test x k), test_y (n_test).
    Each member j has a skill s_j (signal fraction), bias b_j, noise e_j and a
    calibration temperature c_j.  Outcomes are sampled from the latent truth so the
    panel is learnable but no member is perfect.
    """
    rng = Rng(seed)
    # ---- member profiles ----
    skill = []
    bias = []
    noise = []
    temp = []
    for j in range(k):
        u = rng.uniform()
        if profile == "mixed":
            # a spread of experts / mediocre / near-noise members
            if u < 0.34:
                s = rng.urange(0.75, 1.05)      # expert
            elif u < 0.67:
                s = rng.urange(0.35, 0.65)      # mediocre
            else:
                s = rng.urange(0.05, 0.25)      # near-noise
        elif profile == "fewgood":
            # a couple of strong members hidden among many weak ones
            s = rng.urange(0.8, 1.1) if u < 0.3 else rng.urange(0.05, 0.35)
        elif profile == "miscal":
            # decent signal but strong miscalibration is the main issue
            s = rng.urange(0.5, 0.95)
        else:
            s = rng.urange(0.2, 0.9)
        skill.append(s)
        bias.append(rng.gauss() * 0.6)
        noise.append(rng.urange(0.3, 1.3))
        if profile == "miscal":
            temp.append(rng.urange(0.35, 2.4))   # heavy over/under-confidence
        else:
            temp.append(rng.urange(0.6, 1.7))

    def emit(n):
        preds = []
        ys = []
        for _ in range(n):
            z = rng.gauss() * sigma_z
            p_true = _sigmoid(z)
            y = 1 if rng.uniform() < p_true else 0
            row = []
            for j in range(k):
                lj = temp[j] * (skill[j] * z + bias[j] + noise[j] * rng.gauss())
                pj = _sigmoid(lj)
                # clip strictly inside (0,1) so nothing is a degenerate 0/1
                if pj < 1e-4:
                    pj = 1e-4
                elif pj > 1.0 - 1e-4:
                    pj = 1.0 - 1e-4
                row.append(pj)
            preds.append(row)
            ys.append(y)
        return preds, ys

    val_pred, val_y = emit(n_val)
    test_pred, test_y = emit(n_test)
    return val_pred, val_y, test_pred, test_y


def _build_instances():
    """Deterministic instance family: (seed, k, n_val, n_test, sigma_z, profile)."""
    specs = [
        (1101, 6,  260, 260, 1.5, "mixed"),
        (1102, 7,  300, 300, 1.6, "mixed"),
        (1103, 8,  240, 240, 1.4, "fewgood"),
        (1104, 6,  280, 280, 1.7, "miscal"),
        (1105, 9,  320, 300, 1.5, "mixed"),
        (1106, 7,  260, 260, 1.3, "fewgood"),
        (1107, 8,  300, 280, 1.6, "miscal"),
        (1108, 6,  240, 240, 1.5, "mixed"),
        # harder / larger held-out panels: more members, thinner validation
        (2201, 10, 200, 300, 1.6, "fewgood"),
        (2202, 11, 220, 320, 1.5, "mixed"),
        (2203, 9,  180, 300, 1.7, "miscal"),
        (2204, 12, 200, 340, 1.5, "fewgood"),
    ]
    out = []
    for (seed, k, n_val, n_test, sigma_z, profile) in specs:
        vp, vy, tp, ty = _build_committee(seed, k, n_val, n_test, sigma_z, profile)
        out.append({"name": "panel%d" % seed, "k": k, "n_val": n_val, "n_test": n_test,
                    "val_pred": vp, "val_y": vy, "test_pred": tp, "test_y": ty})
    return out


# ----------------------------- references ----------------------------------
def _brier(q, y):
    n = len(y)
    s = 0.0
    for i in range(n):
        d = q[i] - y[i]
        s += d * d
    return s / n


def _mean_forecast(pred, k):
    out = []
    invk = 1.0 / k
    for row in pred:
        s = 0.0
        for v in row:
            s += v
        out.append(s * invk)
    return out


def _solve_ridge(A, b, ridge):
    """Solve (A^T A + ridge I) w = A^T b via Gaussian elimination. Pure python.
       A: n x m (rows), b: n.  Returns w (length m)."""
    n = len(A)
    m = len(A[0])
    # normal equations
    ata = [[0.0] * m for _ in range(m)]
    atb = [0.0] * m
    for r in range(n):
        row = A[r]
        yr = b[r]
        for i in range(m):
            ri = row[i]
            atb[i] += ri * yr
            arow = ata[i]
            for j in range(i, m):
                arow[j] += ri * row[j]
    for i in range(m):
        for j in range(i):
            ata[i][j] = ata[j][i]
        ata[i][i] += ridge
    # Gaussian elimination with partial pivoting
    M = [ata[i][:] + [atb[i]] for i in range(m)]
    for col in range(m):
        piv = col
        best = abs(M[col][col])
        for r in range(col + 1, m):
            v = abs(M[r][col])
            if v > best:
                best = v
                piv = r
        if best < 1e-15:
            continue
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
        pivval = M[col][col]
        for r in range(m):
            if r == col:
                continue
            factor = M[r][col] / pivval
            if factor == 0.0:
                continue
            Mr = M[r]
            Mc = M[col]
            for c in range(col, m + 1):
                Mr[c] -= factor * Mc[c]
    w = [0.0] * m
    for i in range(m):
        d = M[i][i]
        w[i] = (M[i][m] / d) if abs(d) > 1e-15 else 0.0
    return w


def _oracle_brier(test_pred, test_y, k):
    """UNREACHABLE ideal: least-squares linear fuser (intercept + K members) fitted
       on the TEST outcomes; return its residual MSE (a lower bound on any linear
       fuser's Brier on this test set)."""
    A = [[1.0] + row for row in test_pred]   # design with intercept
    w = _solve_ridge(A, [float(y) for y in test_y], ridge=1e-6)
    n = len(test_y)
    s = 0.0
    for r in range(n):
        pr = 0.0
        Ar = A[r]
        for i in range(k + 1):
            pr += Ar[i] * w[i]
        d = pr - test_y[r]
        s += d * d
    return s / n


# ----------------------------- validation ----------------------------------
def _valid_forecast(answer, n_test):
    """Validate candidate answer -> return forecast list in [0,1] or None."""
    if not isinstance(answer, dict):
        return None
    q = answer.get("forecast")
    if not isinstance(q, list) or len(q) != n_test:
        return None
    out = []
    for v in q:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        v = float(v)
        if v != v or v in (float("inf"), float("-inf")):
            return None
        if v < -1e-6 or v > 1.0 + 1e-6:
            return None
        if v < 0.0:
            v = 0.0
        elif v > 1.0:
            v = 1.0
        out.append(v)
    return out


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        k = inst["k"]
        n_test = inst["n_test"]
        test_y = inst["test_y"]
        bs_base = _brier(_mean_forecast(inst["test_pred"], k), test_y)
        bs_orac = _oracle_brier(inst["test_pred"], test_y, k)
        denom = bs_base - bs_orac
        if denom < 1e-9:
            denom = 1e-9

        public = {"name": inst["name"], "k": k,
                  "n_val": inst["n_val"], "n_test": n_test,
                  "val_pred": inst["val_pred"], "val_y": inst["val_y"],
                  "test_pred": inst["test_pred"]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            q = _valid_forecast(ans, n_test)
        except Exception:
            q = None
        if q is None:
            vec.append(0.0)
            continue
        bs_cand = _brier(q, test_y)
        r = 0.1 + 0.9 * (bs_base - bs_cand) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
