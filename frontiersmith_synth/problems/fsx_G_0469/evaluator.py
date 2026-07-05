#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_G_0469 -- "Recalibrating a Weather Model's Rain Odds"
(family: ml-calibration; format B, quality-metric).

THEME.  A numerical weather prediction (NWP) system issues, for each forecast
window, a raw "probability of precipitation" (PoP) s_i in (0,1) -- the model's
own estimate that it will rain.  Raw NWP probabilities are notoriously
MISCALIBRATED: depending on season, region and the model's internal ensemble
spread they are systematically over-confident (they push toward 0/1), under-
confident (they hug the base rate), or biased (a wet/dry offset).  The forecast
desk keeps a labelled VALIDATION history of past (raw PoP, did-it-rain) pairs and
must design a POST-HOC CALIBRATION MAP g: raw PoP -> corrected probability, then
apply it to a TEST batch of raw PoPs whose outcomes are still HIDDEN.  It is judged
by a proper score (Brier) on the realized test outcomes.

Why this is genuinely open-ended.  The objective is the Brier score, the proper
scoring rule whose reliability term IS the calibration error but which -- unlike a
bare Expected-Calibration-Error -- also rewards SHARPNESS, so the degenerate
"always predict the base rate" map is bad, not optimal.  Recalibration is a
monotone-friendly problem: a good map must fix the model's over/under-confidence
and bias WITHOUT throwing away the ranking information in the raw score, and it
must generalize from a finite, possibly shifted validation history.  Platt
scaling, temperature scaling, histogram binning, isotonic regression, beta
calibration and shrinkage-to-prior all buy different amounts on different regimes,
and none reaches the (unreachable) test-fitted monotone oracle -- real headroom.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str,
             "n_val":  Nv,                 # labelled validation windows
             "n_test": Nt,                 # test windows to recalibrate (outcomes HIDDEN)
             "val_score":  [s_0, ..., s_{Nv-1}],   # raw model PoP in (0,1)
             "val_y":      [y_0, ..., y_{Nv-1}],   # validation OUTCOMES (0/1) -- given
             "test_score": [s_0, ..., s_{Nt-1}]}   # raw model PoP on the test windows
  stdout: ONE JSON object:
            {"prob": [q_0, ..., q_{Nt-1}]}
          the corrected probability for every test window.  Each q_i must be a
          finite real in [0,1].  Wrong length, a non-finite value (nan/inf), a value
          outside [0,1], a crash, a timeout, or non-JSON -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  Per instance, with test outcomes y (HIDDEN):
    BS(q)   = mean_i (q_i - y_i)^2                              # Brier, lower is better
    BS_base = Brier of the RAW model PoP (identity map)         # weak reference (-> 0.1)
    BS_orac = Brier of the best MONOTONE-in-score map fitted    # UNREACHABLE ideal (-> 1.0)
              on the TEST outcomes (isotonic / pool-adjacent-violators regression)
    BS_cand = Brier of the candidate's corrected probabilities
  normalized with an affine anchor (raw map -> 0.1, test-fitted oracle -> 1.0):
    r = clamp( 0.1 + 0.9 * (BS_base - BS_cand) / max(1e-9, BS_base - BS_orac), 0, 1 )
  Leaving the raw scores untouched scores ~0.1; a monotone map fitted directly on the
  HELD-OUT test outcomes would score 1.0, so it is a genuinely unreachable upper anchor
  and every honest, validation-only calibrator stays strictly below 1.0.

ISOLATION.  The candidate is untrusted and runs OS-sandboxed in a FRESH SUBPROCESS
via `isorun.run_candidate`; it only ever sees the PUBLIC instance (validation
scores+labels and the raw test scores, NO test outcomes).  BS_base / BS_orac / the
test outcomes are computed by THIS parent process, so a frame-walking / introspecting
candidate learns nothing useful.

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
        return (self._u64() >> 11) / float(1 << 53)

    def urange(self, lo, hi):
        return lo + (hi - lo) * self.uniform()

    def gauss(self):
        if self._spare is not None:
            g = self._spare
            self._spare = None
            return g
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


def _logit(p):
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


# ----------------------------- instance family -----------------------------
def _emit(rng, n, sigma, temp, bias, noise, base_shift, asym):
    """Sample n windows.  Latent truth z ~ N(base_shift, sigma); p_true = sigmoid(z);
       outcome y ~ Bernoulli(p_true).  The model's RAW PoP is a miscalibrated, noisy
       transform of the latent logit.  The effective confidence is ASYMMETRIC across
       the wet/dry regimes (temp scaled by 1+/-asym for z>=0 / z<0), so the true
       recalibration curve is NOT affine-in-logit -- a single Platt/logistic map
       under-fits it, while a flexible monotone (isotonic) map can capture it:
         raw = sigmoid( temp_eff*(z + bias) + noise*eps ),  temp_eff regime-dependent."""
    scores = []
    ys = []
    for _ in range(n):
        z = base_shift + sigma * rng.gauss()
        p_true = _sigmoid(z)
        y = 1 if rng.uniform() < p_true else 0
        temp_eff = temp * (1.0 + asym) if z >= 0.0 else temp * (1.0 - asym)
        u = z + bias
        # cubic warp: a saturating S-distortion the affine-in-logit Platt map cannot
        # represent, so a flexible monotone recalibrator has a genuine bias advantage.
        u = u + 0.12 * u * u * u
        lr = temp_eff * u + noise * rng.gauss()
        s = _sigmoid(lr)
        if s < 1e-4:
            s = 1e-4
        elif s > 1.0 - 1e-4:
            s = 1.0 - 1e-4
        scores.append(s)
        ys.append(y)
    return scores, ys


def _build_instances():
    """Deterministic family. Each spec:
       (seed, n_val, n_test, sigma, temp, bias, noise, val_shift, test_shift, asym, name).
    Regimes: overconfident (temp>1), underconfident (temp<1), biased (bias!=0), with an
    asymmetric wet/dry confidence distortion (asym) making the true recalibration curve
    non-affine-in-logit, plus a mild validation/test shift to keep the oracle unreachable."""
    specs = [
        # ---- core regimes (seed,nv,nt,sigma,temp,bias,noise,vsh,tsh,asym,name) ----
        (3101, 300, 300, 1.4, 2.10, 0.00, 0.55, 0.00, 0.00,  0.35, "overconf_A"),
        (3102, 300, 300, 1.5, 0.55, 0.00, 0.50, 0.00, 0.00, -0.30, "underconf_A"),
        (3103, 320, 300, 1.4, 1.05, 0.90, 0.55, 0.00, 0.00,  0.30, "wetbias_A"),
        (3104, 320, 300, 1.4, 1.05, -0.85, 0.55, 0.00, 0.00, 0.30, "drybias_A"),
        (3105, 300, 300, 1.5, 1.80, 0.55, 0.60, 0.00, 0.00,  0.40, "overconf_wet"),
        (3106, 300, 300, 1.5, 0.60, -0.60, 0.55, 0.00, 0.00, -0.35, "underconf_dry"),
        # ---- mild val/test shift (seasonal drift) ----
        (3107, 300, 320, 1.4, 1.70, 0.30, 0.60, -0.10, 0.10, 0.35, "shift_over"),
        (3108, 300, 320, 1.5, 0.65, -0.20, 0.55, 0.08, -0.08, -0.30, "shift_under"),
        # ---- harder: thin validation + stronger noise / mild shift ----
        (4201, 180, 340, 1.5, 2.20, 0.40, 0.85, -0.10, 0.10, 0.40, "thin_overshift"),
        (4202, 180, 340, 1.6, 0.55, 0.60, 0.80, 0.10, -0.10, -0.30, "thin_undershift"),
        (4203, 160, 340, 1.4, 1.30, 1.10, 0.90, 0.00, 0.00,  0.35, "thin_wetbias"),
        (4204, 200, 360, 1.6, 2.40, -0.50, 0.90, -0.08, 0.08, 0.40, "thin_overdry"),
    ]
    out = []
    for (seed, nv, nt, sigma, temp, bias, noise, vsh, tsh, asym, name) in specs:
        rng = Rng(seed)
        vs, vy = _emit(rng, nv, sigma, temp, bias, noise, vsh, asym)
        ts, ty = _emit(rng, nt, sigma, temp, bias, noise, tsh, asym)
        out.append({"name": name, "n_val": nv, "n_test": nt,
                    "val_score": vs, "val_y": vy,
                    "test_score": ts, "test_y": ty})
    return out


# ----------------------------- references ----------------------------------
def _brier(q, y):
    n = len(y)
    s = 0.0
    for i in range(n):
        d = q[i] - y[i]
        s += d * d
    return s / n


def _pav(y_sorted):
    """Pool-adjacent-violators: best monotone non-decreasing fit (L2) to the
       0/1 sequence given in sorted-by-score order.  Returns fitted values."""
    vals = []
    wts = []
    cnts = []
    for yi in y_sorted:
        vals.append(float(yi))
        wts.append(1.0)
        cnts.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1] + 1e-15:
            v2 = vals.pop(); w2 = wts.pop(); c2 = cnts.pop()
            v1 = vals.pop(); w1 = wts.pop(); c1 = cnts.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
            wts.append(w1 + w2)
            cnts.append(c1 + c2)
    out = []
    for v, c in zip(vals, cnts):
        out.extend([v] * c)
    return out


def _oracle_brier(test_score, test_y):
    """UNREACHABLE ideal: minimal Brier achievable by ANY monotone-in-score map,
       fitted on the TEST outcomes (isotonic regression). Uses HELD-OUT labels the
       candidate never sees -> a genuine lower anchor."""
    order = sorted(range(len(test_score)), key=lambda i: test_score[i])
    y_sorted = [float(test_y[i]) for i in order]
    fit = _pav(y_sorted)
    s = 0.0
    for k in range(len(order)):
        d = fit[k] - y_sorted[k]
        s += d * d
    return s / len(order)


# ----------------------------- validation ----------------------------------
def _valid_prob(answer, n_test):
    if not isinstance(answer, dict):
        return None
    q = answer.get("prob")
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
        n_test = inst["n_test"]
        test_y = inst["test_y"]
        bs_base = _brier(inst["test_score"], test_y)
        bs_orac = _oracle_brier(inst["test_score"], test_y)
        denom = bs_base - bs_orac
        if denom < 1e-9:
            denom = 1e-9

        public = {"name": inst["name"], "n_val": inst["n_val"], "n_test": n_test,
                  "val_score": inst["val_score"], "val_y": inst["val_y"],
                  "test_score": inst["test_score"]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            q = _valid_prob(ans, n_test)
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
