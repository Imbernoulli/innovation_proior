#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_G_0465 -- "Masked-Entry Imputation of Clinical Records".

Family: ml-imputation (breadth-fill: dataset/domain/task diversity), skinned as an
electronic-health-record (EHR) reconstruction task. Each instance is a de-identified
cohort matrix X of shape N patients x D clinical variables (age, BMI, systolic /
diastolic blood pressure, resting heart rate, fasting glucose, HbA1c, total / HDL /
LDL cholesterol, triglycerides, creatinine, ...). The variables are NOT independent:
they are driven by a small number of latent physiological factors (e.g. a metabolic-
syndrome axis couples BMI, glucose, HbA1c, triglycerides; a cardiovascular axis
couples the blood-pressure and heart-rate columns), so the true matrix is
approximately LOW RANK plus per-patient measurement noise.

A deterministic, seeded subset of entries is HELD OUT (masked). The candidate sees
the matrix with those entries blanked (the public view lists every observed value and
the coordinates of the masked cells) and must return an imputed value for every masked
cell. The score is the RECONSTRUCTION ERROR on the masked cells:

    obj = sqrt( mean over masked (i,j) of  ((pred_ij - true_ij) / sd_j)^2 )

where sd_j is the standard deviation of the OBSERVED entries in column j (so columns on
wildly different clinical scales contribute comparably; this is a normalized RMSE).
Objective = MINIMIZE.

This is genuinely open-ended. Predicting each column's mean (the trivial rule) gives
obj ~ 1.0. Exploiting between-variable correlation does strictly better, and there are
many viable strategies -- nearest-neighbour patients, single-best-correlate regression,
per-column multivariate regression, or iterative low-rank matrix completion
(SoftImpute-style) -- that land at different error levels. The measurement noise floor
means even an oracle low-rank completion cannot drive the error to zero, leaving
headroom.

The candidate is UNTRUSTED model output: it is run in an ISOLATED bwrap subprocess via
`isorun`, sees ONLY the public instance on stdin, and returns ONLY its answer on stdout,
so it can never reach the evaluator's frames / scorer / hidden true values / the seeded
generator source (the synth tree is tmpfs-hidden inside the sandbox).

Scoring (deterministic; no wall-time):
  baseline b = obj of the column-mean imputation (evaluator computes it itself; ~1.0).
  For a VALID answer with objective obj:  r = min(1, 0.1 * b / obj)
  -> the column-mean rule maps to ~0.1; an imputation k times more accurate than the
     column mean maps to min(1, 0.1*k). Malformed / non-finite answer -> 0.

CLI:  python3 evaluator.py <candidate.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def _u01():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return ((state >> 11) & ((1 << 53) - 1)) / float(1 << 53)
    return _u01


def _gauss(u01):
    """Box-Muller standard normal from a uniform-01 generator (deterministic)."""
    u1 = u01()
    if u1 < 1e-12:
        u1 = 1e-12
    u2 = u01()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


# ----------------------------- instance family -----------------------------
# Clinical variable templates: (name, mean, scale). scale ~ typical spread.
_VARS = [
    ("age", 55.0, 15.0), ("bmi", 27.0, 5.0), ("sbp", 128.0, 16.0),
    ("dbp", 80.0, 10.0), ("hr", 72.0, 11.0), ("glucose", 100.0, 22.0),
    ("hba1c", 5.7, 0.9), ("chol_total", 190.0, 38.0), ("hdl", 52.0, 14.0),
    ("ldl", 115.0, 32.0), ("trig", 140.0, 60.0), ("creatinine", 0.95, 0.25),
    ("alt", 28.0, 12.0), ("wbc", 7.2, 1.9), ("hemoglobin", 14.1, 1.5),
    ("sodium", 140.0, 3.0), ("potassium", 4.2, 0.4), ("crp", 3.0, 2.5),
    ("egfr", 88.0, 18.0), ("spo2", 97.0, 1.5),
]


def make_instances():
    """Deterministic, seeded. Returns [{'public':..., 'hidden':{'true':[[...]]}}].

    Each instance: a low-rank latent-factor cohort matrix + per-cell Gaussian noise;
    then a seeded mask of held-out cells. 'harder' instances raise the noise level and
    the latent rank (less recoverable) and enlarge the cohort -- held-out generalization
    that keeps a strong low-rank completer well below the score cap."""
    specs = [
        # seed,  N,  D, rank, noise, mask_frac
        (1101, 45, 12, 3, 0.35, 0.20),
        (1102, 50, 13, 3, 0.35, 0.20),
        (1103, 40, 11, 2, 0.30, 0.22),
        (1104, 55, 14, 4, 0.40, 0.20),
        (1105, 48, 12, 3, 0.38, 0.24),
        (1106, 60, 15, 4, 0.40, 0.20),
        # harder / held-out: bigger, noisier, higher rank
        (1107, 70, 16, 5, 0.50, 0.22),
        (1108, 80, 18, 6, 0.55, 0.20),
        (1109, 75, 17, 5, 0.52, 0.24),
        (1110, 90, 20, 6, 0.60, 0.20),
    ]
    out = []
    for seed, N, D, rank, noise, mask_frac in specs:
        u01 = _rng(seed)
        # latent factors
        U = [[_gauss(u01) for _ in range(rank)] for _ in range(N)]
        V = [[_gauss(u01) for _ in range(D)] for _ in range(rank)]
        names = [_VARS[j][0] for j in range(D)]
        mean = [_VARS[j][1] for j in range(D)]
        scale = [_VARS[j][2] for j in range(D)]
        inv_sqrt_r = 1.0 / math.sqrt(rank)
        X = [[0.0] * D for _ in range(N)]
        for i in range(N):
            for j in range(D):
                s = 0.0
                for a in range(rank):
                    s += U[i][a] * V[a][j]
                s *= inv_sqrt_r                    # ~unit-variance latent signal
                s += noise * _gauss(u01)           # measurement noise
                X[i][j] = mean[j] + scale[j] * s
        # seeded mask of held-out cells (at least 1 per column so every column is tested)
        total = N * D
        n_mask = max(D, int(round(mask_frac * total)))
        masked = set()
        # guarantee one masked cell per column first
        for j in range(D):
            i = int(u01() * N) % N
            masked.add((i, j))
        while len(masked) < n_mask:
            i = int(u01() * N) % N
            j = int(u01() * D) % D
            masked.add((i, j))
        # never mask an entire column (need observed entries to define sd_j / mean_j)
        col_count = [0] * D
        for (i, j) in masked:
            col_count[j] += 1
        masked = [(i, j) for (i, j) in masked if not (col_count[j] >= N)]
        masked.sort()
        mask_list = [[i, j] for (i, j) in masked]
        maskset = set((i, j) for (i, j) in masked)
        # public matrix: observed values, None where masked
        pub_matrix = [[(None if (i, j) in maskset else X[i][j]) for j in range(D)]
                      for i in range(N)]
        public = {
            "N": N, "D": D, "names": names,
            "matrix": pub_matrix,          # None marks a held-out cell
            "masked": mask_list,           # coordinates [i,j] to impute (row-major sorted)
        }
        hidden = {"true": [X[i][j] for (i, j) in masked]}   # aligned to mask_list
        out.append({"public": public, "hidden": hidden})
    return out


# ----------------------------- helpers -------------------------------------
def _col_stats(public):
    """Mean and (population) std of the OBSERVED entries per column."""
    N, D = public["N"], public["D"]
    M = public["matrix"]
    mean = [0.0] * D
    sd = [1.0] * D
    for j in range(D):
        vals = [M[i][j] for i in range(N) if M[i][j] is not None]
        if not vals:
            mean[j], sd[j] = 0.0, 1.0
            continue
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        mean[j] = m
        sd[j] = math.sqrt(var) if var > 1e-12 else 1.0
    return mean, sd


# ----------------------------- scoring -------------------------------------
def baseline(inst):
    """Column-mean imputation error (normalized RMSE over masked cells)."""
    pub = inst["public"]
    true = inst["hidden"]["true"]
    mask = pub["masked"]
    mean, sd = _col_stats(pub)
    acc = 0.0
    for k, (i, j) in enumerate(mask):
        res = (mean[j] - true[k]) / sd[j]
        acc += res * res
    return math.sqrt(acc / len(mask))


def score(inst, answer):
    """Strictly validate the answer; return (ok, obj) with obj = normalized RMSE."""
    pub = inst["public"]
    true = inst["hidden"]["true"]
    mask = pub["masked"]
    if not isinstance(answer, dict):
        return False, None
    preds = answer.get("preds", None)
    if not isinstance(preds, list) or len(preds) != len(mask):
        return False, None
    try:
        preds = [float(x) for x in preds]
    except (TypeError, ValueError):
        return False, None
    for x in preds:
        if not math.isfinite(x):
            return False, None
    mean, sd = _col_stats(pub)
    acc = 0.0
    for k, (i, j) in enumerate(mask):
        res = (preds[k] - true[k]) / sd[j]
        acc += res * res
    obj = math.sqrt(acc / len(mask))
    if not math.isfinite(obj) or obj < 0.0:
        return False, None
    return True, float(obj)


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
            ok, obj = score(inst, ans)
        except Exception:
            ok, obj = False, None
        if not ok or obj is None:
            vec.append(0.0); continue
        b = baseline(inst)
        r = min(1.0, 0.1 * b / max(obj, 1e-12))
        vec.append(r if (r == r and 0.0 <= r <= 1.0) else 0.0)
    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
