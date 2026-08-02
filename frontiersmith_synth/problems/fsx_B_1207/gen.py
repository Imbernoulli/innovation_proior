#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE pre-treatment surveillance record to stdout.

Family: antibiotic-resistance-forecast.  A bacterial population carries a
plasmid-borne resistance gene.  Resistant cells pay a fitness cost `c` (hidden,
NEVER printed), so absent drug pressure the resistant frequency `p(t)` settles
at a low mutation/transfer-selection equilibrium.  Once a course of antibiotics
begins, drug-driven selection can outweigh the cost and the resistant lineage
sweeps toward fixation -- but NONE of that is visible in this record: every
row here is measured strictly BEFORE treatment starts (t < T0).

STDOUT prints ONLY: mu, tau, alpha, D, T0, T1 (all instance parameters the
solver needs), the noisy pre-treatment observations, and the query times that
will be scored (their true values are held out).  The hidden fitness cost `c`
and the equilibrium-implied intermediate quantities are NOT printed anywhere.
"""
import sys, random, math

T0 = 20.0
T1 = 24.0
QUERY_FRACS = [0.03, 0.08, 0.15, 0.25, 0.38, 0.52, 0.68, 0.84, 1.0]

# per-testId difficulty/regime ladder: (mult_lo, mult_hi, noise_sd, n_train)
# mult = D / D_threshold, where D_threshold is the dose at which drug-driven
# selection exactly cancels the fitness cost (mult<1 -> sub-therapeutic, no
# sweep; mult>1 -> therapeutic, resistant lineage sweeps).
LADDER = [
    (0.30, 0.70, 0.020, 20),   # 1  calm: dose too low to matter
    (0.35, 0.80, 0.025, 18),   # 2  calm
    (0.90, 1.30, 0.025, 18),   # 3  borderline, mild drift
    (1.20, 1.80, 0.030, 16),   # 4  mild sweep
    (1.50, 2.20, 0.030, 16),   # 5  moderate sweep
    (1.80, 2.50, 0.035, 14),   # 6  moderate sweep
    (2.20, 3.00, 0.035, 14),   # 7  strong sweep
    (2.50, 3.50, 0.045, 12),   # 8  strong sweep, noisier
    (3.00, 4.00, 0.045, 12),   # 9  near-saturating sweep
    (3.50, 4.60, 0.050, 10),   # 10 adversarial: fast sweep, sparse+noisy train
]


def build_instance(test_id):
    """Deterministic instance construction shared (duplicated) by gen.py and
    verify.py. Returns a dict; c/A/B/R/K0 are the hidden ground truth and are
    used ONLY inside verify.py -- gen.py never prints them."""
    lo_idx = max(0, min(len(LADDER) - 1, test_id - 1))
    mult_lo, mult_hi, noise_sd, n_train = LADDER[lo_idx]
    rng = random.Random(90173 + test_id * 7919)

    for _attempt in range(500):
        mu = rng.uniform(1.0e-3, 4.0e-3)
        tau = 0.0 if rng.random() < 0.15 else rng.uniform(0.0, 3.0e-3)
        p0 = rng.uniform(0.03, 0.50)
        A = mu + tau
        c = A / p0
        alpha = rng.uniform(0.8, 2.2)
        d_thresh = c / alpha
        mult = rng.uniform(mult_lo, mult_hi)
        D = mult * d_thresh
        B = alpha * D - c
        R = A + B
        if abs(R) < 0.01:
            continue
        K0 = (A + B * p0) / (1.0 - p0)
        if K0 <= 0:
            continue

        query_times = [T0 + f * T1 for f in QUERY_FRACS]
        ok = True
        for t in query_times:
            M = K0 * math.exp(R * (t - T0))
            denom = M + B
            if abs(denom) < 1e-6:
                ok = False
                break
            pv = (M - A) / denom
            if not math.isfinite(pv) or not (0.0 <= pv <= 1.0):
                ok = False
                break
        if not ok:
            continue

        train_times = sorted(rng.uniform(0.5, T0 - 0.5) for _ in range(n_train))
        train_obs = []
        for _t in train_times:
            noisy = p0 + rng.gauss(0.0, noise_sd)
            train_obs.append(max(0.0, min(1.0, noisy)))

        return dict(mu=mu, tau=tau, alpha=alpha, D=D, T0=T0, T1=T1,
                    train_times=train_times, train_obs=train_obs,
                    query_times=query_times, p0=p0, c=c, A=A, B=B, R=R, K0=K0)
    raise RuntimeError("failed to build a well-posed instance for testId=%d" % test_id)


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    inst = build_instance(t)

    lines = []
    lines.append("%d" % t)
    lines.append("%.9f %.9f %.9f %.9f %.6f %.6f" %
                  (inst["mu"], inst["tau"], inst["alpha"], inst["D"], inst["T0"], inst["T1"]))
    lines.append(str(len(inst["train_times"])))
    for tt, yy in zip(inst["train_times"], inst["train_obs"]):
        lines.append("%.6f %.6f" % (tt, yy))
    lines.append(str(len(inst["query_times"])))
    for q in inst["query_times"]:
        lines.append("%.6f" % q)
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
