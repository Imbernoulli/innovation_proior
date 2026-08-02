#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic grader for the royalty-audit-sample
problem (format C). <ans> is an empty placeholder, ignored.

Reads the PUBLIC instance from <in> (population, budget, defensibility threshold).
Reads the participant's audit PLAN (a set of transaction ids to audit) from <out>.
Regenerates the PRIVATE ground-truth error rate for every transaction deterministically
from (testId, id, stratum) -- the same private formula every time, never printed to the
solver. Scores DEFENSIBLE recoverable value per stratum:

  * full census (100% of a stratum audited)  -> exact true stratum total (certain, no
    extrapolation risk at all -- always fully claimable).
  * partial audit (n>=2 in that stratum) that is BOTH (a) "representative": the mean
    value-percentile-rank of the audited items within that stratum's own value range
    falls in [0.37, 0.63] (i.e. not a judgmental top-only or bottom-only pick), AND
    (b) "precise": the relative margin RM = Z*sd/(sqrt(n)*mean_rate) computed from the
    sample's OWN observed error-rate dispersion is <= this test's threshold
      -> extrapolated claim = sample_mean_error_rate * stratum_total_reported_value.
  * otherwise (n<2, OR fails representativeness, OR fails precision)
      -> a heavily discounted CONSOLATION credit = 0.32 * (actual dollars found among
         the audited items in that stratum) -- real money, but not a defensible
         population-level claim, so it cannot be projected.

Objective = sum of per-stratum claims, maximize, subject to total audit cost <= budget.
Prints exactly one line ending "Ratio: <r>" with r in [0,1]. Any infeasibility (bad
token, duplicate/out-of-range id, non-finite value, over budget, empty output)
=> "Ratio: 0.0".
"""
import sys, math, random

STRATA = [
    (0.020, 0.65, 40000, 140000, 0.010, 60, 140, 0.14),   # 0 Marquee
    (0.055, 0.60,  3000,  18000, 0.020, 15,  40, 0.34),   # 1 Standard
    (0.110, 0.55,   150,  2200,  0.035,  3,  12, 0.52),   # 2 LongTail
]
Z = 1.645
PCT_LO, PCT_HI = 0.37, 0.63
CONSOLATION = 0.32
MAX_TOKENS = 200000


def bail(reason):
    print("infeasible: %s -- Ratio: 0.0" % reason)
    sys.exit(0)


def beta_ab(mean, cv):
    var = (cv * mean) ** 2
    s = mean * (1 - mean) / var - 1
    s = max(s, 2.0)
    a = mean * s
    b = s - a
    return max(a, 0.5), max(b, 0.5)


def hidden_rate(t, tid, stratum):
    """Private ground-truth error rate for transaction `tid` in `stratum`, on test `t`.
    Deterministic (fixed seed per (t,tid,stratum)); identical on every rerun."""
    rng = random.Random(500000 + 100003 * t + 7919 * tid + 31 * stratum)
    mean, cv = STRATA[stratum][0], STRATA[stratum][1]
    a, b = beta_ab(mean, cv)
    return rng.betavariate(a, b)


def read_instance(path):
    try:
        toks = open(path, "r", errors="replace").read().split()
    except Exception:
        bail("cannot read input")
    it = iter(toks)

    def nxt():
        try:
            return next(it)
        except StopIteration:
            bail("truncated input")

    t = int(nxt()); N = int(nxt()); K = int(nxt()); Cmax = int(nxt())
    thresh = float(nxt())
    for _ in range(K):
        nxt(); nxt()  # priors (informational only, not used by the grader)
    rows = []
    for _ in range(N):
        tid = int(nxt()); h = int(nxt()); v = int(nxt()); cost = int(nxt())
        rows.append((tid, h, v, cost))
    return t, N, K, Cmax, thresh, rows


def read_plan(path, N):
    try:
        raw = open(path, "r", errors="replace").read()
    except Exception:
        bail("cannot read output")
    toks = raw.split()
    if not toks:
        bail("empty output")
    if len(toks) > MAX_TOKENS:
        bail("too many tokens")
    ids = []
    seen = set()
    for tok in toks:
        try:
            v = int(tok)
        except ValueError:
            bail("non-integer token %r" % tok[:40])
        if v < 1 or v > N:
            bail("id %d out of range [1,%d]" % (v, N))
        if v in seen:
            bail("duplicate id %d" % v)
        seen.add(v)
        ids.append(v)
    return ids


def main():
    if len(sys.argv) < 3:
        bail("usage")
    t, N, K, Cmax, thresh, rows = read_instance(sys.argv[1])
    ids = read_plan(sys.argv[2], N)

    by_id = {r[0]: r for r in rows}
    audited = set(ids)

    total_cost = 0
    for i in ids:
        total_cost += by_id[i][3]
    if total_cost > Cmax:
        bail("audit cost %d exceeds budget %d" % (total_cost, Cmax))

    members = {h: [r for r in rows if r[1] == h] for h in range(K)}
    V = {h: sum(r[2] for r in members[h]) for h in range(K)}

    total = 0.0
    parts = []
    for h in range(K):
        pop_h = members[h]
        nb = len(pop_h)
        by_val = sorted(pop_h, key=lambda r: (r[2], r[0]))
        pctl_of = {r[0]: (i / (nb - 1) if nb > 1 else 0.5) for i, r in enumerate(by_val)}

        chosen_h = [r for r in pop_h if r[0] in audited]
        n = len(chosen_h)
        if n == 0:
            claim = 0.0
        elif n == nb:
            # full census: exact truth, zero extrapolation risk
            claim = sum(hidden_rate(t, r[0], h) * r[2] for r in pop_h)
        else:
            rates = [hidden_rate(t, r[0], h) for r in chosen_h]
            values = [r[2] for r in chosen_h]
            audited_actual = sum(rt * vv for rt, vv in zip(rates, values))
            mean_pctl = sum(pctl_of[r[0]] for r in chosen_h) / n
            representative = (PCT_LO <= mean_pctl <= PCT_HI)
            if n < 2 or not representative:
                claim = CONSOLATION * audited_actual
            else:
                mean_r = sum(rates) / n
                var = sum((rt - mean_r) ** 2 for rt in rates) / (n - 1)
                sd = math.sqrt(var)
                RM = Z * sd / (math.sqrt(n) * max(mean_r, 1e-6))
                if RM <= thresh:
                    claim = mean_r * V[h]
                else:
                    claim = CONSOLATION * audited_actual
        if not math.isfinite(claim):
            bail("non-finite claim computed (stratum %d)" % h)
        total += claim
        parts.append(claim)

    # internal baseline B: a lazy full census of the SINGLE cheapest-to-fully-audit
    # stratum, nothing else (a trivial, always-feasible reference construction)
    full_cost = {h: sum(r[3] for r in members[h]) for h in range(K)}
    h0 = min(full_cost, key=lambda h: full_cost[h])
    B = sum(hidden_rate(t, r[0], h0) * r[2] for r in members[h0])
    B = max(B, 1e-9)

    sc = min(1000.0, 100.0 * total / max(1e-9, B))
    ratio = sc / 1000.0
    print("claim=%.2f B=%.2f parts=%s Ratio: %.6f" % (total, B, [round(p, 1) for p in parts], ratio))
    sys.exit(0)


if __name__ == "__main__":
    main()
