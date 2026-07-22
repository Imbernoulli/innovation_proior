# TIER: strong
# Insight (composes all three mechanisms):
#   1. strata-size-estimation: Laplace/Dirichlet-smoothed frequency w_hat_s that
#      survives zero-count classes (never collapses to 0, unlike raw proportions).
#   2. variance estimation that HEDGES under uncertainty: a class's per-stratum
#      sample variance is shrunk toward the pooled burn-in variance (stabilizes
#      noisy small samples) AND then INFLATED the fewer observations it has --
#      "few samples => don't trust that it's safe" -- rather than trusting a small
#      or absent sample estimate at face value like the greedy recipe does.
#   3. variance-balancing-reallocation: Neyman-style slot allocation
#      n_s ~ w_hat_s * sqrt(sigma2_hat_s), which is the allocation that minimizes
#      sum_s w_s^2*sigma_s^2/n_s for FIXED weight*std products -- applied to the
#      hedged estimates from (1)+(2), not the raw burn-in numbers.
# Final allocation is largest-remainder rounded to land on an integer plan that
# uses the full budget K.
import sys, json, math


def main():
    inst = json.load(sys.stdin)
    S = inst["S"]; K = inst["K"]; burnin = inst["burnin"]

    counts = [0] * S
    sums = [0.0] * S
    sqsums = [0.0] * S
    for rec in burnin:
        s = rec["s"]; v = float(rec["v"])
        if 0 <= s < S:
            counts[s] += 1
            sums[s] += v
            sqsums[s] += v * v

    B = len(burnin)

    # Pooled WITHIN-stratum variance prior (correct ANOVA-style pooling: average the
    # per-stratum sample variances, weighted by their own counts -- NOT the variance
    # of all values pooled together, which would conflate between-stratum mean
    # differences with within-stratum spread and systematically overestimate it).
    wsum, wtot = 0.0, 0.0
    for s in range(S):
        c = counts[s]
        if c >= 2:
            m = sums[s] / c
            s2 = max(0.0, sqsums[s] / c - m * m)
            wsum += c * s2
            wtot += c
    global_var = max(1e-6, wsum / wtot) if wtot > 0 else 1.0

    ALPHA = 1.0            # Dirichlet smoothing for the frequency estimate
    PRIOR_N = 1.0          # pseudo-observations of shrinkage toward the pooled variance
    INFLATE_C = 1.5        # extra uncertainty inflation for under-observed strata

    w_hat = [(counts[s] + ALPHA) / (B + S * ALPHA) for s in range(S)]

    scores = []
    for s in range(S):
        c = counts[s]
        if c >= 2:
            m = sums[s] / c
            samp_var = max(0.0, sqsums[s] / c - m * m)
        else:
            samp_var = global_var
        shrunk = (c * samp_var + PRIOR_N * global_var) / (c + PRIOR_N)
        inflated = shrunk * (1.0 + INFLATE_C / (c + 1.0))
        sigma_hat = math.sqrt(max(inflated, 1e-9))
        scores.append(w_hat[s] * sigma_hat)

    tot = sum(scores)
    if tot <= 0:
        cont = [K / S] * S
    else:
        cont = [K * sc / tot for sc in scores]

    floors = [math.floor(x) for x in cont]
    rema = K - sum(floors)
    rema = max(0, min(S, int(round(rema))))
    fracs = sorted(range(S), key=lambda i: (cont[i] - floors[i]), reverse=True)
    alloc = list(floors)
    for i in fracs[:rema]:
        alloc[i] += 1

    print(json.dumps({"alloc": [float(x) for x in alloc]}))


if __name__ == "__main__":
    main()
