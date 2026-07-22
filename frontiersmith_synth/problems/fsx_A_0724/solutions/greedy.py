# TIER: greedy
# The obvious "rank options by unit cost, spend top-down" recipe: for EACH
# measurement, compute its best-case per-quantity initial marginal slope
# max_q(coverage[m][q] * gain_a[m]/gain_b[m]) -- how much info-per-run it looks worth
# for whichever single quantity it serves best -- completely ignoring how much prior
# precision that quantity already has, and ignoring that a measurement might ALSO
# help several OTHER quantities at once (a shared probe's cross-coverage is never
# counted as a bonus, only its best single-quantity slope matters for its rank).
# Measurements are then funded, one after another, in that FIXED steepest-first order
# up to their caps until the budget runs out -- a static, one-shot ranking that is
# never revisited as the true state (who is actually still the worst quantity)
# evolves.
#
# This wins comfortably on calm instances, where the steepest-looking probes really
# are close to the right ones and everything gets funded eventually anyway. But on
# trap instances a quantity that is ALREADY well-estimated (high prior precision) can
# still own the single steepest-slope private probe (its curve just happens to have a
# tiny denominator), so this recipe spends its FIRST, most valuable budget units on an
# already-fine quantity -- "over-investing in an already-pinned quantity" -- before
# ever reaching the genuinely needy quantities or the efficient shared probe, which
# rank lower simply because their initial slope (viewed one quantity at a time) looks
# less steep. Since the objective is the WORST quantity's variance, whichever quantity
# is left starved decides the score.
import sys, json


def gain(a, b, n):
    if n <= 0:
        return 0.0
    return a * n / (n + b)


inst = json.load(sys.stdin)
Q = inst["n_quantities"]
M = inst["n_measurements"]
B = inst["budget"]
cap = inst["cap"]
a = inst["gain_a"]
b = inst["gain_b"]
cov = inst["coverage"]

val = []
for m in range(M):
    best = -1.0
    for q in range(Q):
        w = cov[m][q]
        if w <= 0:
            continue
        s = w * (a[m] / b[m]) if b[m] > 0 else float("inf")
        if s > best:
            best = s
    val.append(best)

order = sorted(range(M), key=lambda m: -val[m])

alloc = [0] * M
budget = B
for m in order:
    if budget <= 0:
        break
    take = min(cap[m] - alloc[m], budget)
    if take > 0:
        alloc[m] += take
        budget -= take

print(json.dumps({"alloc": alloc}))
