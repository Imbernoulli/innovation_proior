# TIER: strong
# Insight: this is a maximin resource-allocation problem (minimize the WORST posterior
# variance == maximize the worst quantity's posterior information), so spending on
# whatever measurement looks cheapest for ONE quantity in isolation (the greedy
# recipe) is the wrong objective. Water-fill the information matrix one integer unit
# at a time: after every single unit committed, recompute which quantity is CURRENTLY
# worst (real posterior info, prior included) and fund whichever available
# measurement gives THAT quantity the largest marginal gain. Because state is
# refreshed after every unit, a shared measurement's cross-coverage is naturally
# "funded once": one run raises every quantity it covers simultaneously, so as soon as
# a previously-worst quantity is lifted the very next unit re-targets whatever is now
# the new bottleneck instead of continuing to pour more budget into an
# already-adequately-covered quantity's steep-looking-but-irrelevant private probe.
# Every step commits a whole integer run (never a fractional dial setting) and
# immediately re-derives the next target from the freshly repaired state, so the
# integer constraint is handled directly rather than needing a separate rounding pass.
# If no measurement still covers the current worst quantity (all capped out), fall
# back to whichever measurement's next unit yields the largest total VARIANCE
# reduction summed over every quantity it covers -- weighting each quantity's
# marginal info gain by 1/info_q^2, the exact derivative of variance -- so budget is
# never simply wasted once the immediate bottleneck is unreachable.
import sys, json


def gain(a, b, n):
    if n <= 0:
        return 0.0
    return a * n / (n + b)


def dgain(a, b, n):
    return gain(a, b, n + 1) - gain(a, b, n)


inst = json.load(sys.stdin)
Q = inst["n_quantities"]
M = inst["n_measurements"]
B = inst["budget"]
cap = inst["cap"]
a = inst["gain_a"]
b = inst["gain_b"]
cov = inst["coverage"]
prior = inst["prior_precision"]


def infos_of(alloc):
    out = list(prior)
    for m in range(M):
        g = gain(a[m], b[m], alloc[m])
        if g == 0.0:
            continue
        row = cov[m]
        for q in range(Q):
            w = row[q]
            if w:
                out[q] += w * g
    return out


alloc = [0] * M
for _ in range(B):
    infos = infos_of(alloc)
    worst_q = min(range(Q), key=lambda q: infos[q])
    best_m, best_delta = None, -1.0
    for m in range(M):
        if alloc[m] >= cap[m]:
            continue
        w = cov[m][worst_q]
        if w <= 0:
            continue
        delta = w * dgain(a[m], b[m], alloc[m])
        if delta > best_delta:
            best_delta = delta
            best_m = m
    if best_m is None:
        best_val = -1.0
        for m in range(M):
            if alloc[m] >= cap[m]:
                continue
            val = 0.0
            for q in range(Q):
                wq = cov[m][q]
                if wq:
                    val += wq * dgain(a[m], b[m], alloc[m]) / (infos[q] ** 2)
            if val > best_val:
                best_val = val
                best_m = m
    if best_m is None:
        break
    alloc[best_m] += 1

print(json.dumps({"alloc": alloc}))
