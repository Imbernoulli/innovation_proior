# TIER: greedy
"""The obvious first attempt: use a generic local-averaging/diffusive coupling
kernel (positive weights, monotonically decaying with distance -- the kind of
"coupling" most people reach for without doing any spectral analysis), and
spend the whole bias budget on ONE strong, spatially localized perturbation
(a single bump) hoping that a big-enough seed will force the target shape.

This is the described trap: a diffusive kernel's dispersion relation is
monotonically decreasing in wavenumber, so under the zero-mean dynamics it
always favors the LOWEST surviving mode -- a single domain (k=1) -- no matter
what k is requested, and no matter how the (small, budget-capped) bias is
aimed. It looks like a completely reasonable heuristic and even solves the
k=1 case outright; it just cannot generalize to other targets."""
import sys, json, math


def main():
    inst = json.load(sys.stdin)
    n = inst["N"]
    Lmax = inst["L_max"]
    wmax = inst["W_max"]
    bbud = inst["B"]

    decay = 0.25
    kernel = [min(math.exp(-decay * j), wmax) for j in range(Lmax + 1)]

    width = 2.0
    center = 0
    gauss = [math.exp(-(min((i - center) % n, (center - i) % n) ** 2) / (2 * width * width))
             for i in range(n)]
    norm = math.sqrt(sum(g * g for g in gauss))
    if norm > 1e-12:
        bias = [g / norm * bbud for g in gauss]
    else:
        bias = [0.0] * n

    print(json.dumps({"kernel": kernel, "bias": bias}))


main()
