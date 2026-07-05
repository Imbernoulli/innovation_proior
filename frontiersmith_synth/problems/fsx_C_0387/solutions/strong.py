# TIER: strong
# Chebyshev semi-iterative step schedule.  Read the operator H, take the range
# [a, b] of the REAL parts of its spectrum (the effective conditioning of the
# monotone saddle field), and set the descent-ascent step sizes to the reciprocals
# of the Chebyshev nodes on [a, b].  This is the classic min-residual acceleration:
# instead of one fixed step it spreads the K steps across the spectrum, driving the
# residual polynomial down at the optimal geometric rate for a real interval.
#
# We use pure descent-ascent (alpha_k = 0), so the extrapolation eval is spent as a
# second field evaluation at the SAME point -- i.e. z_{k+1} = z_k - beta_k V(z_k) --
# which is stable here because the coupling is moderate.  The state dimension (2d)
# exceeds 2K, so the Chebyshev polynomial cannot annihilate the whole spectrum and
# the residual stays strictly positive: strong but never perfect.
import sys, json, math

inst = json.load(sys.stdin)
K = inst["K"]

try:
    import numpy as np
    H = np.array(inst["H"], dtype=float)
    w = np.linalg.eigvals(H)
    re = w.real
    a = float(re.min())
    b = float(re.max())
    if not math.isfinite(a) or a <= 0.0:
        a = 1e-6
    if not math.isfinite(b) or b <= a:
        b = a * 10.0
    beta = []
    for k in range(K):
        node = (a + b) / 2.0 + (b - a) / 2.0 * math.cos((2 * k + 1) * math.pi / (2 * K))
        if node < 1e-9:
            node = 1e-9
        # round to kill any last-bit jitter -> deterministic schedule
        beta.append(round(1.0 / node, 12))
    alpha = [0.0] * K
except Exception:
    # numpy unavailable -> fall back to the reference schedule (still valid)
    ra = inst["ref_alpha"]
    alpha = [ra] * K
    beta = [ra] * K

print(json.dumps({"alpha": alpha, "beta": beta}))
