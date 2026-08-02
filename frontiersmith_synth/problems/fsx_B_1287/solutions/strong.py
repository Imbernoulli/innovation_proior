# TIER: strong
"""No-trade band: the insight is that continuous delta-neutrality is not worth its
transaction cost. Only rebalance once the drift error leaves a band whose width
scales as sqrt(cost / gamma) -- wide where gamma is low (drift is cheap, trading is
not worth it) and narrow where gamma is high (drift is expensive, worth correcting).
When the band is breached, move only to the NEAREST band edge (not all the way to
the target) -- this is the classical local-time / impulse-control shape for control
under proportional+fixed action costs.

Crucially, this also encodes acceptance of jump-risk residual: the drift cost already
incurred moving from t-1 to t is sunk (it was priced with the position h[t-1] that had
to be chosen before the jump was seen) -- no trade executed at time t can undo it, so
chasing a jump (or a whipsaw that mostly reverts next step) all the way to the new
target is often just wasted transaction cost."""
import sys, math

G_MIN = 1e-6
K_BAND = 0.3


def main():
    toks = sys.stdin.read().split()
    idx = 0
    N = int(toks[idx]); idx += 1
    S = [float(toks[idx + i]) for i in range(N + 1)]; idx += N + 1
    D = [float(toks[idx + i]) for i in range(N + 1)]; idx += N + 1
    G = [float(toks[idx + i]) for i in range(N + 1)]; idx += N + 1
    cost_prop = float(toks[idx]); idx += 1
    cost_fixed = float(toks[idx]); idx += 1

    out = []
    prev = D[0]
    for t in range(1, N + 1):
        band = K_BAND * math.sqrt((cost_fixed + cost_prop * S[t]) / max(G[t], G_MIN))
        target = D[t]
        if prev < target - band:
            newh = target - band
        elif prev > target + band:
            newh = target + band
        else:
            newh = prev
        out.append(newh)
        prev = newh

    print(" ".join("%.10g" % v for v in out))


if __name__ == "__main__":
    main()
