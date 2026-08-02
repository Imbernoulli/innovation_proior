# TIER: strong
"""The insight: don't fight the losing size-sieving battle when the twin is
almost the same size as the target -- exploit the SOLUBILITY / chemical-
affinity channel instead (functionalization loading alpha), which moves
along an independent trade-off curve from the geometric pore-size channel.

Reformulation: for a FIXED alpha, maximizing S = P_T/P_C over a single pore
radius is a pointwise ratio-maximization that is always improved by
shrinking r (monotone), so the real design freedom is a 2-point pore-size
DISTRIBUTION: blend a "selective" radius r* (chosen to maximize the raw
D_T/D_C ratio) with the wide-open radius r_max (which maximizes raw
permeability), using just enough weight on r_max to clear the P_min
throughput requirement -- and simultaneously dial up alpha to (a) boost the
target's own solubility (which ALSO buys throughput headroom, letting you
lean harder on the small, more selective radius) and (b) suppress the
twin's solubility multiplicatively, on top of whatever the geometric ratio
achieves. This is a genuine 2-channel joint optimization, not "greedy plus
more search" along the size axis alone: a deterministic coarse grid search
over (alpha, r*, blend weight w) that always considers alpha > 0."""
import sys
import math

BETA = 4.0
N_ALPHA = 25
N_R = 121
N_W = 21


def sigmoid_D(lam):
    x = BETA * (lam - 1.0)
    if x > 700:
        return 0.0
    if x < -700:
        return 1.0
    return 1.0 / (1.0 + math.exp(x))


def main():
    toks = sys.stdin.read().split()
    (d_T, d_C, chi_T, chi_C, base_sol_T, base_sol_C,
     K_max_f, r_min, r_max, alpha_max, delta_coat, P_min) = [float(x) for x in toks[:12]]

    def transport(alpha, pairs):
        Sol_T = max(0.0, base_sol_T * (1.0 + alpha * chi_T))
        Sol_C = max(0.0, base_sol_C * (1.0 + alpha * chi_C))
        P_T = 0.0
        P_C = 0.0
        for (r, w) in pairs:
            r_eff = r - delta_coat * alpha
            if r_eff <= 1e-9:
                D_T = D_C = 0.0
            else:
                D_T = sigmoid_D(d_T / (2.0 * r_eff))
                D_C = sigmoid_D(d_C / (2.0 * r_eff))
            P_T += w * D_T * Sol_T
            P_C += w * D_C * Sol_C
        S = P_T / max(P_C, 1e-12)
        thr = min(1.0, P_T / P_min) if P_min > 0 else 1.0
        return S * thr

    best_F = -1.0
    best_alpha = 0.0
    best_pairs = [(r_max, 1.0)]

    for ia in range(N_ALPHA):
        alpha = alpha_max * ia / (N_ALPHA - 1)
        for ir in range(N_R):
            rstar = r_min + (r_max - r_min) * ir / (N_R - 1)
            for iw in range(N_W):
                w = iw / (N_W - 1)
                pairs = [(rstar, w), (r_max, 1.0 - w)]
                F = transport(alpha, pairs)
                if F > best_F:
                    best_F = F
                    best_alpha = alpha
                    best_pairs = pairs

    # emit only pore families with nonzero weight (still a valid distribution)
    kept = [(r, w) for (r, w) in best_pairs if w > 1e-9]
    if not kept:
        kept = [(r_max, 1.0)]
    wsum = sum(w for _, w in kept)
    kept = [(r, w / wsum) for (r, w) in kept]

    print(len(kept))
    print("%.6f" % best_alpha)
    for (r, w) in kept:
        print("%.6f %.6f" % (r, w))


if __name__ == "__main__":
    main()
