#!/usr/bin/env python3
"""verify.py <in> <out> <ans>  --  deterministic scorer for the membrane
selectivity-design problem (family: membrane-selectivity-design).

Instance tokens (12 floats/ints, whitespace separated):
  d_T d_C chi_T chi_C base_sol_T base_sol_C K_max r_min r_max alpha_max
  delta_coat P_min

Participant artifact (the membrane design), whitespace separated tokens:
  K alpha r_1 w_1 r_2 w_2 ... r_K w_K

Feasibility (any violation -> Ratio: 0.0):
  * output parses into an integer K in [1, K_max] and finite floats alpha,
    (r_k, w_k) for k=1..K
  * alpha in [0, alpha_max]
  * every r_k in [r_min, r_max]
  * every w_k >= 0 and sum_k w_k == 1 (area fractions of a distribution)

Transport model (both mechanisms fold into one objective):
  * pore-size channel: r_eff_k = r_k - delta_coat*alpha (functionalization
    coats the pore wall). lam_s,k = d_s / (2*r_eff_k). Steric passage
    fraction D(lam) = 1/(1+exp(BETA*(lam-1))) -- smooth, monotone in r,
    never exactly 0 or 1 (the permeability-selectivity bound: shrinking a
    pore always buys some selectivity but never for free).
  * solubility channel: Sol_s(alpha) = base_sol_s * (1 + alpha*chi_s) -- an
    independent chemical-affinity axis, boosted for the target (chi_T>0)
    and suppressed for the twin (chi_C<0) as functionalization loading
    alpha increases.
  * permeability: P_s = sum_k w_k * D(lam_s,k) * Sol_s(alpha).
  * separation factor S = P_T / P_C.
  * throughput factor thr = min(1, P_T / P_min) -- realized separation is
    discounted proportionally if the required target throughput is missed.
  * objective F = S * thr.

Internal baseline B = F of the checker's own "wide open, no chemistry"
construction (K=1, r=r_max, w=1, alpha=0). Maximization:
  sc = min(1000, 100*F/max(1e-9,B));  Ratio = sc/1000.
"""
import sys
import math

BETA = 4.0


def read_tokens(path):
    with open(path) as f:
        return f.read().split()


def sigmoid_D(lam):
    x = BETA * (lam - 1.0)
    if x > 700.0:
        return 0.0
    if x < -700.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(x))


def main():
    inf, outf = sys.argv[1], sys.argv[2]

    itoks = read_tokens(inf)
    (d_T, d_C, chi_T, chi_C, base_sol_T, base_sol_C,
     K_max_f, r_min, r_max, alpha_max, delta_coat, P_min) = [float(x) for x in itoks[:12]]
    K_max = int(round(K_max_f))

    def transport(alpha, pairs):
        Sol_T = max(0.0, base_sol_T * (1.0 + alpha * chi_T))
        Sol_C = max(0.0, base_sol_C * (1.0 + alpha * chi_C))
        P_T = 0.0
        P_C = 0.0
        for (r, w) in pairs:
            r_eff = r - delta_coat * alpha
            if r_eff <= 1e-9:
                D_T = 0.0
                D_C = 0.0
            else:
                D_T = sigmoid_D(d_T / (2.0 * r_eff))
                D_C = sigmoid_D(d_C / (2.0 * r_eff))
            P_T += w * D_T * Sol_T
            P_C += w * D_C * Sol_C
        S = P_T / max(P_C, 1e-12)
        thr = min(1.0, P_T / P_min) if P_min > 0 else 1.0
        return P_T, P_C, S * thr

    # ---- checker's own baseline: wide-open single pore, no functionalization ----
    _, _, B = transport(0.0, [(r_max, 1.0)])
    B = max(B, 1e-9)

    # ---- parse participant output ----
    try:
        otoks = read_tokens(outf)
        idx = 0
        K = int(otoks[idx]); idx += 1
        alpha = float(otoks[idx]); idx += 1
        pairs = []
        for _ in range(K):
            r = float(otoks[idx]); idx += 1
            w = float(otoks[idx]); idx += 1
            pairs.append((r, w))
    except Exception:
        print("Ratio: 0.0 (unparseable output)")
        return

    if not (1 <= K <= K_max):
        print("Ratio: 0.0 (K=%d out of range 1..%d)" % (K, K_max))
        return

    if not math.isfinite(alpha):
        print("Ratio: 0.0 (alpha not finite)")
        return
    if alpha < -1e-9 or alpha > alpha_max + 1e-9:
        print("Ratio: 0.0 (alpha=%.6f out of range [0,%.6f])" % (alpha, alpha_max))
        return
    alpha = min(max(alpha, 0.0), alpha_max)

    wsum = 0.0
    for (r, w) in pairs:
        if not (math.isfinite(r) and math.isfinite(w)):
            print("Ratio: 0.0 (non-finite pore parameter)")
            return
        if r < r_min - 1e-9 or r > r_max + 1e-9:
            print("Ratio: 0.0 (radius %.6f outside [%.6f, %.6f])" % (r, r_min, r_max))
            return
        if w < -1e-9:
            print("Ratio: 0.0 (negative area fraction %.6f)" % w)
            return
        wsum += w

    if abs(wsum - 1.0) > 1e-6:
        print("Ratio: 0.0 (area fractions sum to %.6f, must equal 1)" % wsum)
        return

    P_T, P_C, F = transport(alpha, pairs)
    if not math.isfinite(F) or F < 0:
        print("Ratio: 0.0 (non-finite objective)")
        return

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("Ratio: %.6f (F=%.6f B=%.6f P_T=%.6f P_C=%.6f)" % (sc / 1000.0, F, B, P_T, P_C))


if __name__ == "__main__":
    main()
