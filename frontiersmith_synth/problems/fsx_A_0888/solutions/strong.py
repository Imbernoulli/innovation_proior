# TIER: strong
"""
The insight: don't fit each sieve size's curve on its own.  Pool ALL training
rows -- every (L, p, Pi_hat) triple across every small sieve size at once --
into a SINGLE joint regression for (pc, nu).  The ansatz

    Pi(p, L) = OFF + AMP * sigmoid( (p - pc) * L**(1/nu) )

ties the threshold's drift toward pc and the transition's sharpening
together through the ONE exponent 1/nu.  Fitting (pc, nu) jointly uses every
row as evidence for the SAME two numbers, instead of four independent,
noisy per-L curves -- far more data per parameter, and (crucially) it
extrapolates correctly to L=128/512 because the fitted law genuinely IS a
power law in L, not a low-order polynomial standing in for one.

We recover (pc, nu) with a coarse deterministic grid search over (pc, 1/nu)
followed by local coordinate-descent refinement on the pooled squared error
-- no external optimisation libraries, fully deterministic given the input.
"""
import sys, math

OFF, AMP = 0.1, 0.8


def sig(x):
    x = max(-60.0, min(60.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def pooled_sse(rows, pc, invnu):
    se = 0.0
    for L, p, pih in rows:
        x = (p - pc) * (L ** invnu)
        pred = OFF + AMP * sig(x)
        se += (pred - pih) ** 2
    return se


def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    _t = int(data[idx]); idx += 1
    rows = []
    for _ in range(n):
        L = int(data[idx]); p = float(data[idx + 1]); pih = float(data[idx + 2])
        idx += 3
        rows.append((L, p, pih))

    # ---- coarse deterministic grid search over the joint (pc, 1/nu) ----
    best = None
    PC_GRID = [0.10 + 0.02 * i for i in range(41)]        # 0.10 .. 0.90
    INVNU_GRID = [0.25 + 0.05 * i for i in range(24)]      # 0.25 .. 1.40
    for pc in PC_GRID:
        for invnu in INVNU_GRID:
            e = pooled_sse(rows, pc, invnu)
            if best is None or e < best[0]:
                best = (e, pc, invnu)
    _, pc, invnu = best

    # ---- local coordinate-descent refinement (deterministic, shrinking step) ----
    step_pc, step_inu = 0.02, 0.05
    cur = pooled_sse(rows, pc, invnu)
    for _round in range(60):
        improved = False
        for dpc in (step_pc, -step_pc, 0.0):
            for dinu in (step_inu, -step_inu, 0.0):
                if dpc == 0.0 and dinu == 0.0:
                    continue
                npc = min(0.98, max(0.02, pc + dpc))
                ninu = min(2.0, max(0.15, invnu + dinu))
                e = pooled_sse(rows, npc, ninu)
                if e < cur - 1e-12:
                    cur, pc, invnu = e, npc, ninu
                    improved = True
        if not improved:
            step_pc *= 0.5
            step_inu *= 0.5
            if step_pc < 1e-5 and step_inu < 1e-5:
                break

    expr = "%.6f + %.6f * sig( (p - %.6f) * pw(L, %.6f) )" % (OFF, AMP, pc, invnu)
    print(expr)


if __name__ == "__main__":
    main()
