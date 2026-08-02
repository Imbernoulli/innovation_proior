# TIER: greedy
"""The obvious approach: blend to hit the TARGET SCENT PROFILE AT TIME ZERO
ONLY. This is the textbook fragrance-matching recipe -- treat perceived
intensity as simply equal to concentration (ignore evaporation entirely,
ignore masking entirely, i.e. pretend t=0 is the whole story) and run
greedy matching pursuit: repeatedly pick the ingredient whose descriptor
direction best reduces the remaining t=0 residual, add as much of it as
its IFRA cap and the total budget allow, and stop when nothing helps.

This reliably nails the profile at t=0 -- but the resulting blend is never
reconsidered against how it will actually smell an hour, two hours, or
four hours later. A blend loaded up on the fast-evaporating ingredient
that matches the t=0 target best will have essentially evaporated by
t=2h, and nothing was ever allocated to cover the heart/base target
phases, because this optimizer never looked past t=0.
"""
import sys


def main():
    toks = sys.stdin.read().split()
    p = 0
    K = int(toks[p]); p += 1
    D = int(toks[p]); p += 1
    T = int(toks[p]); p += 1
    descs, caps = [], []
    for _ in range(K):
        desc = [float(toks[p + a]) for a in range(D)]
        p += D
        p += 1  # k (decay rate) -- ignored by this solver
        cap = float(toks[p]); p += 1
        descs.append(desc); caps.append(cap)
    p += K * K  # mask table -- ignored by this solver
    p += T       # times
    g0 = [float(toks[p + a]) for a in range(D)]
    # remaining checkpoint targets are never read -- t=0-only matching

    TOTAL_CAP = 1.0
    alloc = [0.0] * K
    budget = TOTAL_CAP
    residual = list(g0)

    MAX_PASSES = 3 * K + 2
    for _ in range(MAX_PASSES):
        if budget <= 1e-9:
            break
        best_i, best_x, best_gain = -1, 0.0, 0.0
        for i in range(K):
            room = caps[i] - alloc[i]
            if room <= 1e-9:
                continue
            room = min(room, budget)
            dv = descs[i]
            dot = sum(residual[a] * dv[a] for a in range(D))
            nrm = sum(dv[a] * dv[a] for a in range(D))
            if nrm <= 1e-12:
                continue
            x_star = dot / nrm
            x = min(max(x_star, 0.0), room)
            if x <= 1e-9:
                continue
            # gain = reduction in ||residual||^2 from moving x along dv
            gain = 2.0 * x * dot - x * x * nrm
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_i = i
                best_x = x
        if best_i < 0:
            break
        dv = descs[best_i]
        for a in range(D):
            residual[a] -= best_x * dv[a]
        alloc[best_i] += best_x
        budget -= best_x

    used = [(i + 1, alloc[i]) for i in range(K) if alloc[i] > 1e-9]
    if not used:
        used = [(1, min(caps[0], TOTAL_CAP))]

    out = [str(len(used))]
    for idx, c in used:
        out.append("%d %.6f" % (idx, c))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
