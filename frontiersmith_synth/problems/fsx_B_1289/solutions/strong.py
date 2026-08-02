# TIER: strong
"""The insight: stage SIZE should track uncertainty resolved per dollar, not
engineering convenience. Any module whose signal is actually diagnostic
(accuracy far from a coin flip) is carved out into its OWN thin, cheap stage
-- an early probe -- instead of being batched with a big, uninformative
(but expensive) tranche for "efficiency". Everything else is built as one
large stage (capturing whatever scale economy is available on the part of
the project that genuinely can't be de-risked further).

Given that partition, the abandon/continue call at every checkpoint is not a
vote -- it is computed exactly: at each reachable signal history, compare
the expected value of stopping now (irreversible salvage) against the
expected value of continuing (which requires knowing the TRUE posterior
probability the project is Good given everything observed, using each
module's own accuracy -- not a naive nose count), via backward induction
from the end of the project to the front. This is provably optimal FOR THE
CHOSEN PARTITION; it does not, however, search every possible partition --
that headroom is deliberately left on the table.
"""
import sys

INFO_THRESH = 0.12


def choose_boundaries(M, acc):
    cuts = set()
    for j in range(1, M + 1):
        if abs(acc[j - 1] - 0.5) > INFO_THRESH:
            if j - 1 >= 1:
                cuts.add(j - 1)
            cuts.add(j)
    cuts.add(M)
    cuts.discard(0)
    return sorted(cuts)


def optimal_decision_tables(case, boundaries):
    costs, acc, p, sigma, F, r = (case[k] for k in
                                   ['costs', 'acc', 'p', 'sigma', 'F', 'r'])
    K = len(boundaries)
    g = [0] + boundaries
    tables = {}

    def rec(stage_idx, h, cum_raw, cum_disc, wG_in, wB_in):
        modules = range(g[stage_idx - 1] + 1, g[stage_idx] + 1)
        modules = list(modules)
        stage_cost = F + sum(costs[j - 1] for j in modules)
        disc = r ** (stage_idx - 1)
        new_raw = cum_raw + stage_cost
        new_disc = cum_disc + disc * stage_cost
        nbits = len(modules)
        valG = 0.0
        valB = 0.0
        for combo in range(1 << nbits):
            newh = h
            probG = 1.0
            probB = 1.0
            for bit_i, j in enumerate(modules):
                sig = (combo >> bit_i) & 1
                if sig:
                    newh |= (1 << (j - 1))
                a = acc[j - 1]
                if sig == 1:
                    probG *= a
                    probB *= (1 - a)
                else:
                    probG *= (1 - a)
                    probB *= a
            wG = wG_in * probG
            wB = wB_in * probB
            if stage_idx == K:
                vg = -new_disc + (r ** K) * case['VG']
                vb = -new_disc + (r ** K) * case['VB']
            else:
                cvg, cvb = rec(stage_idx + 1, newh, new_raw, new_disc, wG, wB)
                aval = -new_disc + disc * sigma * new_raw
                cont_w = wG * cvg + wB * cvb
                aband_w = (wG + wB) * aval
                decide = 1 if cont_w > aband_w else 0
                tables.setdefault(stage_idx, {})[newh] = decide
                vg, vb = (cvg, cvb) if decide else (aval, aval)
            valG += probG * vg
            valB += probB * vb
        return valG, valB

    rec(1, 0, 0, 0, case['p'], 1 - case['p'])
    # convert sparse dict tables -> dense arrays indexed 0..2**gk-1
    dense = {}
    for k in range(1, K):
        gk = g[k]
        d = tables.get(k, {})
        dense[k] = [d.get(h, 1) for h in range(1 << gk)]  # default continue if unreached
    return dense


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    M = int(next(it))
    costs = [int(next(it)) for _ in range(M)]
    acc = [float(next(it)) for _ in range(M)]
    p = float(next(it))
    VG = float(next(it))
    VB = float(next(it))
    sigma = float(next(it))
    F = float(next(it))
    r = float(next(it))
    case = dict(M=M, costs=costs, acc=acc, p=p, VG=VG, VB=VB, sigma=sigma, F=F, r=r)

    boundaries = choose_boundaries(M, acc)
    K = len(boundaries)
    tables = optimal_decision_tables(case, boundaries)

    out = [str(K)]
    out.append(" ".join(str(x) for x in boundaries))
    for k in range(1, K):
        out.append(" ".join(str(x) for x in tables[k]))
    print("\n".join(out))


if __name__ == "__main__":
    main()
