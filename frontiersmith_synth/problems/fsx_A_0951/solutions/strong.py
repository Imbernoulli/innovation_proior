# TIER: strong
# Insight: the mechanism is public (it's spelled out in the statement), so instead
# of guessing a single "best" species we can build the metabolite hand-off GRAPH
# from the instance's own consumes/produces fields, find each waste-producer's
# best matching waste-consumer (by vmax * yield_biomass, not just raw vmax), and
# then use our OWN faithful re-implementation of the reaction-diffusion dynamics
# to compare a short list of candidate constructions: every monoculture, AND an
# interleaved producer/partner layout for every producer that has a usable
# partner. We keep whichever construction our internal simulation scores highest.
# This is a decomposition (graph of who-feeds-whom) + local search over spatial
# arrangements, not just "pick the fastest species everywhere".
import sys, json


def simulate(L, T, boundary_conc, diffusion, species, assign):
    K = 2
    conc = [[0.0] * L for _ in range(K)]
    conc[0][0] = boundary_conc
    conc[0][L - 1] = boundary_conc
    biomass = [0.0] * L
    d = diffusion
    for _ in range(T):
        newconc = [row[:] for row in conc]
        for k in range(K):
            dk = d[k]
            ck = conc[k]
            nk = newconc[k]
            for i in range(L):
                left = ck[i - 1] if i > 0 else ck[i]
                right = ck[i + 1] if i < L - 1 else ck[i]
                nk[i] = ck[i] + dk * (left + right - 2 * ck[i])
        conc = newconc
        conc[0][0] = boundary_conc
        conc[0][L - 1] = boundary_conc
        for i in range(L):
            s = assign[i]
            if s is None or s < 0:
                continue
            sp = species[s]
            c = sp["consumes"]; p = sp["produces"]
            c_up = conc[c][i]
            denom = sp["Km"] + c_up
            monod = c_up / denom if denom > 0 else 0.0
            if p != -1:
                inhib = sp["Ki"] / (sp["Ki"] + conc[p][i])
            else:
                inhib = 1.0
            rate = sp["vmax"] * monod * inhib
            consumed = rate if rate < c_up else c_up
            if consumed < 0:
                consumed = 0.0
            conc[c][i] -= consumed
            biomass[i] += consumed * sp["yield_biomass"]
            if p != -1:
                conc[p][i] += consumed * sp["yield_byproduct"]
    return sum(biomass)


def main():
    inst = json.load(sys.stdin)
    L = inst["L"]; T = inst["T"]; bc = inst["boundary_conc"]
    diffusion = inst["diffusion"]; species = inst["species"]
    S = len(species)

    candidates = []
    # every monoculture
    for s in range(S):
        candidates.append([s] * L)
    # every producer, interleaved (period 2) with its best matching consumer
    for prod_idx, sp in enumerate(species):
        p = sp["produces"]
        if p == -1:
            continue
        partners = [j for j in range(S) if species[j]["consumes"] == p]
        if not partners:
            continue
        best_partner = max(partners, key=lambda j: species[j]["vmax"] * species[j]["yield_biomass"])
        pat = [prod_idx, best_partner]
        candidates.append([pat[i % 2] for i in range(L)])
        # also try a 2:1 producer-heavy interleave (more producer cells, since
        # the producer is usually the scarcer / more valuable role)
        pat2 = [prod_idx, best_partner, prod_idx]
        candidates.append([pat2[i % 3] for i in range(L)])

    best_assign, best_val = None, -1.0
    for a in candidates:
        v = simulate(L, T, bc, diffusion, species, a)
        if v > best_val:
            best_val = v
            best_assign = a

    print(json.dumps({"assign": best_assign}))


main()
