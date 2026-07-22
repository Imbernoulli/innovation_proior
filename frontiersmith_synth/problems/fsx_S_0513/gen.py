import sys, random

# fsx_S_0513 -- braess-pruning-equilibrium
# Emits a directed s-t network with superlinear congestion latencies
#   ell_e(x) = a_e + b_e * x^k .
# The graph is a chain of "diamond" layers between hubs H0=s .. HL=t.
# Some layers are Braess TRAPS (a tempting shortcut whose presence RAISES the
# selfish-routing equilibrium cost) and some are genuine CAPACITY layers (an
# extra edge whose presence LOWERS it). They are structurally indistinguishable;
# only the coefficients (and the resulting equilibrium) reveal which is which.

def main():
    testId = int(sys.argv[1])
    rng = random.Random(90513 * 1000 + testId)

    L = 2 + (testId - 1) // 2                      # 2..6 layers
    k = 2 if testId <= 3 else (3 if testId <= 6 else 4)
    D = 6.0 + testId                               # 7..16 demand

    edges = []          # (u, v, a, b)
    nid = L + 1         # hubs 0..L already reserved (s=0, t=L)

    def rc(lo, hi):     # rounded coefficient, 4 decimals -> exact/short in text
        return round(rng.uniform(lo, hi), 4)

    eps = 0.02
    for i in range(L):
        Hi, Hj = i, i + 1
        A = nid; nid += 1
        B = nid; nid += 1

        r = rng.random()
        if i <= 1:
            typ = 'trap'
        elif r < 0.65:
            typ = 'trap'
        elif r < 0.85:
            typ = 'cap'
        else:
            typ = 'neutral'

        bc = rc(0.8, 1.2)
        ac = round(rng.uniform(0.9, 1.1) * bc * (D ** k) * 0.5, 4)
        # base diamond: two parallel (congestible -> constant) / (constant -> congestible) routes
        edges.append((Hi, A, 0.02, bc))            # congestible
        edges.append((A, Hj, ac, eps))             # near-constant
        edges.append((Hi, B, ac, eps))             # near-constant
        edges.append((B, Hj, 0.02, bc))            # congestible
        if typ == 'trap':
            edges.append((A, B, 0.02, eps))        # tempting shortcut (HARMFUL Braess edge)
        elif typ == 'cap':
            bd = rc(0.4, 0.7)
            ad = round(rng.uniform(0.3, 0.6) * bc * (D ** k), 4)
            edges.append((Hi, Hj, ad, bd))         # extra direct capacity (BENEFICIAL edge)
        # neutral: no extra edge

    N = nid
    s, t = 0, L

    out = ["%d %d %d %d %d %s" % (N, len(edges), s, t, k, repr(D))]
    for (u, v, a, b) in edges:
        out.append("%d %d %s %s" % (u, v, repr(a), repr(b)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
