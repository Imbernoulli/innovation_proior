import sys, random

# gen.py <testId>  --  "one furnace schedule to forge many alloys"
#
# Prints ONE batch-exponentiation instance to stdout:
#     k
#     e_1
#     ...
#     e_k
# where each e_i is a large positive integer.  The participant must emit a
# straight-line multiplication program over a formal symbol g that produces
# g^{e_i} for EVERY i, using as few multiplications as possible.
#
# PLANTED STRUCTURE (the hidden furnace schedule):
#   A small HIDDEN modulus m (odd, non-power-of-two, discoverable by brute force
#   over small m) is chosen.  Every exponent is written  e_i = q_i * m + r_i
#   with
#     - r_i drawn from a SMALL set R of residues mod m  (few distinct residues),
#     - q_i drawn with LOW popcount (few set bits) -> cheap to raise g^m to.
#   Multiplying a low-popcount quotient by an ODD m scatters the bits, so e_i
#   itself has HIGH popcount.  A solver that does not see m pays ~popcount(e_i)
#   per exponent (a full binary/square-and-multiply batch = the greedy trap);
#   a solver that recovers m collapses the whole batch onto ONE shared chain
#   for the g^m-powers plus tiny per-exponent residue corrections.
#
# All randomness is seeded from testId only -> fully deterministic.

#            m ,  t(#residues), s(quotient popcount), qb(quotient bits), k(batch)
SPECS = {
    1:  (23,  3, 3, 18,  8),
    2:  (29,  3, 3, 20,  9),
    3:  (37,  4, 3, 22, 10),
    4:  (41,  4, 4, 24, 10),
    5:  (53,  4, 4, 26, 11),
    6:  (67,  5, 4, 28, 12),
    7:  (83,  5, 4, 30, 12),
    8:  (101, 5, 5, 32, 13),
    9:  (131, 6, 5, 34, 14),
    10: (173, 6, 5, 36, 14),
}


def low_popcount(rng, qb, s):
    # a positive integer with exactly s set bits, top bit at position qb-1
    positions = set([qb - 1])
    while len(positions) < s:
        positions.add(rng.randrange(0, qb - 1))
    q = 0
    for p in positions:
        q |= (1 << p)
    return q


def main():
    tid = int(sys.argv[1])
    m, t, s, qb, k = SPECS[tid]
    rng = random.Random(918273 + 1000 * tid)

    # small residue set (few distinct residues mod m)
    residues = rng.sample(range(0, m), min(t, m))

    quots = set()
    while len(quots) < k:
        quots.add(low_popcount(rng, qb, s))
    quots = list(quots)
    rng.shuffle(quots)

    exps = set()
    order = []
    for q in quots:
        r = rng.choice(residues)
        e = q * m + r
        if e in exps:
            # nudge with another residue to keep exponents distinct
            for r2 in residues:
                e2 = q * m + r2
                if e2 not in exps:
                    e = e2
                    break
        exps.add(e)
        order.append(e)

    out = [str(len(order))]
    out += [str(e) for e in order]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
