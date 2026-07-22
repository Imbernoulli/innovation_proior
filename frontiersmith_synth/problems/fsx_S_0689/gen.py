import sys, math, random


def gen_instance(i):
    rng = random.Random(9001 + 17 * i)
    if i <= 2: N = 6
    elif i <= 4: N = 8
    elif i <= 6: N = 10
    elif i <= 8: N = 12
    else: N = 14

    c0 = 0.6 + 0.8 * rng.random()          # [0.6, 1.4)
    T = math.pi / (2.0 * c0)

    # >=3 of the 10 cases are engineered traps (deliberately mismatched frozen edges)
    trap = i in (3, 5, 7, 9, 10)

    J_LO, J_HI = 0.05, 11.0

    # defects: interior sites 2..N-1 (1-indexed), never the ports (site 1 / site N)
    num_def = 1 + (i % 2)
    interior = list(range(2, N))
    rng.shuffle(interior)
    defect_sites = sorted(interior[:num_def])
    defects = {}
    for s in defect_sites:
        mag = c0 * (1.0 + 1.5 * rng.random())
        if trap:
            mag *= 1.3
        sign = rng.choice([-1, 1])
        defects[s] = sign * mag

    # frozen couplings: some edges are fixed and cannot be chosen by the solver
    num_frozen = 1 + (i % 2)
    edges = list(range(1, N))
    rng.shuffle(edges)
    frozen_edges = sorted(edges[:num_frozen])
    frozen = {}
    for e in frozen_edges:
        ideal = c0 * math.sqrt(e * (N - e))
        if trap:
            # deliberately mismatched: pinned to the OPPOSITE extreme of the natural shape
            if ideal > 0.6 * c0 * N / 2.0:
                frozen[e] = J_LO + 0.02 * rng.random()
            else:
                frozen[e] = J_HI - 0.02 * rng.random()
        else:
            v = ideal * (0.8 + 0.4 * rng.random())
            frozen[e] = min(J_HI, max(J_LO, v))

    return dict(N=N, T=T, J_LO=J_LO, J_HI=J_HI, defects=defects, frozen=frozen)


def main():
    i = int(sys.argv[1])
    inst = gen_instance(i)
    N = inst['N']
    out = [str(N), "%.10f" % inst['T'], "%.10f %.10f" % (inst['J_LO'], inst['J_HI'])]
    defs = sorted(inst['defects'].items())
    out.append(str(len(defs)))
    for s, v in defs:
        out.append("%d %.10f" % (s, v))
    froz = sorted(inst['frozen'].items())
    out.append(str(len(froz)))
    for e, v in froz:
        out.append("%d %.10f" % (e, v))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
