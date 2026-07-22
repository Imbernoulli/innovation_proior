import sys, random

# Difficulty ladder / trap plan (testId 1..10):
#   D (canopy depth) grows 3 -> 6.
#   high_cost cases {4,7,9}: biomass costs pushed up (trap: a fixed large-taper/large-r
#     recipe blows its structural budget).
#   multidir cases {3,6,10}: a wide 3-ray sun schedule (-50,0,50 deg) that a single fixed
#     spacing angle cannot serve well from every direction at once.
#   remaining cases: 1-2 rays at modest angles (still deterministic, still nontrivial).

def gen_instance(i):
    rng = random.Random(42131 + 7 * i)
    Dseq = [3, 3, 4, 4, 5, 5, 5, 6, 4, 6]
    D = Dseq[i - 1]
    L0 = 1.0
    A0 = 1.0
    high_cost = i in (4, 7, 9)
    multidir = i in (3, 6, 10)

    cost_len = rng.uniform(0.08, 0.115) if high_cost else rng.uniform(0.03, 0.05)
    cost_leaf = rng.uniform(0.035, 0.06) if high_cost else rng.uniform(0.015, 0.03)

    if multidir:
        angs = [-50.0, 0.0, 50.0]
        ws = [0.3, 0.4, 0.3]
    elif high_cost:
        angs = [rng.uniform(-28, -8), rng.uniform(8, 28)]
        ws = [0.5, 0.5]
    else:
        K = rng.choice([1, 2])
        if K == 1:
            angs = [rng.uniform(-15, 15)]
            ws = [1.0]
        else:
            angs = [rng.uniform(-25, -5), rng.uniform(5, 25)]
            ws = [0.5, 0.5]
    return D, L0, A0, cost_len, cost_leaf, list(zip(angs, ws))


def main():
    i = int(sys.argv[1])
    D, L0, A0, cost_len, cost_leaf, suns = gen_instance(i)
    out = [str(D), "%.6f %.6f" % (L0, A0), "%.6f %.6f" % (cost_len, cost_leaf), str(len(suns))]
    for ang, w in suns:
        out.append("%.6f %.6f" % (ang, w))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
