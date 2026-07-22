import sys, random

# gen.py <testId>  -- prints ONE land-plot auction instance to stdout.
#
# n bidders stand in a fixed row (plot 1 .. plot n) along a coastal access road.
# Adjacent plots (i, i+1) share one access easement, so at most one of every
# adjacent pair may be awarded (a maximum-weight independent-set-on-a-path
# welfare problem -- the interval-scheduling special case where every plot's
# frontage overlaps only its immediate neighbour's).
#
# Difficulty ladder (testId 1..10) grows n. The op-count TRAP is entirely in n:
# an approach that reruns the O(n) welfare DP fresh for every one of the n
# leave-one-out counterfactual economies costs ~4*n^2 ops; one that ALSO forgets
# to reuse the O(n) recurrence (recomputing forward AND backward every time)
# costs ~8*n^2; sharing one forward + one backward DP pass across every
# counterfactual (prefix/suffix stitching) costs ~9*n. As n grows the gap
# between "recompute per economy" and "share the prefix/suffix tables" widens
# sharply -- that gap is what the checker's baseline exposes.
#
# ANTI-MEMORIZATION: each test case bundles T independent bid vectors that
# share the same n (same wiring/topology). The submitted circuit is a single,
# fixed straight-line program that gets RE-EVALUATED once per trial with that
# trial's bid values on input wires 0..n-1; it must reproduce every trial's
# exact VCG payments. A circuit that "hardcodes" one instance's numeric
# answer (e.g. via bare CONST literals instead of real input-dependence)
# cannot also match the other, independently-random trials.

N_BY_TEST = {1: 4, 2: 6, 3: 8, 4: 10, 5: 12, 6: 14, 7: 15, 8: 16, 9: 17, 10: 18}
TRIALS = 3

def one_trial(rng, n, regime):
    v = []
    for k in range(n):
        if regime == 0:
            v.append(rng.randint(1, 1000))
        elif regime == 1:
            # occasional high spikes force real leave-one-out swings
            v.append(rng.randint(400, 1000) if rng.random() < 0.3 else rng.randint(1, 150))
        else:
            v.append(rng.randint(1, 1000) + (50 * (k % 5)))
    return v

def main():
    tid = int(sys.argv[1])
    n = N_BY_TEST[tid]
    rng = random.Random(4271183 + 97 * tid)

    # Mix a few value regimes across the ladder so the optimal winner set is
    # never a trivial "take everything" or "alternate" pattern the checker's
    # own DP has to discover honestly from the data (not from the statement).
    regime = tid % 3

    lines = ["%d %d" % (n, TRIALS)]
    for _ in range(TRIALS):
        v = one_trial(rng, n, regime)
        lines.append(" ".join(str(x) for x in v))

    sys.stdout.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
