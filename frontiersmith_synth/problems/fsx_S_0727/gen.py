import sys, random

# gen.py <testId> -- prints one "monotone register ratchet" addition-chain
# instance to stdout.
#
# Layout:
#   N K B0
#   N distinct positive integers (the targets)
#
# The targets are constructed as: a majority "shared" block (multiples of a
# common hidden base C, order shuffled but printed FIRST) plus a minority
# "independent" block (random integers not divisible by C, printed LAST).
# This ordering is a deliberate trap: a naive solver that processes targets
# in the given input order discovers & commits to the shared base within the
# first couple of targets -- while the register-creation ratchet still grants
# it only a tiny reuse budget -- and then runs out of budget partway through
# the (long) shared block. A solver that looks at the WHOLE target set first
# can defer committing to the shared base until later in the program, when
# the ratchet has grown generous enough to support the full reuse demand.
#
# Difficulty grows with testId via N (more targets => more reuse demand on
# the shared base) and via C's magnitude (longer chains all around).

def spec(tid):
    rng = random.Random(727000 + 97 * tid)
    N = 10 + 3 * tid                      # 13 .. 40
    n_shared = max(4, round(N * 0.8))
    n_indep = N - n_shared
    K = 6
    B0 = 2
    C = rng.randint(150 + 40 * tid, 400 + 260 * tid)

    mult_pool = list(range(3, 41))        # 38 candidate multipliers
    rng.shuffle(mult_pool)
    order = mult_pool[:n_shared]
    rng.shuffle(order)
    shared_targets = [C * m for m in order]

    indep = []
    seen = set(shared_targets)
    lo, hi = 3 * C, 40 * C
    tries = 0
    while len(indep) < n_indep and tries < 200000:
        tries += 1
        v = rng.randint(max(2, lo), max(3, hi))
        if v in seen or v % C == 0:
            continue
        seen.add(v)
        indep.append(v)

    targets = shared_targets + indep
    return N, K, B0, targets


def main():
    tid = int(sys.argv[1])
    N, K, B0, targets = spec(tid)
    out = ["%d %d %d" % (N, K, B0), " ".join(str(x) for x in targets)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
