#!/usr/bin/env python3
"""gen.py <testId> -> prints one 'tempering chart' instance to stdout.

Format:
    L Tmax C0 n_max B
    d_1 ... d_L        (initial defect severity per grain)
    theta_1 ... theta_L  (nucleation threshold per grain, 1 <= theta_i <= Tmax-1)

Deterministic: fully seeded from testId (random.Random(seed)). Ladder 1..10 grows
in size and structural difficulty; testIds {4, 6, 9} are engineered "trap" cases
where a small fragile minority (low theta) is buried inside a robust majority, so
any single global temperature calibrated only to the AVERAGE threshold overshoots
the minority and triggers unbounded per-step nucleation there.
"""
import sys
import random


def build(test_id: int):
    Tmax = 12
    sizes = [25, 50, 150, 300, 500, 700, 900, 1200, 1600, 2000]
    L = sizes[test_id - 1]
    rng = random.Random(10_000 + 97 * test_id)

    trap_ids = {4, 6, 9}
    d = [0] * L
    theta = [0] * L
    n_max = 40
    # target per-cell heal capacity scale: grows with difficulty so severities
    # stay genuinely budget-limited (no schedule can zero out the hardest cells,
    # which keeps the score off the 1.0 ceiling even for the reference 'strong').
    HC = 35 + 16 * test_id

    if test_id in trap_ids:
        # skewed two-cluster billet: small fragile minority + robust majority.
        # Mobility above the offset-2 activation floor is what actually heals;
        # nucleation only fires once T exceeds a grain's own theta.
        frac_fragile = rng.uniform(0.04, 0.12)
        n_frag = max(1, int(L * frac_fragile))
        theta_frag = rng.randint(1, 3)
        theta_rob = rng.randint(9, Tmax - 1)
        idx = list(range(L))
        rng.shuffle(idx)
        frag_set = set(idx[:n_frag])
        C0 = rng.randint(2, 4)
        for i in range(L):
            if i in frag_set:
                theta[i] = theta_frag
                d[i] = rng.randint(3, 10)
            else:
                theta[i] = theta_rob
                d[i] = rng.randint(int(0.65 * HC), int(1.8 * HC))
        mob = theta_rob - 2 if theta_rob > 2 else 1
        n_req = max(3, -(-HC // mob))  # ceil
        B = n_req * (C0 + theta_rob)
    else:
        K = rng.randint(3, 7)
        block_theta = [rng.randint(2, Tmax - 1) for _ in range(K)]
        # random block boundaries partitioning [0, L)
        if K > 1 and L > K:
            cuts = sorted(rng.sample(range(1, L), K - 1))
        else:
            cuts = []
        bounds = [0] + cuts + [L]
        C0 = rng.randint(2, 6)
        for k in range(K):
            lo, hi = bounds[k], bounds[k + 1]
            th = block_theta[k]
            for i in range(lo, hi):
                theta[i] = th
                d[i] = rng.randint(int(0.65 * HC), int(1.8 * HC))
        min_theta = min(block_theta)
        mob = min_theta - 2 if min_theta > 2 else 1
        n_req = max(3, -(-HC // mob))  # ceil
        B = n_req * (C0 + min_theta)

    return L, Tmax, C0, n_max, B, d, theta


def main():
    if len(sys.argv) != 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    test_id = int(sys.argv[1])
    L, Tmax, C0, n_max, B, d, theta = build(test_id)
    out = []
    out.append(f"{L} {Tmax} {C0} {n_max} {B}")
    out.append(" ".join(str(x) for x in d))
    out.append(" ".join(str(x) for x in theta))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
