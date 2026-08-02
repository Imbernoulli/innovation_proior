#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE dynamic-hedge-rebalance instance to stdout.
Deterministic: all randomness seeded from testId only.
"""
import sys, math, random

K_STRIKE = 100.0
BETA = 9.0
GAMMA_SCALE = 40.0
COST_FIXED = 0.08
COST_PROP = 0.001


def make_instance(test_id: int):
    rng = random.Random(1000 + test_id * 7919)
    N = 24 + 12 * (test_id - 1)              # 24 .. 132: warm-ups -> adversarial
    n_jumps = 1 + test_id // 2                # 1 .. 6 shock events
    noise_sigma = 0.010 + 0.001 * test_id
    jump_shock_lo, jump_shock_hi = 0.09, 0.16
    revert_frac_lo, revert_frac_hi = 0.55, 0.85

    if N > 6:
        candidates = list(range(3, N - 2))
        rng.shuffle(candidates)
        chosen = sorted(candidates[:n_jumps])
    else:
        chosen = []
    jump_at = set(chosen)

    S = [K_STRIKE]
    t = 1
    while t <= N:
        prev = S[-1]
        if t in jump_at:
            shock = rng.uniform(jump_shock_lo, jump_shock_hi) * rng.choice([-1, 1])
            S.append(prev * math.exp(shock))
            t += 1
            if t <= N:
                revert_frac = rng.uniform(revert_frac_lo, revert_frac_hi)
                target = prev * math.exp(shock * (1 - revert_frac))
                target *= math.exp(rng.gauss(0, noise_sigma * 0.5))
                S.append(target)
                t += 1
        else:
            S.append(prev * math.exp(rng.gauss(0, noise_sigma)))
            t += 1
    S = S[:N + 1]

    D, G = [], []
    for s in S:
        x = math.log(s / K_STRIKE)
        d = math.tanh(BETA * x)
        g = GAMMA_SCALE * (BETA / s) * (1.0 - d * d)
        D.append(d)
        G.append(g)

    return N, S, D, G, COST_PROP, COST_FIXED


def main():
    test_id = int(sys.argv[1])
    N, S, D, G, cost_prop, cost_fixed = make_instance(test_id)
    out = []
    out.append(str(N))
    out.append(" ".join("%.10g" % v for v in S))
    out.append(" ".join("%.10g" % v for v in D))
    out.append(" ".join("%.10g" % v for v in G))
    out.append("%.10g %.10g" % (cost_prop, cost_fixed))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
