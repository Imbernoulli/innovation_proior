#!/usr/bin/env python3
"""gen.py <testId> -- prints one rigid-origami fold-to-target instance to stdout.
Deterministic: all randomness seeded from testId only.
"""
import sys, math, random
import riglib as R

# difficulty ladder: (K modules, |t_k| upper bound for the hidden planted solution)
# t is the free per-flower kinematic parameter (radians). Larger t => bigger fold =>
# an independently-per-crease ("obvious") assignment departs further from the true
# 1-parameter constraint curve => bigger loop-closure tear for the naive approach.
SCHEDULE = {
    1: (1, 0.15),
    2: (1, 0.35),
    3: (2, 0.25),
    4: (2, 0.70),
    5: (2, 0.90),
    6: (3, 0.30),
    7: (3, 0.70),
    8: (3, 0.95),
    9: (3, 1.05),
    10: (3, 1.10),
}


def gen_module(rng):
    a1 = rng.uniform(0.55, math.pi - 0.55)
    a3 = math.pi - a1
    a2 = rng.uniform(0.55, math.pi - 0.55)
    a4 = math.pi - a2
    a = [a1, a2, a3, a4]
    L = [rng.uniform(1.0, 2.4) for _ in range(4)]
    return {'a': a, 'L': L}


def main():
    tid = int(sys.argv[1])
    K, tmax = SCHEDULE.get(tid, SCHEDULE[10])
    rng = random.Random(20260726 + 97 * tid)

    modules = [gen_module(rng) for _ in range(K)]

    # hidden planted solution: one free parameter t_k per module + one bridge angle
    t_hidden = [rng.uniform(-tmax, tmax) for _ in range(K)]
    b_hidden = [rng.uniform(-0.9, 0.9) for _ in range(K - 1)]

    angles = []
    for k in range(K):
        us, pts = R.flower_geom(modules[k]['a'], modules[k]['L'])
        t1, t3, t4, resid = R.solve_flower_path(us, t_hidden[k], nsteps=90)
        assert resid < 1e-6, "generator: closure solve failed to converge"
        angles += [t1, t_hidden[k], t3, t4]
        if k < K - 1:
            angles.append(b_hidden[k])

    tips, closures = R.simulate_instance(modules, angles)
    assert all(c < 1e-6 for c in closures), "generator: planted solution is not feasible"

    # Perturb the planted (exactly-reachable) tip positions off the constraint manifold
    # so the checker's optimum is not exactly attainable by ANY angle choice (keeps
    # headroom -- see AGENT_BRIEF: strong must not saturate).
    noise_frac = 0.15
    targets = []
    for tip in tips:
        scale = noise_frac * max(1.0, R.norm(tip))
        d = (rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))
        d = R.normalize(d)
        targets.append(R.add(tip, R.scale(d, scale)))

    out = [str(K)]
    for k in range(K):
        a = modules[k]['a']; L = modules[k]['L']
        out.append(" ".join("%.9f" % v for v in a))
        out.append(" ".join("%.9f" % v for v in L))
    for tg in targets:
        out.append(" ".join("%.9f" % v for v in tg))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
