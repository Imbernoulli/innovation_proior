#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE training sample to stdout.

Orbital mechanics -- conserved-quantity discovery.  A planar two-body (Kepler)
system: a test particle orbits a central mass with gravitational parameter mu.
Each test id fixes a DIFFERENT hidden mu and a DIFFERENT set of orbits.  You are
given noisy state samples

    x1 = position x      x2 = position y
    x3 = velocity vx     x4 = velocity vy

drawn from several TRAINING orbits (each orbit = one "trajectory": a group of
samples that lie on the same physical path).  Your task is to output a closed-form
expression C(x1,x2,x3,x4) that is CONSERVED -- constant along each trajectory --
yet still varies from orbit to orbit.

The grading orbits are DIFFERENT, LARGER orbits (an extrapolation region) and are
regenerated only inside the grader; the hidden mu, the orbital elements and the
seeds are NEVER printed here.

Difficulty ladder (testId 1..10): heavier measurement noise on the samples.

STDOUT format:
    line 1:  "<n_traj> <n_per_traj> <test_id>"
    then n_traj*n_per_traj rows, each:  "<traj_id> <x1> <x2> <x3> <x4>"
    Rows sharing a traj_id lie on the same orbit.
"""
import sys, random, math

J_TRAIN = 14          # training orbits
K = 16                # samples per orbit
A_LO, A_HI = 1.0, 2.0
E_LO, E_HI = 0.1, 0.5


def mu_of(t):
    rng = random.Random(90001 + t * 7919)
    return rng.uniform(0.8, 1.5)


def noise_level(t):
    return 0.030 + (t - 1) * 0.005


def kepler_states(a, e, mu, omega, E_list):
    n = math.sqrt(mu / (a ** 3))
    b = math.sqrt(1.0 - e * e)
    co, so = math.cos(omega), math.sin(omega)
    out = []
    for E in E_list:
        cE, sE = math.cos(E), math.sin(E)
        xo = a * (cE - e)
        yo = a * b * sE
        Edot = n / (1.0 - e * cE)
        vxo = -a * sE * Edot
        vyo = a * b * cE * Edot
        x = co * xo - so * yo
        y = so * xo + co * yo
        vx = co * vxo - so * vyo
        vy = so * vxo + co * vyo
        out.append((x, y, vx, vy))
    return out


def gen_split(t, seed, jr, ar, er):
    """Generate a list of orbits (each a list of noisy (x1,x2,x3,x4))."""
    mu = mu_of(t)
    nl = noise_level(t)
    rng = random.Random(seed)
    trajs = []
    for _ in range(jr):
        a = rng.uniform(*ar)
        e = rng.uniform(*er)
        omega = rng.uniform(0.0, 2.0 * math.pi)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        E_list = [phase + 2.0 * math.pi * k / K + rng.uniform(-0.05, 0.05)
                  for k in range(K)]
        clean = kepler_states(a, e, mu, omega, E_list)
        sp = nl * a
        sv = nl * math.sqrt(mu / a)
        noisy = []
        for (x, y, vx, vy) in clean:
            noisy.append((x + rng.gauss(0, sp), y + rng.gauss(0, sp),
                          vx + rng.gauss(0, sv), vy + rng.gauss(0, sv)))
        trajs.append(noisy)
    return trajs


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    trajs = gen_split(t, 500 + t * 104729, J_TRAIN, (A_LO, A_HI), (E_LO, E_HI))
    out = ["%d %d %d" % (J_TRAIN, K, t)]
    for j, tr in enumerate(trajs):
        for (x1, x2, x3, x4) in tr:
            out.append("%d %r %r %r %r" % (j, x1, x2, x3, x4))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
