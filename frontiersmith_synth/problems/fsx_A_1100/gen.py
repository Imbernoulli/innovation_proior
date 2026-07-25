#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE training instance to stdout.

The Collision-Rig Consortium runs K=5 two-body collision rigs.  Rig r has fixed
masses (m1,m2), damping gamma, and sampling interval dt; every recorded row is
one collision: pre-impact velocities (v1,v2) and measured post-impact
velocities (v1',v2').

A hidden conservation-style law governs every rig (stylised physics, shared
across rigs, different per testId):

    m1^alpha * v1  +  m2^alpha * v2      is CONSERVED by the collision
    v2' - v1'  =  -e * (v2 - v1)   with  e = e0 * exp(-beta*gamma - eta*|v2-v1|)

The shared exponent alpha, the restitution scale e0, the damping coefficient
beta, and the impact-speed decay eta are NEVER printed.  Post-impact
velocities carry sensor noise whose scale depends on the rig's sampling
interval (slower sampling => driftier readout).

The HELD-OUT grading rig (regenerated only inside the checker) has masses
OUTSIDE every training rig's range -- further outside for harder test ids --
and, for hard ids, damping outside the training band too.

STDOUT prints ONLY: header "<n_rows> <test_id> 5", then 5 rig lines
"RIG <m1> <m2> <gamma> <dt>", then n_rows data lines
"ROW <rig> <v1> <v2> <v1p> <v2p>".  No seeds, no hidden constants.
"""
import sys, random, math

K = 5


def hidden(t):
    """Hidden law parameters for this test id (live in gen AND checker, never printed)."""
    rng = random.Random(917331 + t * 7919)
    if t <= 3:
        alpha = rng.uniform(0.94, 1.06)
    elif t <= 6:
        alpha = rng.choice([rng.uniform(0.84, 0.93), rng.uniform(1.07, 1.16)])
    else:
        alpha = rng.choice([rng.uniform(0.70, 0.82), rng.uniform(1.18, 1.32)])
    e0 = rng.uniform(0.55, 0.80)
    beta = rng.uniform(0.40, 1.00)
    eta = rng.uniform(0.55, 0.95)
    return alpha, e0, beta, eta


def train_rigs(t):
    rng = random.Random(40009 + t * 104729)
    rigs = []
    for _ in range(K):
        m1 = rng.uniform(1.0, 8.0)
        m2 = rng.uniform(1.0, 8.0)
        g = rng.uniform(0.05, 0.40)
        dt = rng.choice([0.5, 1.0, 2.0])
        rigs.append((m1, m2, g, dt))
    return rigs


def sigma0(t):
    return 0.10 + 0.005 * t


def vscale(t):
    return 1.0 + 0.02 * t


def collide(m1, m2, g, v1, v2, alpha, e0, beta, eta):
    a1 = m1 ** alpha
    a2 = m2 ** alpha
    w = v2 - v1
    e = e0 * math.exp(-beta * g - eta * abs(w))
    P = a1 * v1 + a2 * v2
    v1p = (P + a2 * e * w) / (a1 + a2)
    v2p = (P - a1 * e * w) / (a1 + a2)
    return v1p, v2p


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    alpha, e0, beta, eta = hidden(t)
    rigs = train_rigs(t)
    s = vscale(t)
    rngv = random.Random(20261 + t * 15485863)
    rngn = random.Random(555 + t * 13)

    rows = []
    for ri, (m1, m2, g, dt) in enumerate(rigs):
        n_r = int(150.0 / dt)
        sig = sigma0(t) * dt
        for _ in range(n_r):
            v1 = s * rngv.uniform(-1.1, 0.4)
            v2 = s * rngv.uniform(-0.4, 1.1)
            y1, y2 = collide(m1, m2, g, v1, v2, alpha, e0, beta, eta)
            y1 += rngn.gauss(0.0, sig)
            y2 += rngn.gauss(0.0, sig)
            rows.append((ri, v1, v2, y1, y2))

    out = ["%d %d %d" % (len(rows), t, K)]
    for (m1, m2, g, dt) in rigs:
        out.append("RIG %.6f %.6f %.6f %.6f" % (m1, m2, g, dt))
    for (ri, v1, v2, y1, y2) in rows:
        out.append("ROW %d %.6f %.6f %.6f %.6f" % (ri, v1, v2, y1, y2))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
