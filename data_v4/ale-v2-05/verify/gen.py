#!/usr/bin/env python3
"""Instance generator for ale-v2-05 "Facility Location with Opening Cost" (UFLP).

Usage:  python3 gen.py SEED  > instance.txt

Instance format (stdout):
    line 1:  F C                                 (#facilities, #clients)
    next F lines:  fx fy fcost                    (facility x,y and opening cost)
    next C lines:  cx cy                          (client x,y)

All coordinates are integers in [0, COORD_MAX]; opening costs are integers.

Semantics: the uncapacitated facility location problem. We must choose a
NON-EMPTY subset S of facilities to open. Each client is served by its single
nearest OPEN facility at Euclidean service cost. The objective to MINIMIZE is

    total = sum_{i in S} opening_cost[i]
          + sum_{c}      min_{i in S} euclid(client c, facility i).

Facilities and clients are drawn from a shared mixture of 2D-Gaussian
"neighbourhood" clusters (the realistic regime where some regions should host a
facility and some should not), plus a little uniform noise. Opening costs are
scaled to sit in the same order of magnitude as a typical cluster's aggregate
service cost, so the open-cost vs service-cost trade-off is genuinely live:
opening too few facilities wastes service distance, opening too many wastes
opening cost. Everything is a deterministic function of SEED.
"""
import sys
import random

COORD_MAX = 1_000_000


def gen(seed: int):
    rng = random.Random(seed * 2654435761 + 911)

    # Sizes: moderate so an O(C*F)-naive drop move is the lever to beat.
    F = rng.randint(60, 120)
    C = rng.randint(400, 700)

    # Latent clusters shared by facilities and clients.
    n_clusters = rng.randint(6, 14)
    centers = []
    for _ in range(n_clusters):
        cx = rng.uniform(0.05, 0.95) * COORD_MAX
        cy = rng.uniform(0.05, 0.95) * COORD_MAX
        sigma = rng.uniform(0.02, 0.09) * COORD_MAX
        weight = rng.uniform(0.5, 2.0)
        centers.append((cx, cy, sigma, weight))
    wsum = sum(c[3] for c in centers)

    def sample_point():
        r = rng.uniform(0, wsum)
        acc = 0.0
        ci = 0
        for j, c in enumerate(centers):
            acc += c[3]
            if r <= acc:
                ci = j
                break
        cx, cy, sigma, _ = centers[ci]
        if rng.random() < 0.05:
            x = rng.uniform(0, COORD_MAX)
            y = rng.uniform(0, COORD_MAX)
        else:
            x = rng.gauss(cx, sigma)
            y = rng.gauss(cy, sigma)
        xi = min(COORD_MAX, max(0, int(round(x))))
        yi = min(COORD_MAX, max(0, int(round(y))))
        return xi, yi

    facilities = []
    # Opening costs: a base level plus per-facility variation. We size the base
    # so that the optimum opens only a fraction of the facilities (otherwise the
    # trivial open-all / open-one baselines would already be near-optimal).
    # Typical inter-point distance is O(0.3*COORD_MAX); a cluster aggregates a
    # few hundred clients, so per-facility opening cost on the order of a few
    # thousand * sigma keeps both terms comparable.
    base_cost = rng.uniform(0.10, 0.35) * COORD_MAX  # ~1e5..3.5e5
    for _ in range(F):
        fx, fy = sample_point()
        mult = rng.uniform(0.5, 1.8)
        fcost = int(round(base_cost * mult))
        facilities.append((fx, fy, fcost))

    clients = []
    for _ in range(C):
        cx, cy = sample_point()
        clients.append((cx, cy))

    return F, C, facilities, clients


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py SEED\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    F, C, facilities, clients = gen(seed)
    out = [f"{F} {C}"]
    for (fx, fy, fcost) in facilities:
        out.append(f"{fx} {fy} {fcost}")
    for (cx, cy) in clients:
        out.append(f"{cx} {cy}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
