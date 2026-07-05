#!/usr/bin/env python3
"""gen.py <testId>  ->  prints ONE routing instance to stdout.

Warehouse-robotics skin of quantum-circuit qubit-routing / SWAP minimization.

Instance schema (stdin the solver reads):
    V E m
    E lines:  u v        conveyor track between docking bays u,v (undirected)
    m lines:  a b        ordered handoff task: robots a and b must be at
                         adjacent bays to transfer a parcel

Bays form a rectangular grid (a "warehouse floor"); every bay holds exactly one
robot, so there are V robots (ids 0..V-1) and V bays (0..V-1). The solver chooses
the initial robot placement, then inserts conveyor SWAPs so that each handoff,
executed IN ORDER, happens between two robots on adjacent bays.

Structure: robots belong to hidden work ZONES; most handoffs are within-zone, but
zone membership is scrambled relative to robot id -- so the naive identity
placement scatters each zone across the floor while a good placement gathers it.
Difficulty ladder testId 1..10: floor grows, tasks get denser. Seeded by testId.
"""
import sys, random


def dims(t):
    rows = 3 + (t - 1) // 2      # t1:3 ... t10:7
    cols = 3 + t // 3            # t1:3 ... t10:6
    return rows, cols


def main():
    t = int(sys.argv[1])
    rows, cols = dims(t)
    V = rows * cols
    rng = random.Random(1000 + 7 * t)

    # grid edges (right + down neighbours) -- the conveyor network
    edges = []
    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            if c + 1 < cols:
                edges.append((u, u + 1))
            if r + 1 < rows:
                edges.append((u, u + cols))

    # hidden work zones (scattered over robot ids -> scattered over the grid)
    K = max(2, V // 8)
    zone_of = [rng.randrange(K) for _ in range(V)]
    members = [[] for _ in range(K)]
    for robot in range(V):
        members[zone_of[robot]].append(robot)
    # guarantee every zone has >= 2 members so within-zone tasks are well defined
    for z in range(K):
        while len(members[z]) < 2:
            r = rng.randrange(V)
            zone_of[r] = z
            members = [[] for _ in range(K)]
            for robot in range(V):
                members[zone_of[robot]].append(robot)
    nonempty = [z for z in range(K) if len(members[z]) >= 2]

    p_in = 0.75
    m = V + 3 * t
    tasks = []
    for _ in range(m):
        if rng.random() < p_in:
            z = rng.choice(nonempty)
            a = rng.choice(members[z])
            b = rng.choice(members[z])
            while b == a:
                b = rng.choice(members[z])
        else:
            a = rng.randrange(V)
            b = rng.randrange(V)
            while b == a:
                b = rng.randrange(V)
        tasks.append((a, b))

    out = ["%d %d %d" % (V, len(edges), m)]
    out += ["%d %d" % (u, v) for (u, v) in edges]
    out += ["%d %d" % (a, b) for (a, b) in tasks]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
