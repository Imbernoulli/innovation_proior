#!/usr/bin/env python3
"""Deterministic local scorer for the Capacitated Multi-Vehicle Routing problem.

Usage:
    python3 score.py <instance_file> <solution_file>

Reads the instance and the candidate solution, validates feasibility, and prints
a single float: the SCORE. Higher is better.

Feasibility rules (any violation -> score 0):
  * the solution must contain exactly K routes (K lines);
  * each customer id 1..n must appear exactly once across all routes
    (served exactly once, no missing, no duplicate);
  * every id printed must be a valid customer id in [1, n];
  * each route's total demand must be <= Q (capacity).
  * empty routes are allowed (a vehicle may stay at the depot).

Objective:
  raw total distance D = sum over routes of the closed-tour Euclidean length
  depot -> c1 -> ... -> ck -> depot. Lower D is better.

Score (continuous, higher = better, 0 = infeasible):
  We normalise against the Clarke-Wright savings baseline distance D_cw computed
  by this scorer on the same instance:
        score = D_cw / D            (capped at 0 if infeasible)
  So score = 1 means "ties the Clarke-Wright savings baseline"; score > 1 means
  "shorter total distance than Clarke-Wright". This makes the metric continuous
  and comparable across seeds and instance sizes.

The scorer also accepts an env var ALE17_RAW=1 to instead print the raw distance
D (used by self-verification to compare absolute distances directly).
"""
import sys
import os
import math


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    n = int(next(it)); K = int(next(it)); Q = int(next(it))
    dx = float(next(it)); dy = float(next(it))
    depot = (dx, dy)
    xs = [0.0] * (n + 1)
    ys = [0.0] * (n + 1)
    dem = [0] * (n + 1)
    xs[0], ys[0] = depot
    for i in range(1, n + 1):
        xs[i] = float(next(it))
        ys[i] = float(next(it))
        dem[i] = int(next(it))
    return n, K, Q, xs, ys, dem


def dist(xs, ys, i, j):
    return math.hypot(xs[i] - xs[j], ys[i] - ys[j])


def route_length(route, xs, ys):
    if not route:
        return 0.0
    total = dist(xs, ys, 0, route[0])
    for a, b in zip(route, route[1:]):
        total += dist(xs, ys, a, b)
    total += dist(xs, ys, route[-1], 0)
    return total


def parse_solution(path, K):
    """Return list of routes (each a list of ints), or None if it cannot parse
    into exactly K routes. Blank lines count as empty routes."""
    with open(path) as f:
        lines = f.read().splitlines()
    # Trim trailing fully-blank lines beyond K so a trailing newline doesn't add
    # a spurious route; but keep intentional empty routes within the first K.
    routes = []
    for ln in lines:
        toks = ln.split()
        try:
            routes.append([int(t) for t in toks])
        except ValueError:
            return None
    # Drop trailing empty lines so exactly-K can be matched when there is a
    # trailing newline. Empty routes that are followed by a non-empty route are
    # preserved.
    while len(routes) > K and routes and not routes[-1]:
        routes.pop()
    if len(routes) < K:
        routes += [[] for _ in range(K - len(routes))]
    if len(routes) != K:
        return None
    return routes


def clarke_wright_distance(n, K, Q, xs, ys, dem):
    """Compute the Clarke-Wright savings baseline total distance.

    Parallel savings: start with one route per customer (depot-i-depot). Sort
    customer pairs by saving s(i,j)=d(0,i)+d(0,j)-d(i,j) descending; merge the
    routes of i and j when i is an endpoint of its route, j an endpoint of its,
    they are in different routes, and the merged demand <= Q. If the resulting
    number of routes exceeds K, that is acceptable for a *baseline distance*
    reference (the baseline is only a normaliser, not a competitor that must
    respect K). Returns the total distance of the constructed routes.
    """
    if n == 0:
        return 1.0
    # endpoints tracking
    route_of = list(range(n + 1))     # route id for each customer (1..n)
    routes = {i: [i] for i in range(1, n + 1)}
    load = {i: dem[i] for i in range(1, n + 1)}

    pairs = []
    for i in range(1, n + 1):
        d0i = dist(xs, ys, 0, i)
        for j in range(i + 1, n + 1):
            s = d0i + dist(xs, ys, 0, j) - dist(xs, ys, i, j)
            if s > 0:
                pairs.append((s, i, j))
    pairs.sort(reverse=True)

    def endpoints(r):
        return routes[r][0], routes[r][-1]

    for s, i, j in pairs:
        ri, rj = route_of[i], route_of[j]
        if ri == rj:
            continue
        if load[ri] + load[rj] > Q:
            continue
        ai, bi = endpoints(ri)
        aj, bj = endpoints(rj)
        if i not in (ai, bi) or j not in (aj, bj):
            continue
        # orient so that i is the tail of ri and j is the head of rj
        seq_i = routes[ri][:]
        seq_j = routes[rj][:]
        if seq_i[0] == i:
            seq_i.reverse()
        if seq_j[-1] == j:
            seq_j.reverse()
        merged = seq_i + seq_j
        # commit merge into ri
        routes[ri] = merged
        load[ri] += load[rj]
        for c in seq_j:
            route_of[c] = ri
        del routes[rj]
        del load[rj]

    total = 0.0
    for r in routes.values():
        total += route_length(r, xs, ys)
    return total


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py <instance> <solution>\n")
        sys.exit(1)
    inst, sol = sys.argv[1], sys.argv[2]
    n, K, Q, xs, ys, dem = read_instance(inst)

    routes = parse_solution(sol, K)
    if routes is None:
        print(0.0)
        return

    # Feasibility: ids valid, each served exactly once, capacity respected.
    seen = [0] * (n + 1)
    for r in routes:
        load = 0
        for c in r:
            if c < 1 or c > n:
                print(0.0)
                return
            seen[c] += 1
            load += dem[c]
        if load > Q:
            print(0.0)
            return
    for c in range(1, n + 1):
        if seen[c] != 1:
            print(0.0)
            return

    D = sum(route_length(r, xs, ys) for r in routes)

    if os.environ.get("ALE17_RAW") == "1":
        print(f"{D:.6f}")
        return

    if D <= 1e-12:
        # only possible when n==0; treat as perfect
        print(1.0)
        return

    D_cw = clarke_wright_distance(n, K, Q, xs, ys, dem)
    score = D_cw / D
    print(f"{score:.6f}")


if __name__ == "__main__":
    main()
