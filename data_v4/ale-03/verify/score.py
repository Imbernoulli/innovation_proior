#!/usr/bin/env python3
"""
Deterministic scorer for "Drone Courier under Time Windows" (ale-03).

Usage:
    python3 score.py <instance_file> <solution_file>
    # or: python3 score.py <instance_file>   < solution_on_stdin

Prints a single integer: the score = number of served requests, OR 0 if the
solution is INFEASIBLE for any reason (the feasibility -> 0 floor).

Solution format (stdout of the solver):
    K
    then K integers: the request ids (1..N) visited, in order.
  (K may be 0, meaning "serve nobody" -- that is FEASIBLE and scores 0.)

Replay rule (must match the solver exactly):
  - The drone starts at the depot (id 0, coord (L/2,L/2)) at time t = 0.
  - Travel time from p to q is ceil(euclidean_distance(p, q)).
  - Arriving at request i at time `arr`: if arr < r_i the drone WAITS until r_i
    (start = r_i); the START of service must satisfy start <= d_i, else INFEASIBLE.
    After service, time = start + s_i.
  - After the last served request the drone must travel back to the depot and
    arrive by time T, else INFEASIBLE.
  - Each request id may appear AT MOST ONCE; ids must be in [1, N]; duplicates,
    out-of-range ids, or a malformed solution => INFEASIBLE (score 0).
"""
import sys, math

L = 1000


def ceil_dist(ax, ay, bx, by):
    d = math.hypot(ax - bx, ay - by)
    # robust ceil to integer travel time
    c = math.ceil(d - 1e-9)
    return int(c)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    N = int(next(it)); T = int(next(it))
    reqs = []  # 1-indexed; index 0 placeholder (depot handled separately)
    reqs.append((L // 2, L // 2, 0, 0, 0))  # depot at id 0
    for _ in range(N):
        x = int(next(it)); y = int(next(it))
        r = int(next(it)); d = int(next(it)); s = int(next(it))
        reqs.append((x, y, r, d, s))
    return N, T, reqs


def score(instance_path, sol_text):
    N, T, reqs = read_instance(instance_path)
    toks = sol_text.split()
    if not toks:
        # empty output is treated as infeasible (solver must at least print K)
        return 0
    it = iter(toks)
    try:
        K = int(next(it))
    except StopIteration:
        return 0
    if K < 0 or K > N:
        return 0
    route = []
    try:
        for _ in range(K):
            route.append(int(next(it)))
    except StopIteration:
        return 0  # fewer ids than K announced

    # uniqueness + range
    seen = set()
    for rid in route:
        if rid < 1 or rid > N:
            return 0
        if rid in seen:
            return 0
        seen.add(rid)

    # replay
    cx, cy = reqs[0][0], reqs[0][1]   # depot
    t = 0
    for rid in route:
        x, y, r, d, s = reqs[rid]
        t += ceil_dist(cx, cy, x, y)
        if t < r:
            t = r            # wait
        if t > d:
            return 0         # missed the window -> infeasible
        t += s               # service
        cx, cy = x, y
    # return to depot
    t += ceil_dist(cx, cy, reqs[0][0], reqs[0][1])
    if t > T:
        return 0             # didn't make it back in time -> infeasible

    return len(route)        # number of served requests


def main():
    inst = sys.argv[1]
    if len(sys.argv) >= 3:
        with open(sys.argv[2]) as f:
            sol_text = f.read()
    else:
        sol_text = sys.stdin.read()
    print(score(inst, sol_text))


if __name__ == "__main__":
    main()
