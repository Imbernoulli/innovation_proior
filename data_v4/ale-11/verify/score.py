#!/usr/bin/env python3
"""
Deterministic scorer for "Vehicle Dispatch over Time" (ale-11).

Usage:
    python3 score.py <instance_file> <solution_file>
    # or: python3 score.py <instance_file>   < solution_on_stdin

Prints a single integer: the score = number of FULFILLED rides, OR 0 if the
solution is INFEASIBLE for any reason (the feasibility -> 0 floor).

Solution format (stdout of the solver):
    M
    then M lines, each "v i":  vehicle v (0..V-1) serves request i (0..N-1).

  (M may be 0, meaning "serve nobody" -- that is FEASIBLE and scores 0.)

Replay rule (must match the solver exactly):
  - Travel time between two cells = Manhattan distance |dx| + |dy| (integer).
  - Each vehicle v starts free at its start cell (sx,sy) at time t = 0.
  - The M assignment lines are processed IN OUTPUT ORDER. For a line "v i", the
    vehicle v (at its current cell vc and current free-time vt) serves request i:
        start  = max(vt + manhattan(vc, pickup_i), r_i)
        if start > e_i                         -> INFEASIBLE (missed window)
        finish = start + manhattan(pickup_i, dropoff_i)
        if finish > T                          -> INFEASIBLE (past horizon)
    On success the vehicle becomes free at dropoff_i at time finish.
  - Each request id may be assigned AT MOST ONCE across the whole output.
    Duplicate request ids, an out-of-range vehicle id (not in [0,V-1]) or
    request id (not in [0,N-1]), a malformed line, or wrong count => INFEASIBLE.

The score is the count of fulfilled rides == M (every listed assignment must be
feasible; one infeasible assignment floors the whole solution to 0). The solver
therefore must only list rides it can actually complete.
"""
import sys


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    V = int(next(it)); N = int(next(it)); T = int(next(it))
    starts = []
    for _ in range(V):
        sx = int(next(it)); sy = int(next(it))
        starts.append((sx, sy))
    reqs = []
    for _ in range(N):
        px = int(next(it)); py = int(next(it))
        qx = int(next(it)); qy = int(next(it))
        r = int(next(it)); e = int(next(it))
        reqs.append((px, py, qx, qy, r, e))
    return V, N, T, starts, reqs


def man(ax, ay, bx, by):
    return abs(ax - bx) + abs(ay - by)


def score(instance_path, sol_text):
    V, N, T, starts, reqs = read_instance(instance_path)

    toks = sol_text.split()
    if not toks:
        return 0
    it = iter(toks)
    try:
        M = int(next(it))
    except (StopIteration, ValueError):
        return 0
    if M < 0 or M > N:
        return 0

    assigns = []
    try:
        for _ in range(M):
            v = int(next(it))
            i = int(next(it))
            assigns.append((v, i))
    except (StopIteration, ValueError):
        return 0  # fewer/garbled tokens than M announced

    # vehicle running state: (cell_x, cell_y, free_time)
    vcx = [s[0] for s in starts]
    vcy = [s[1] for s in starts]
    vt = [0] * V

    used = set()
    for (v, i) in assigns:
        if v < 0 or v >= V:
            return 0
        if i < 0 or i >= N:
            return 0
        if i in used:
            return 0          # a request served twice -> infeasible
        used.add(i)
        px, py, qx, qy, r, e = reqs[i]
        start = vt[v] + man(vcx[v], vcy[v], px, py)
        if start < r:
            start = r          # wait for release
        if start > e:
            return 0           # missed the pickup window -> infeasible
        finish = start + man(px, py, qx, qy)
        if finish > T:
            return 0           # cannot finish within horizon -> infeasible
        # advance the vehicle
        vcx[v] = qx
        vcy[v] = qy
        vt[v] = finish

    return M                   # number of fulfilled rides


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
