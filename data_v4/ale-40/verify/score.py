#!/usr/bin/env python3
"""Deterministic local scorer for "Simulated Epidemic Containment".

Usage:
    python3 score.py <instance_file> <solution_file>

Prints a single integer: the score. HIGHER is better. INFEASIBLE -> 0.

Scoring rule (see context.md "Evaluation settings"):

  * Instance:
        n m T b
        beta gamma kappa
        u_e v_e w_e          (m undirected weighted edges)
        I0_0 .. I0_{n-1}     (initial infected fraction per region)
    There are n regions; each region r carries SIR compartment fractions
    (S_r, I_r, R_r) with S_r + I_r + R_r = 1, starting at
    S_r = 1 - I0_r, I_r = I0_r, R_r = 0.

  * SOLUTION format (read from <solution_file>): a lockdown SCHEDULE. The solver
    writes T lines (line t = day t, 0-indexed). Line t starts with an integer
    c_t (the number of regions locked down on day t), followed by c_t distinct
    region ids in [0, n). Tokens are whitespace-separated; line breaks do not
    matter for parsing -- the scorer reads, for each of the T days in order, one
    count c_t and then c_t ids. The TOTAL token count must be exactly
    sum_t (1 + c_t).

  * FEASIBILITY (any violation -> score 0):
      - the file parses as the T (count, ids...) blocks above with no leftover and
        no missing tokens;
      - every c_t is an integer with 0 <= c_t <= b (the DAILY lockdown budget);
      - every region id is an integer in [0, n);
      - within a single day the locked ids are DISTINCT (no region locked twice
        on the same day).
    If any of these fail, the solution is INFEASIBLE and scores 0.

  * DYNAMICS (the deterministic SIR-on-a-graph control loop). For each day
    t = 0 .. T-1, in order:
      1. Read the locked set L_t from the schedule. factor_r = kappa if r in L_t
         else 1.0  (a locked region's transmission, internal and across its
         incident edges, is scaled by kappa for THIS day only).
      2. Force of infection on region r:
             lambda_r = factor_r * ( beta * I_r
                          + sum over neighbours j of region r:
                                  w_{rj} * beta * I_j * factor_j )
         (the cross-edge term is damped by BOTH endpoints' factors: locking
          either end of an edge cuts transmission across it).
      3. New infections in r: newinf_r = S_r * (1 - exp(-lambda_r))
         (this is in [0, S_r], so S_r stays >= 0 -- no clamping needed, but we
          clamp defensively). New recoveries: newrec_r = gamma * I_r.
      4. Simultaneous update (all regions use the SAME pre-update I values):
             S_r -= newinf_r
             I_r += newinf_r - newrec_r
             R_r += newrec_r
      5. Accumulate the objective:  total_new_infections += sum_r newinf_r.
    Lower total_new_infections is better.

  * SCORE (higher better), normalized against a deterministic baseline the scorer
    recomputes itself -- the "lock the b most-infected regions today" greedy:
        score = round(1_000_000 * baseline_infections / max(1e-9, solver_infections))
    The baseline scores ~1_000_000; a schedule that lets through FEWER infections
    scores more. INFEASIBLE -> 0.
"""
import sys
import math


# ----------------------------------------------------------------------------- IO
def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    n = int(next(it))
    m = int(next(it))
    T = int(next(it))
    b = int(next(it))
    beta = float(next(it))
    gamma = float(next(it))
    kappa = float(next(it))
    adj = [[] for _ in range(n)]
    for _ in range(m):
        u = int(next(it))
        v = int(next(it))
        w = float(next(it))
        adj[u].append((v, w))
        adj[v].append((u, w))
    I0 = [float(next(it)) for _ in range(n)]
    return n, m, T, b, beta, gamma, kappa, adj, I0


def read_schedule(path, n, T, b):
    """Parse + validate the schedule. Return list of T sets (locked ids) or None."""
    try:
        with open(path) as f:
            toks = f.read().split()
    except OSError:
        return None
    it = iter(toks)
    schedule = []
    for _t in range(T):
        try:
            c = next(it)
        except StopIteration:
            return None  # missing a day's count
        try:
            c = int(c)
        except ValueError:
            return None
        if c < 0 or c > b:
            return None  # over the daily budget (or negative)
        day = []
        seen = set()
        for _ in range(c):
            try:
                r = next(it)
            except StopIteration:
                return None  # promised c ids but ran out
            try:
                r = int(r)
            except ValueError:
                return None
            if r < 0 or r >= n:
                return None
            if r in seen:
                return None  # same region locked twice on one day
            seen.add(r)
            day.append(r)
        schedule.append(seen)
    # there must be NO leftover tokens (a strict, exact format)
    for _extra in it:
        return None
    return schedule


# --------------------------------------------------------------------- dynamics
def simulate(n, T, beta, gamma, kappa, adj, I0, schedule):
    """Run the deterministic SIR-on-a-graph loop with the given lockdown schedule.
    Returns total new infections over all days and regions (lower is better)."""
    S = [1.0 - I0[r] for r in range(n)]
    I = [I0[r] for r in range(n)]
    R = [0.0] * n
    total_new = 0.0
    for t in range(T):
        locked = schedule[t]
        factor = [kappa if r in locked else 1.0 for r in range(n)]
        newinf = [0.0] * n
        newrec = [0.0] * n
        for r in range(n):
            cross = 0.0
            for (j, w) in adj[r]:
                cross += w * beta * I[j] * factor[j]
            lam = factor[r] * (beta * I[r] + cross)
            ni = S[r] * (1.0 - math.exp(-lam))
            if ni < 0.0:
                ni = 0.0
            if ni > S[r]:
                ni = S[r]
            newinf[r] = ni
            newrec[r] = gamma * I[r]
        for r in range(n):
            S[r] -= newinf[r]
            I[r] += newinf[r] - newrec[r]
            R[r] += newrec[r]
            total_new += newinf[r]
    return total_new


def baseline_infections(n, T, b, beta, gamma, kappa, adj, I0):
    """Deterministic 'lock the b most-infected regions today' greedy.

    Each day, BEFORE the spread step, lock the b regions with the largest current
    infected fraction I_r (ties broken by smaller region id). This is the obvious
    myopic controller and the normalizer the score is measured against."""
    sched = []
    S = [1.0 - I0[r] for r in range(n)]
    I = [I0[r] for r in range(n)]
    R = [0.0] * n
    for t in range(T):
        order = sorted(range(n), key=lambda r: (-I[r], r))
        locked = set(order[:b])
        sched.append(locked)
        factor = [kappa if r in locked else 1.0 for r in range(n)]
        newinf = [0.0] * n
        newrec = [0.0] * n
        for r in range(n):
            cross = 0.0
            for (j, w) in adj[r]:
                cross += w * beta * I[j] * factor[j]
            lam = factor[r] * (beta * I[r] + cross)
            ni = S[r] * (1.0 - math.exp(-lam))
            if ni < 0.0:
                ni = 0.0
            if ni > S[r]:
                ni = S[r]
            newinf[r] = ni
            newrec[r] = gamma * I[r]
        for r in range(n):
            S[r] -= newinf[r]
            I[r] += newinf[r] - newrec[r]
            R[r] += newrec[r]
    # score the produced schedule with the canonical simulator (same numbers)
    return simulate(n, T, beta, gamma, kappa, adj, I0, sched)


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py <instance> <solution>\n")
        sys.exit(1)
    n, m, T, b, beta, gamma, kappa, adj, I0 = read_instance(sys.argv[1])

    schedule = read_schedule(sys.argv[2], n, T, b)
    if schedule is None:
        print(0)
        return

    solver = simulate(n, T, beta, gamma, kappa, adj, I0, schedule)
    base = baseline_infections(n, T, b, beta, gamma, kappa, adj, I0)
    score = int(round(1_000_000.0 * base / max(1e-9, solver)))
    print(score)


if __name__ == "__main__":
    main()
