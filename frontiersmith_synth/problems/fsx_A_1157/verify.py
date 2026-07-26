import sys
from fractions import Fraction

MAX_MACHINES = 60


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def simulate(types_list, cost, jobs):
    """Frozen list scheduler: fixed job-index priority order, precedence chains within
    a job, machine chosen as whichever purchased machine becomes free earliest (ties
    broken by ascending machine index). Returns the scenario makespan, or None on a
    pathological non-terminating configuration (guarded, should not occur)."""
    M = len(types_list)
    if M == 0:
        return None
    n_jobs = len(jobs)
    next_free = [0] * M
    cursor = [0] * n_jobs
    ready = [0] * n_jobs
    completion = [0] * n_jobs
    remaining = sum(len(j) for j in jobs)
    if remaining == 0:
        return 0
    max_iter = 20 * (remaining + n_jobs + M) + 2000
    it = 0
    while remaining > 0:
        it += 1
        if it > max_iter:
            return None
        m = min(range(M), key=lambda x: (next_free[x], x))
        t = next_free[m]
        chosen = -1
        for j in range(n_jobs):
            if cursor[j] < len(jobs[j]) and ready[j] <= t:
                chosen = j
                break
        if chosen == -1:
            pending = [j for j in range(n_jobs) if cursor[j] < len(jobs[j])]
            nxt = min(ready[j] for j in pending)
            if nxt <= t:
                return None  # would not make progress; guard
            next_free[m] = nxt
            continue
        k, w = jobs[chosen][cursor[chosen]]
        dur = w * cost[types_list[m]][k]
        finish = t + dur
        next_free[m] = finish
        cursor[chosen] += 1
        if cursor[chosen] < len(jobs[chosen]):
            ready[chosen] = finish
        else:
            completion[chosen] = finish
        remaining -= 1
    return max(completion)


def quantile_regret(types_list, cost, scenarios):
    regrets = []
    for jobs, oracle in scenarios:
        ms = simulate(types_list, cost, jobs)
        if ms is None:
            return None
        regrets.append(Fraction(ms, oracle))
    regrets.sort(reverse=True)
    idx = min(2, len(regrets) - 1)  # 3rd-worst (0-indexed rank 2)
    return regrets[idx]


def portfolio_list(counts, T):
    """Round-robin interleave across types so tie-break-by-index doesn't
    systematically starve whichever type was purchased in smaller quantity."""
    tl = []
    idx = [0] * T
    remaining = sum(counts)
    while remaining > 0:
        for t in range(T):
            if idx[t] < counts[t]:
                tl.append(t)
                idx[t] += 1
                remaining -= 1
    return tl


def main():
    try:
        data = open(sys.argv[1]).read().split()
        it = iter(data)
        T = int(next(it)); KINDS = int(next(it)); budget = int(next(it))
        cost = []; price = []
        for _ in range(T):
            c = [int(next(it)) for _ in range(KINDS)]
            p = int(next(it))
            cost.append(c); price.append(p)
        K = int(next(it))
        scenarios = []
        for _ in range(K):
            n_jobs = int(next(it)); oracle = int(next(it))
            jobs = []
            for _ in range(n_jobs):
                L = int(next(it))
                ops = []
                for _ in range(L):
                    k = int(next(it)); w = int(next(it))
                    ops.append((k, w))
                jobs.append(ops)
            scenarios.append((jobs, oracle))
    except Exception:
        fail("bad input")

    try:
        out_tokens = open(sys.argv[2]).read().split()
    except Exception:
        fail("cannot read output")

    if len(out_tokens) != T:
        fail("expected %d counts, got %d" % (T, len(out_tokens)))

    counts = []
    for tok in out_tokens:
        try:
            v = int(tok)
        except ValueError:
            fail("non-integer token %r" % tok)
        if v < 0:
            fail("negative count %r" % tok)
        counts.append(v)

    total_units = sum(counts)
    if total_units < 1:
        fail("empty portfolio")
    if total_units > MAX_MACHINES:
        fail("portfolio too large (%d > %d)" % (total_units, MAX_MACHINES))

    total_cost = sum(counts[t] * price[t] for t in range(T))
    if total_cost > budget:
        fail("over budget %d > %d" % (total_cost, budget))

    types_list = portfolio_list(counts, T)
    F_frac = quantile_regret(types_list, cost, scenarios)
    if F_frac is None:
        fail("simulation did not terminate")
    F = float(F_frac)

    # ---- internal baseline B: spend the whole budget on the single cheapest type ----
    cheapest = min(range(T), key=lambda t: (price[t], t))
    b_counts = [0] * T
    b_counts[cheapest] = max(1, budget // price[cheapest])
    B_frac = quantile_regret(portfolio_list(b_counts, T), cost, scenarios)
    B = float(B_frac)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
