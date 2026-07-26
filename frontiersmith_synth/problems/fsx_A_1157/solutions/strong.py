# TIER: strong
import sys
from fractions import Fraction


def simulate(types_list, cost, jobs):
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
                return None
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
    idx = min(2, len(regrets) - 1)
    return regrets[idx]


def portfolio_list(counts, T):
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
    data = sys.stdin.read().split()
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

    def eval_counts(c):
        tl = portfolio_list(c, T)
        if not tl:
            return None
        return quantile_regret(tl, cost, scenarios)

    counts = [0] * T
    cheapest = min(range(T), key=lambda t: (price[t], t))
    counts[cheapest] = 1
    spent = price[cheapest]
    best_val = eval_counts(counts)

    # Insight: instead of sizing a portfolio for the *average* job mix, search
    # directly on the real multi-scenario quantile-regret objective -- buying
    # against the operation-kind bottleneck distribution across scenarios, which
    # naturally favors a mix of a couple of specialists plus "wasteful" generalist
    # units the frozen scheduler can always fall back on.
    while spent < budget:
        best_choice = -1
        best_new_val = None
        for t in range(T):
            if spent + price[t] > budget:
                continue
            counts[t] += 1
            v = eval_counts(counts)
            counts[t] -= 1
            if v is not None and (best_new_val is None or v < best_new_val):
                best_new_val = v; best_choice = t
        if best_choice == -1:
            break
        counts[best_choice] += 1
        spent += price[best_choice]
        best_val = best_new_val

    # Local search: swap one purchased unit for a different type if it strictly
    # improves the true quantile regret (escapes the marginal-greedy's local optima).
    improved = True
    passes = 0
    while improved and passes < 8:
        improved = False
        passes += 1
        for a in range(T):
            for b in range(T):
                if a == b or counts[a] == 0:
                    continue
                new_spent = spent - price[a] + price[b]
                if new_spent > budget:
                    continue
                counts[a] -= 1; counts[b] += 1
                v = eval_counts(counts)
                if v is not None and best_val is not None and v < best_val:
                    best_val = v
                    spent = new_spent
                    improved = True
                else:
                    counts[a] += 1; counts[b] -= 1

    print(" ".join(map(str, counts)))


if __name__ == "__main__":
    main()
