# TIER: strong
# Insight: a dollar's value is NOT the recipient's own shortfall -- it is the
# dollar's marginal effect on the clearing FIXED POINT. Money handed to a bank
# that is already able to pay everyone (or that nobody downstream depends on)
# changes nothing; money handed to a node whose own default is dragging a
# chain of dependents into default can resolve the whole chain at once. So we
# never rank banks by their own gap. Instead we *simulate*: repeatedly probe,
# for every currently-defaulting bank, what re-clearing the network looks like
# if that bank received the next slice of budget, and hand the slice to
# whichever probe reduced total system-wide shortfall the most. Re-simulating
# after every slice lets the search see cascades unlock as we go, instead of
# committing to a single static ranking.
import sys

def clearing(pbar, incoming, e_ext, max_iters):
    n = len(pbar)
    p = list(pbar)
    for _ in range(max_iters):
        changed = False
        newp = [0.0] * n
        for i in range(n):
            pb = pbar[i]
            if pb <= 0:
                newp[i] = 0.0
                continue
            inflow = 0.0
            for (j, w) in incoming[i]:
                pbj = pbar[j]
                if pbj > 0:
                    inflow += (w / pbj) * p[j]
            val = e_ext[i] + inflow
            pi_new = pb if val >= pb else val
            newp[i] = pi_new
            if abs(pi_new - p[i]) > 1e-9:
                changed = True
        p = newp
        if not changed:
            break
    return p


def total_shortfall(pbar, p):
    return sum(pbar[i] - p[i] for i in range(len(pbar)))


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); c = int(next(it))
    e = [float(next(it)) for _ in range(n)]
    m = int(next(it))
    pbar = [0.0] * n
    incoming = [[] for _ in range(n)]
    for _ in range(m):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        w = float(next(it))
        pbar[u] += w
        incoming[v].append((u, w))

    max_iters = min(n + 5, 60)

    e_cur = list(e)
    budget = float(c)
    STEPS = 14
    chunk0 = budget / STEPS if STEPS > 0 else 0.0

    for step in range(STEPS):
        if budget <= 1e-9:
            break
        p_now = clearing(pbar, incoming, e_cur, max_iters)
        defaulting = [i for i in range(n) if pbar[i] - p_now[i] > 1e-6]
        if not defaulting:
            break
        chunk = min(chunk0, budget)
        base_F = total_shortfall(pbar, p_now)
        best_i, best_F = None, base_F
        for i in defaulting:
            trial_e = e_cur[:]
            trial_e[i] += chunk
            p_trial = clearing(pbar, incoming, trial_e, max_iters)
            F_trial = total_shortfall(pbar, p_trial)
            if F_trial < best_F - 1e-9:
                best_F = F_trial
                best_i = i
        if best_i is None:
            # no defaulting bank benefits from a chunk here (rare); spend
            # the rest as directly as possible on the single largest
            # residual shortfall to avoid wasting the budget.
            best_i = max(defaulting, key=lambda i: pbar[i] - p_now[i])
        e_cur[best_i] += chunk
        budget -= chunk

    # spend any leftover (rounding) on the currently worst residual shortfall
    if budget > 1e-9:
        p_now = clearing(pbar, incoming, e_cur, max_iters)
        residual = [pbar[i] - p_now[i] for i in range(n)]
        i = max(range(n), key=lambda i: residual[i])
        if residual[i] > 1e-9:
            e_cur[i] += budget
            budget = 0.0

    delta = [e_cur[i] - e[i] for i in range(n)]
    # numerical safety: clip tiny negatives, and leave a small safety margin
    # below the budget so per-value decimal rounding on print can never push
    # the printed sum over C.
    delta = [max(0.0, d) for d in delta]
    s = sum(delta)
    safety_cap = c - 1e-3
    if s > safety_cap and s > 0:
        scale = safety_cap / s
        delta = [d * scale for d in delta]

    print(" ".join("%.6f" % x for x in delta))

main()
