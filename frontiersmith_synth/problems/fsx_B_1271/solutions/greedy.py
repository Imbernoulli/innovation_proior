# TIER: greedy
# The obvious "recipe" solution: rank purely by baseline credit score and assign
# a limit tier proportional to the score percentile -- exactly the trap the
# problem describes. It never looks at the per-tier default/utilization tables
# to decide WHO gets WHAT tier; it only consults them afterwards, to trim tiers
# down (starting from the lowest-score applicants, protecting the "best"
# customers by score) if the naive assignment would break the loss cap.
import sys

SCORE_LO, SCORE_HI = 300, 850


def profit_loss(L, m, d_bps, u_bps, apr_bps, sev_bps):
    if m == 0:
        return 0, 0
    balance = L[m] * u_bps // 10000
    revenue = balance * apr_bps // 10000 * (10000 - d_bps) // 10000
    loss = balance * d_bps // 10000 * sev_bps // 10000
    return revenue - loss, loss


def main():
    toks = sys.stdin.read().split()
    p = 0
    N = int(toks[p]); p += 1
    M = int(toks[p]); p += 1
    L = [0] * (M + 1)
    for m in range(1, M + 1):
        L[m] = int(toks[p]); p += 1
    apr_bps = int(toks[p]); p += 1
    sev_bps = int(toks[p]); p += 1
    cap = int(toks[p]); p += 1

    scores = [0] * N
    d_tab = [None] * N
    u_tab = [None] * N
    for i in range(N):
        scores[i] = int(toks[p]); p += 1
        d = [0] * (M + 1)
        u = [0] * (M + 1)
        for m in range(1, M + 1):
            d[m] = int(toks[p]); p += 1
            u[m] = int(toks[p]); p += 1
        d_tab[i] = d; u_tab[i] = u

    def loss_of(i, m):
        _, l = profit_loss(L, m, d_tab[i][m] if m else 0, u_tab[i][m] if m else 0, apr_bps, sev_bps)
        return l

    tier = [0] * N
    for i in range(N):
        frac = (scores[i] - SCORE_LO) / float(SCORE_HI - SCORE_LO)
        t = int(round(frac * M))
        tier[i] = max(0, min(M, t))

    total_loss = sum(loss_of(i, tier[i]) for i in range(N))
    order_low_first = sorted(range(N), key=lambda i: scores[i])
    idx = 0
    while total_loss > cap and idx < N:
        i = order_low_first[idx]
        if tier[i] > 0:
            old_l = loss_of(i, tier[i])
            tier[i] -= 1
            new_l = loss_of(i, tier[i])
            total_loss += (new_l - old_l)
        else:
            idx += 1

    print(" ".join(str(x) for x in tier))


if __name__ == "__main__":
    main()
