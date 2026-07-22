# TIER: strong
"""The insight: reformulate as "what is a dollar that becomes free at period t
worth, if I play optimally from here on?" Because capital is frozen the moment
it's committed (irreversible-commit) and only re-enters the pool when its lock
matures (compounding-lock-in), the value of committing to instrument k at
period t depends on the FULL window [t, t+L_k-1] of the known rate schedule
(rate-schedule-anticipation), not just the rate this instant. That gives a
clean backward recurrence over "release time" t:

    V[T+1] = 1
    V[t]   = max( V[t+1],                                   # wait one period
                  max_k  R(k,t) * V[min(t+L_k, T+1)] )       # commit to k now
    R(k,t) = product over s in [t, min(t+L_k-1,T)] of (1 + rate[s][k])

V[t] is the best achievable multiplier for a dollar first free at period t.
Since the payoff of committing an amount x is linear in x (no per-instrument
capacity limits), any batch of cash that becomes free at the same period faces
the identical decision, so replaying the argmax forward realizes exactly
C0 * V[1] -- the true optimum of this formulation, timing every commitment so
each lock matures right as (or waits for) the best remaining window opens.
"""
import sys


def main():
    data = sys.stdin.read().split()
    p = 0
    T = int(data[p]); p += 1
    K = int(data[p]); p += 1
    C0 = int(data[p]); p += 1
    L = [int(data[p + i]) for i in range(K)]; p += K
    rate = []
    for _t in range(T):
        row = [int(data[p + i]) for i in range(K)]
        p += K
        rate.append(row)

    # precompute window multiplier and target bucket for every (t,k)
    win_mult = [[1.0] * K for _ in range(T + 2)]
    win_bucket = [[T + 1] * K for _ in range(T + 2)]
    for t in range(1, T + 1):
        for k in range(K):
            Lk = L[k]
            window_end = min(t + Lk - 1, T)
            mult = 1.0
            for s in range(t, window_end + 1):
                mult *= (1.0 + rate[s - 1][k] / 10000.0)
            win_mult[t][k] = mult
            win_bucket[t][k] = min(t + Lk, T + 1)

    V = [1.0] * (T + 2)
    for t in range(T, 0, -1):
        best = V[t + 1]
        for k in range(K):
            val = win_mult[t][k] * V[win_bucket[t][k]]
            if val > best:
                best = val
        V[t] = best

    free_cash = [0.0] * (T + 2)
    free_cash[1] = float(C0)
    out_rows = []

    for t in range(1, T + 1):
        avail = free_cash[t]
        row = [0.0] * K
        if avail > 0:
            best_k, best_val = -1, V[t + 1]
            for k in range(K):
                val = win_mult[t][k] * V[win_bucket[t][k]]
                if val > best_val + 1e-12:
                    best_val, best_k = val, k
            if best_k == -1:
                free_cash[t + 1] += avail
            else:
                row[best_k] = avail
                bucket = win_bucket[t][best_k]
                free_cash[bucket] += avail * win_mult[t][best_k]
        out_rows.append(" ".join(("%.6f" % x) for x in row))

    sys.stdout.write("\n".join(out_rows) + "\n")


if __name__ == "__main__":
    main()
