# TIER: greedy
# The obvious approach: per-ride STATIC demand-curve pricing.
# For each ride assume the fast lane stays empty, so every interested visitor's
# willingness-to-pay is v_i * S0_j with S0_j = n_j / s_reg_j (the full regular wait).
# Pick the revenue-maximizing price on that static demand curve.  This ignores the
# crowding fixed point (and budgets): once the lane fills, savings collapse.
import sys

def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    M = int(next(it)); N = int(next(it)); LAM = int(next(it)); T = int(next(it))
    s_reg = [0]*M
    for j in range(M):
        s_reg[j] = int(next(it)); _sf = int(next(it)); _r = int(next(it))
    wtp = [[] for _ in range(M)]     # per ride: list of v_i (interested visitors)
    for i in range(N):
        v = int(next(it)); K = int(next(it)); ks = int(next(it))
        for _ in range(ks):
            j = int(next(it)) - 1
            wtp[j].append(v)
    out = []
    for j in range(M):
        vs = wtp[j]
        if not vs:
            out.append("1")
            continue
        n_j = len(vs)
        S0 = n_j / s_reg[j]                  # static (empty-fast-lane) max saving
        vs.sort(reverse=True)
        best_rev = -1.0; best_p = 1.0
        # candidate prices = each interested visitor's static WTP
        for k in range(1, n_j + 1):
            p = vs[k-1] * S0                 # price that keeps the top-k "willing"
            rev = p * k
            if rev > best_rev:
                best_rev = rev; best_p = p
        out.append("%.6f" % best_p)
    sys.stdout.write("\n".join(out) + "\n")

main()
