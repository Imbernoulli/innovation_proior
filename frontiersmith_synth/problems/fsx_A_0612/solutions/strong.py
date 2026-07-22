# TIER: strong
# Insight: F is a low-pass filter, so the reachable target lives in the low-frequency subspace
# and the high-frequency detail of y is unreachable. Two moves the greedy misses:
#   (1) Select on the LOW-PASSED residual  F(res)  instead of the raw residual, so high-frequency
#       spikes cannot lure a source into the wrong cell (projection onto the reachable subspace).
#   (2) Use the LEAST-SQUARES amplitude a = <res,atom>/<atom,atom>, which is exactly the factor
#       that undoes the diffusion attenuation -- no undershoot.
# Add a source only while its marginal error reduction a^2*<atom,atom> exceeds the per-source cost.
import sys

def diffuse(u, steps, alpha):
    n = len(u)
    cur = [row[:] for row in u]
    for _ in range(steps):
        nxt = [[0.0] * n for _ in range(n)]
        for i in range(n):
            up = cur[(i - 1) % n]; dn = cur[(i + 1) % n]; me = cur[i]
            for j in range(n):
                lap = up[j] + dn[j] + me[(j - 1) % n] + me[(j + 1) % n] - 4.0 * me[j]
                nxt[i][j] = me[j] + alpha * lap
        cur = nxt
    return cur

def main():
    toks = iter(sys.stdin.read().split())
    N = int(next(toks)); T = int(next(toks))
    alpha = float(next(toks)); cost = float(next(toks))
    y = [[float(next(toks)) for _ in range(N)] for _ in range(N)]

    e = [[0.0] * N for _ in range(N)]; e[0][0] = 1.0
    K0 = diffuse(e, T, alpha)
    denom = sum(v * v for row in K0 for v in row)

    s = [[0.0] * N for _ in range(N)]
    res = [row[:] for row in y]

    for _ in range(15):
        lp = diffuse(res, T, alpha)          # low-pass projection; (F res)[p] = <res, atom_p>
        bi = bj = 0; bv = lp[0][0]
        for i in range(N):
            li = lp[i]
            for j in range(N):
                if li[j] > bv:
                    bv = li[j]; bi = i; bj = j
        if bv <= 0.0:
            break
        a = bv / denom                       # least-squares amplitude (undoes attenuation)
        if a <= 0.0:
            break
        if a * a * denom < cost:             # marginal fit gain below per-source charge -> stop
            break
        s[bi][bj] += a
        for i in range(N):
            ri = res[i]; ki = K0[(i - bi) % N]
            for j in range(N):
                ri[j] -= a * ki[(j - bj) % N]

    out = []
    for i in range(N):
        out.append(" ".join("%.9g" % s[i][j] for j in range(N)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
