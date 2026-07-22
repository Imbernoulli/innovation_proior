# TIER: greedy
# The obvious approach: "put heat where the target is hot, equal to the temperature I want."
# Matching pursuit on the RAW target residual, source amplitude = residual peak value.
# Flaws it cannot see: (1) one unit of source diffuses to a peak << 1, so setting the source
# equal to the desired temperature badly UNDERSHOOTS; (2) it picks peaks of the raw residual,
# so sharp high-frequency spikes in y lure it to place (or refuse) sources in the wrong places.
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

    # kernel = diffused unit source at origin; denom = <atom,atom> (same for every location)
    e = [[0.0] * N for _ in range(N)]; e[0][0] = 1.0
    K0 = diffuse(e, T, alpha)
    denom = sum(v * v for row in K0 for v in row)

    s = [[0.0] * N for _ in range(N)]
    res = [row[:] for row in y]

    for _ in range(12):
        # peak of the RAW residual (fooled by high-frequency spikes)
        bi = bj = 0; bv = res[0][0]
        for i in range(N):
            ri = res[i]
            for j in range(N):
                if ri[j] > bv:
                    bv = ri[j]; bi = i; bj = j
        if bv <= 0.0:
            break
        a = bv  # naive amplitude = desired temperature (undershoots)
        # <res, atom>
        corr = 0.0
        for i in range(N):
            ri = res[i]; ki = K0[(i - bi) % N]
            for j in range(N):
                corr += ri[j] * ki[(j - bj) % N]
        gain = 2.0 * a * corr - a * a * denom
        if gain < cost:
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
