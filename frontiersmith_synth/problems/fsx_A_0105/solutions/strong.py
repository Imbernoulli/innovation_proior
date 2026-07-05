# TIER: strong
# Seeded coordinate-descent local search that pushes C1 below the uniform value (2.0).
# Starts from the uniform profile and repeatedly perturbs a single segment's energy,
# accepting the change only if it does not increase C1. The autoconvolution g and its peak
# are maintained incrementally so each proposal costs O(n) -> reaches C1 ~ 1.7.
import sys, random

def main():
    n = int(sys.stdin.read().split()[0])

    f = [1.0] * n
    g = [0.0] * (2 * n - 1)
    for i in range(n):
        base = i
        for j in range(n):
            g[base + j] += 1.0
    S = float(n)
    curmax = max(g)

    rng = random.Random(12345)
    budget = 8_000_000
    K = min(30000, budget // max(1, n))

    for _ in range(K):
        i = rng.randrange(n)
        old = f[i]
        delta = rng.uniform(-0.25, 0.25)
        new = old + delta
        if new < 0.0:
            new = 0.0
        d = new - old
        if d == 0.0:
            continue
        # apply the change to g
        g[2 * i] += new * new - old * old
        for j in range(n):
            if j != i:
                g[i + j] += 2.0 * d * f[j]
        newS = S + d
        m = max(g)
        cur_val = 2.0 * n * curmax / (S * S)
        new_val = 2.0 * n * m / (newS * newS)
        if new_val <= cur_val:
            f[i] = new
            S = newS
            curmax = m
        else:
            # revert
            g[2 * i] -= new * new - old * old
            for j in range(n):
                if j != i:
                    g[i + j] -= 2.0 * d * f[j]

    sys.stdout.write(" ".join("%.6f" % x for x in f) + "\n")

if __name__ == "__main__":
    main()
