# TIER: strong
# Simulated-annealing / coordinate search over integer schedules to drive the
# peak self-interference index c well below the flat schedule (2.0), reaching
# ~1.55-1.60. Fully seeded -> deterministic. Uses numpy if present for a fast
# convolution; falls back to a lighter pure-Python search otherwise.
import sys, math

def main():
    n, V = map(int, sys.stdin.read().split())
    seed0 = 1000 + 17 * n

    try:
        import numpy as np

        def cval(a):
            a = np.asarray(a, dtype=np.float64)
            S = a.sum()
            if S <= 0:
                return 9e18
            return 2 * n * float(np.convolve(a, a).max()) / (S * S)

        def anneal(iters, seed):
            rng = np.random.default_rng(seed)
            cur = np.full(n, float(V))
            curv = cval(cur)
            best = cur.copy(); bestv = curv
            for it in range(iters):
                frac = 1.0 - it / iters
                T = 0.02 * frac
                maxstep = max(1, int(V * (0.05 + 0.45 * frac)))
                i = int(rng.integers(n))
                old = cur[i]
                step = int(rng.integers(1, maxstep + 1))
                cur[i] = max(0.0, old - step) if rng.random() < 0.5 else min(float(V), old + step)
                v = cval(cur)
                if v <= curv or rng.random() < math.exp((curv - v) / max(1e-9, T * curv)):
                    curv = v
                    if v < bestv:
                        bestv = v; best = cur.copy()
                else:
                    cur[i] = old
            return best, bestv

        best = None; bestv = 9e18
        for r in range(4):
            b, v = anneal(30000, seed0 + r * 7919)
            if v < bestv:
                bestv = v; best = b
        out = [int(round(x)) for x in best]

    except Exception:
        # pure-Python fallback (no numpy): lighter single-restart hillclimb
        import random
        random.seed(seed0)

        def conv_peak(a):
            best = 0
            for k in range(2 * n - 1):
                s = 0
                lo = max(0, k - n + 1); hi = min(k, n - 1)
                for i in range(lo, hi + 1):
                    s += a[i] * a[k - i]
                if s > best:
                    best = s
            return best

        def cval(a):
            S = sum(a)
            if S <= 0:
                return 9e18
            return 2 * n * conv_peak(a) / (S * S)

        cur = [V] * n
        curv = cval(cur)
        best = cur[:]; bestv = curv
        iters = 4000
        for it in range(iters):
            frac = 1.0 - it / iters
            maxstep = max(1, int(V * (0.05 + 0.45 * frac)))
            i = random.randrange(n)
            old = cur[i]
            step = random.randint(1, maxstep)
            cur[i] = max(0, old - step) if random.random() < 0.5 else min(V, old + step)
            v = cval(cur)
            if v <= curv:
                curv = v
                if v < bestv:
                    bestv = v; best = cur[:]
            else:
                cur[i] = old
        out = best

    sys.stdout.write(" ".join(map(str, out)) + "\n")

if __name__ == "__main__":
    main()
