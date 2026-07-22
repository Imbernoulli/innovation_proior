# TIER: strong
"""The insight: no polynomial degree ever drives the training residual to
the noise floor, because the true update is RATIONAL, not polynomial --
x(t+1) = (a + x(t)) / x(t-1). This single-parameter family algebraically
preserves the "reserve" invariant (x+1)(y+1)(x+y+a)/(x*y) every single step,
by construction -- so once 'a' is recovered, the rollout cannot wander off
the true orbit the way a curve-fit can.

Recover 'a' robustly: rearranging the recurrence, every training triple
(x(t-1), x(t), x(t+1)) satisfies a = x(t+1)*x(t-1) - x(t) exactly in the
noiseless case. Take the MEDIAN of this estimator over all training triples
(robust to the per-year shock), then locally refine it by a small grid
search that minimises one-step-ahead training residual around the median."""
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    vals = [float(v) for v in data[2:2 + n]]

    cands = []
    for i in range(1, n - 1):
        cands.append(vals[i + 1] * vals[i - 1] - vals[i])
    cands.sort()
    a_med = cands[len(cands) // 2]

    def train_sse(a):
        se = 0.0
        for i in range(1, n - 1):
            pred = (a + vals[i]) / vals[i - 1]
            se += (pred - vals[i + 1]) ** 2
        return se

    best_a, best_sse = a_med, train_sse(a_med)
    span = max(0.05, 0.3 * abs(a_med) + 0.05)
    for step in range(-40, 41):
        a_try = a_med + span * step / 40.0
        if a_try <= -0.999:
            continue
        s = train_sse(a_try)
        if s < best_sse:
            best_sse, best_a = s, a_try

    print("OUT (%.8f + x) / xk1" % best_a)


if __name__ == "__main__":
    main()
