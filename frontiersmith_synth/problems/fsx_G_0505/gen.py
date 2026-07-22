#!/usr/bin/env python3
import math
import random
import sys


def coeffs(t):
    rng = random.Random(90731 + 104729 * t)
    return (
        rng.uniform(0.010, 0.040),   # offset
        rng.uniform(0.070, 0.150),   # diffusion-like SEI scale
        rng.uniform(0.46, 0.62),
        rng.uniform(0.20, 0.36),
        rng.uniform(0.012, 0.040),   # throughput interaction scale
        rng.uniform(0.34, 0.58),
        rng.uniform(0.010, 0.030),   # high-cycle/high-temp activation scale
        rng.uniform(1.25, 1.55),
        rng.uniform(0.45, 0.78),
        rng.uniform(1.45, 2.15),     # smooth knee in units of N/100
        rng.uniform(0.003, 0.010),   # cross stress term
    )


def softplus(z):
    if z > 45.0:
        return z
    if z < -45.0:
        return math.exp(z)
    return math.log1p(math.exp(z))


def fval(N, T, D, R, cf):
    q0, A, alpha, bt, B, bo, C, gamma, bh, knee, H = cf
    u = N / 100.0
    theta = (T - 20.0) / 12.0
    stress = D * R
    act = 0.55 * softplus((u - knee) / 0.55)

    sei = A * (u ** alpha) * math.exp(bt * theta) * (D ** 1.05)
    throughput = B * u * math.exp(bo * theta) * stress
    hot_knee = C * (act ** gamma) * math.exp(bh * theta) * (stress ** 1.15)
    cross = H * (u ** 0.80) * math.exp(0.35 * theta) * D * (0.25 + R) * (0.8 + 0.4 * theta)
    return q0 + sei + throughput + hot_knee + cross


def gen_train(t):
    n = 210 - 10 * (t - 1)
    sigma = 0.010 + 0.005 * t
    cf = coeffs(t)
    rng = random.Random(55621 + 65537 * t)
    rows = []
    for _ in range(n):
        if rng.random() < 0.72:
            N = rng.uniform(25.0, 180.0)
        else:
            N = rng.uniform(180.0, 320.0)
        T = rng.uniform(6.0, 24.0)
        D = rng.uniform(0.20, 0.72)
        R = rng.uniform(0.25, 1.15)
        y = fval(N, T, D, R, cf) + rng.gauss(0.0, sigma)
        rows.append((N, T, D, R, y))
    return rows


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: gen.py <testId>")
    t = int(sys.argv[1])
    if not (1 <= t <= 10):
        raise SystemExit("testId must be in 1..10")
    rows = gen_train(t)
    print(len(rows), t)
    for row in rows:
        print("%.10g %.10g %.10g %.10g %.10g" % row)


if __name__ == "__main__":
    main()
