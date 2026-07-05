#!/usr/bin/env python3
"""gen.py <testId>  -- print ONE low-load ISP-telemetry training log.

Difficulty ladder (testId 1..N): larger testId => fewer training rows and higher
measurement noise, so the hidden congestion law is harder to recover and to
extrapolate into the (unobserved) high-load regime.

STDOUT is DATA ROWS ONLY: three whitespace-separated feature floats plus the
measured congestion metric, i.e. "rho cv hop y" per line. The hidden law, its
coefficients, the sampling seed and the held-out high-load region are NEVER
printed here -- the ground truth lives only inside the grader (verify.py).
"""
import sys, random


def hidden_law(rho, cv, hop):
    # M/G/1-flavoured congestion metric (mean normalized queueing delay).
    # The Pollaczek-Khinchine-style pole rho/(1-rho) is the crux: at LOW load it
    # looks almost linear, but it blows up as utilization -> 1. Kept private; the
    # grader holds an identical copy. Solvers must rediscover this FORM.
    q = rho / (1.0 - rho)
    return 0.5 + 0.9 * hop + 1.3 * (1.0 + 0.6 * cv) * q


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        return 1
    t = int(sys.argv[1])
    if t < 1:
        t = 1

    rng = random.Random(4360 + t)
    n_rows = 380 - 26 * t          # t=1 -> 354 rows ... t=10 -> 120 rows
    noise = 0.04 + 0.03 * t        # low-load telemetry noise (grows with t)

    out = []
    for _ in range(n_rows):
        # NOMINAL / LOW-LOAD operating regime actually observed by the ISP:
        rho = rng.uniform(0.05, 0.60)   # link utilization (offered load / capacity)
        cv = rng.uniform(0.0, 1.5)      # service-time variability index (packet-size mix)
        hop = rng.uniform(0.0, 1.0)     # normalized path length / propagation share
        y = hidden_law(rho, cv, hop) + rng.gauss(0.0, noise)
        out.append("%.6f %.6f %.6f %.6f" % (rho, cv, hop, y))

    sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
