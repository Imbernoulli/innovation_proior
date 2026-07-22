#!/usr/bin/env python3
"""
gen.py <testId>  -- prints the LIGHT-TRAFFIC train sample for one instance of
the merge-junction congestion-law extrapolation problem.

Each line after the header is a data row:  "<x0> <x1> <x2> <y>"
  x0 = flow on the monitored (target) link
  x1 = flow on feeder link 1 (partially spills into the target link's queue)
  x2 = flow on feeder link 2 (partially spills into the target link's queue)
  y  = measured travel delay on the target link (multiplicative noise)

The hidden ground-truth law (shared congestion exponent, per-feeder coupling
strengths, scale, capacity), the random seed, and the held-out HEAVY-traffic
region are NOT emitted here -- data rows only. They
live exclusively inside the checker (verify.py), which contains an identical
copy of the derivation logic below.
"""
import sys


# ---- deterministic PRNG (LCG); identical logic must live in verify.py ----
def _rng(seed):
    state = [(seed * 2654435761 + 12345) & 0x7FFFFFFF]

    def nxt():
        state[0] = (1103515245 * state[0] + 12345) & 0x7FFFFFFF
        return state[0] / 0x7FFFFFFF

    return nxt


def derive_params(test_id):
    """Ground-truth law:  y = b * ((x0 + w1*x1 + w2*x2) / c) ** k
    a shared SUPERLINEAR congestion exponent k, own-flow weight normalized
    to 1, and per-feeder coupling weights w1, w2 (spillover strengths)."""
    r = _rng(1000 + test_id)
    k = 1.5 + 1.2 * r()    # shared congestion exponent, superlinear in [1.5, 2.7]
    c = 8.0 + 4.0 * r()    # effective-load capacity scale in [8, 12]
    b = 2.5 + 2.0 * r()    # congestion scale coefficient in [2.5, 4.5]
    w1 = 0.3 + 0.4 * r()   # feeder-1 coupling (spillover) strength in [0.3, 0.7]
    w2 = 0.2 + 0.4 * r()   # feeder-2 coupling (spillover) strength in [0.2, 0.6]
    return k, c, b, w1, w2


def n_train(test_id):
    return 60 + 6 * (test_id - 1)          # difficulty ladder: 60 .. 114 rows


def noise_rel(test_id):
    return 0.18 + 0.02 * (test_id - 1)     # difficulty ladder: more noise later


LOW_LO, LOW_HI = 0.10, 3.0   # LIGHT-traffic (train) box for x0, x1, x2


def clean_y(x0, x1, x2, params):
    k, c, b, w1, w2 = params
    u = x0 + w1 * x1 + w2 * x2
    return b * (u / c) ** k


def make_train(test_id):
    params = derive_params(test_id)
    n = n_train(test_id)
    nr = noise_rel(test_id)
    rx = _rng(3000 + test_id)   # train x-stream
    rn = _rng(5000 + test_id)   # train noise-stream
    rows = []
    for _ in range(n):
        x0 = LOW_LO + (LOW_HI - LOW_LO) * rx()
        x1 = LOW_LO + (LOW_HI - LOW_LO) * rx()
        x2 = LOW_LO + (LOW_HI - LOW_LO) * rx()
        clean = clean_y(x0, x1, x2, params)
        y = clean * (1.0 + nr * (2.0 * rn() - 1.0))
        rows.append((x0, x1, x2, y))
    return rows


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    test_id = int(sys.argv[1])
    rows = make_train(test_id)
    out = [str(len(rows))]
    for x0, x1, x2, y in rows:
        out.append("%.10g %.10g %.10g %.10g" % (x0, x1, x2, y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
