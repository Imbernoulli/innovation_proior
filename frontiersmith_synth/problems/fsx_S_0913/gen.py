#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN trace to stdout.

Plague scrolls counted only on market days.

A hidden contagion grows exponentially at rate `beta` in the pre-decree window
(days t = 0 .. n-1).  What the town actually RECORDS each day is not the true
new-case count X(t) but a REPORTED tally that has been passed through a fixed
reporting operator: a market-day rhythm (a weekly multiplier w[t mod 7], with
one particularly quiet day) and a soft scribe-capacity saturation (a
Michaelis-Menten cap -- the scribes can only inscribe so many scrolls on any
single day, market or not).

On day `n` a magistrate's decree takes effect: the contagion's growth rate is
multiplied by a KNOWN factor `f` (given).  The market rhythm and the scribes'
capacity are administrative facts of the town, untouched by the decree.

STDOUT prints ONLY: a header "<n> <test_id> <f>" then n rows "<t> <reported>".
The hidden growth rate, initial load, weekly rhythm and scribe cap are NEVER
printed -- they live only in this function (duplicated, not imported, inside
verify.py) and are reconstructed solely from the test id.
"""
import sys, random, math


def get_params(tid):
    rng = random.Random(900001 + tid * 104729)
    diff = (tid - 1) / 9.0  # 0..1 difficulty ramp across the ladder

    beta = 0.070 + 0.035 * rng.random()      # pre-decree daily growth rate
    X0 = 4.0 + 6.0 * rng.random()             # initial latent daily incidence
    factor = 0.30 + 0.30 * rng.random()       # known post-decree growth multiplier
    n = 28 + 2 * (tid % 6)                    # training window length (28..38)
    H = 12 + (tid % 4)                        # held-out horizon (12..15), not printed

    base_w = [1.15, 0.95, 0.80, 0.90, 1.05, 1.35, 0.55]   # market-day rhythm shape
    amp = 0.35 + 0.55 * diff                   # rhythm swing grows with difficulty
    w = []
    for k in range(7):
        jitter = 1.0 + (rng.random() - 0.5) * 0.25
        val = base_w[k] * jitter
        if k == 6:
            val *= (1.0 - 0.55 * amp)          # the quiet day gets quieter
        w.append(max(0.04, val))
    m = sum(w) / 7.0
    w = [v / m for v in w]                     # normalise mean rhythm to 1.0

    peak_load = X0 * math.exp(beta * (n - 1)) * max(w)
    cap_ratio = max(1.08, 4.2 - 3.0 * diff)     # cap binds harder as difficulty rises
    Cap = peak_load * cap_ratio

    return dict(beta=beta, X0=X0, factor=factor, n=n, H=H, w=w, Cap=Cap)


def true_load(t, p):
    n, beta, X0, factor, w = p["n"], p["beta"], p["X0"], p["factor"], p["w"]
    if t < n:
        X = X0 * math.exp(beta * t)
    else:
        X_n = X0 * math.exp(beta * n)
        X = X_n * math.exp(beta * factor * (t - n))
    dow = t % 7
    return X * w[dow]


def true_reported(t, p, tid, noise=True):
    """Latent load passed through the (regime-invariant) reporting operator."""
    load = true_load(t, p)
    Cap = p["Cap"]
    base = Cap * load / (Cap + load)
    if noise:
        nrng = random.Random(31337 * tid + 7 * t + 991)
        eps = (nrng.random() - 0.5) * 0.18
        base *= (1.0 + eps)
    return max(0.0, base)


def rounded_reported(t, p, tid, noise=True):
    return float(int(round(true_reported(t, p, tid, noise=noise))))


def main():
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    p = get_params(tid)
    n, factor = p["n"], p["factor"]

    out = ["%d %d %.8f" % (n, tid, factor)]
    for t in range(n):
        r = rounded_reported(t, p, tid, noise=True)
        out.append("%d %d" % (t, int(r)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
