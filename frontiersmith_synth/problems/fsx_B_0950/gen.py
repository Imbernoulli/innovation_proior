#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN ledger to stdout.

Rug-bazaar haggling rule (hidden). Each testId fixes a DIFFERENT bazaar with
its own competition-shading strength `rho` and appraisal-risk-surcharge
strength `kappa`, held around a FIXED public benchmark price MU = 100.0.

True settled price (never printed; lives here AND in verify.py, never
imported so nothing is importable from this directory):

    price(n, v) = v * (n-1) / ((n-1) + rho)          # shading -> 0 as n -> inf
                + kappa * (v - MU)**2 / MU            # appraisal-risk surcharge

Training rows are logged ONLY on quiet days: n in {2,3,4,5,6}, v drawn from a
moderate range. The held-out grading ledger (regenerated only inside
verify.py) covers much larger n and much more extreme v -- genuine
extrapolation on both axes, never shown here.

STDOUT prints ONLY: header "<num_rows> <testId> <mu>" then rows "n v price".
rho, kappa and the RNG seed are never printed.
"""
import sys, random

MU = 100.0
N_LIST = (2, 3, 4, 5, 6)


def hidden_params(t):
    """Hidden bazaar law for this test id (duplicated verbatim in verify.py)."""
    rng = random.Random(7000003 + t * 104729)
    rho = rng.uniform(0.3, 3.0)
    kappa = rng.uniform(0.01, 0.10)
    return rho, kappa


def true_price(n, v, rho, kappa):
    shade = v * (n - 1) / ((n - 1) + rho)
    risk = kappa * (v - MU) ** 2 / MU
    return shade + risk


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rho, kappa = hidden_params(t)

    reps = max(5, 12 - t)               # fewer repeats per n at harder test ids
    sigma_train = 1.5 + 0.35 * t        # noisier ledger at harder test ids

    rng = random.Random(31337 + t * 7919)
    rows = []
    for n in N_LIST:
        for _ in range(reps):
            v = rng.uniform(20.0, 200.0)
            price = true_price(n, v, rho, kappa) + rng.gauss(0.0, sigma_train)
            price = max(0.01, price)
            rows.append((n, v, price))

    out = ["%d %d %.6f" % (len(rows), t, MU)]
    for n, v, price in rows:
        out.append("%d %.6f %.6f" % (n, v, price))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
