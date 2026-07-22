#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE ledger instance to stdout.
Deterministic: all randomness is seeded from testId only.
The hidden rule (p, q, fee, theta) is NEVER printed -- only the daily
balances (data rows). verify.py independently re-derives the same hidden
rule from testId via the identical derive_params() below (kept in sync by
hand; not imported from a shared module)."""
import sys
import random
from math import gcd

L = 30          # month length (public constant, also printed in the header)
T_TRAIN = 200   # training days logged to the solver
K_BASE = 4      # accounts: K_BASE..K_BASE+2


def derive_params(test_id):
    rng = random.Random(10000 + test_id)
    if test_id <= 3:
        q = rng.randint(40, 80)
    elif test_id <= 6:
        q = rng.randint(80, 150)
    else:
        q = rng.randint(150, 300)
    r_target = rng.uniform(0.008, 0.035)
    p = max(1, round(r_target * q))
    while gcd(p, q) != 1:
        p += 1
    S = rng.randint(300, 900)
    theta = S + rng.randint(-100, 100)
    fee = rng.randint(max(2, int(0.01 * S)), max(3, int(0.04 * S)))
    K = K_BASE + (test_id % 3)
    train_b0 = [rng.randint(int(0.6 * S), int(1.6 * S)) for _ in range(K)]
    M = 4
    held_b0 = [rng.randint(int(0.3 * S), int(2.5 * S)) for _ in range(M)]
    return dict(p=p, q=q, S=S, theta=theta, fee=fee, K=K,
                train_b0=train_b0, M=M, held_b0=held_b0)


def simulate(b0_list, p, q, fee, theta, ndays):
    """Roll the ledger rule forward ndays days from each starting balance."""
    out = []
    for b0 in b0_list:
        b = b0
        month_start = b0
        bal = [b0]
        for t in range(ndays):
            if t % L == 0:
                month_start = b
            grown = (b * (q + p)) // q
            if t % L == L - 1:
                nb = grown - fee if month_start < theta else grown
            else:
                nb = grown
            if nb < 0:
                nb = 0
            bal.append(nb)
            b = nb
        out.append(bal)
    return out


def main():
    test_id = int(sys.argv[1])
    params = derive_params(test_id)
    train_traj = simulate(params["train_b0"], params["p"], params["q"],
                           params["fee"], params["theta"], T_TRAIN)
    K = params["K"]
    out = []
    out.append(f"{test_id} {K} {T_TRAIN} {L}")
    for bal in train_traj:
        out.append(" ".join(str(x) for x in bal))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
