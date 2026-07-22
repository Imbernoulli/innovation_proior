#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for the Guild Ledger problem.
Prints "... Ratio: <float in [0,1]>" on its last line and exits 0.

Re-derives the SAME hidden (p, q, fee, theta) and held-out starting balances
from the test id (found on line 1 of <in>) via derive_params(), which is
kept byte-identical to gen.py's by hand (no shared/importable module, so a
sandboxed solution can never read the ground truth)."""
import sys
import math
import random
from math import gcd

L = 30
T_TRAIN = 200
HELD_DAYS = 300
K_BASE = 4


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


def mae(a, b):
    tot = 0
    n = 0
    for ba, bb in zip(a, b):
        for x, y in zip(ba, bb):
            tot += abs(x - y)
            n += 1
    return tot / n if n else 0.0


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


def main():
    if len(sys.argv) < 3:
        fail("bad invocation")
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path) as f:
        header = f.readline().split()
    if len(header) != 4:
        fail("malformed input header")
    test_id = int(header[0])

    try:
        with open(out_path) as f:
            toks = f.read().split()
    except FileNotFoundError:
        fail("no output file")

    if len(toks) != 4:
        fail("expected exactly 4 tokens (p q fee theta), got %d" % len(toks))

    parsed = []
    for tok in toks:
        try:
            v = int(tok)
        except ValueError:
            fail("token %r is not an integer" % tok)
        if not math.isfinite(v):
            fail("non-finite token")
        parsed.append(v)
    p, q, fee, theta = parsed

    if q < 1 or q > 10_000_000:
        fail("q out of range")
    if p < 0 or p >= q:
        fail("p out of range (need 0 <= p < q)")
    if fee < 0 or fee > 10_000_000:
        fail("fee out of range")
    if theta < 0 or theta > 1_000_000_000:
        fail("theta out of range")

    params = derive_params(test_id)
    held_b0 = params["held_b0"]

    held_true = simulate(held_b0, params["p"], params["q"], params["fee"],
                          params["theta"], HELD_DAYS)
    held_pred = simulate(held_b0, p, q, fee, theta, HELD_DAYS)
    held_flat = simulate(held_b0, 0, 1, 0, 0, HELD_DAYS)

    F = mae(held_true, held_pred)
    B = mae(held_true, held_flat)
    eps = B / 8.0
    sc = min(1000.0, 100.0 * (B + eps) / max(1e-9, F + eps))
    ratio = sc / 1000.0

    print("guild-ledger F=%.6f B=%.6f Ratio: %.6f" % (F, B, ratio))
    sys.exit(0)


if __name__ == "__main__":
    main()
