#!/usr/bin/env python3
"""verify.py <in> <out> <ans>  -- deterministic checker for the Scrip-Ward market problem.

Reads the instance (N, R, ALPHA_NUM, ALPHA_DEN, S, TAX_DEN, v[r][i]) from <in>, the
submitted policy (endowments E, tax rates T, refill weights W) from <out>, validates
feasibility strictly, replays the market for R rounds, and prints the normalized score
as the last line: "... Ratio: <float in [0,1]>".
"""
import sys

W_MAX = 1_000_000


def fail(reason):
    print("INVALID: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    with open(path, "r") as f:
        toks = f.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        v = toks[pos]
        pos += 1
        return v

    N = int(nxt()); R = int(nxt())
    ALPHA_NUM = int(nxt()); ALPHA_DEN = int(nxt()); S = int(nxt()); TAX_DEN = int(nxt())
    v = []
    for _r in range(R):
        row = [int(nxt()) for _ in range(N)]
        v.append(row)
    return N, R, ALPHA_NUM, ALPHA_DEN, S, TAX_DEN, v


def read_policy_tokens(path, N, R):
    try:
        with open(path, "r") as f:
            raw = f.read().split()
    except Exception as e:
        fail("cannot read output: %s" % e)

    expected = N + R + N * R
    if len(raw) != expected:
        fail("wrong token count: got %d, expected %d (N + R + N*R)" % (len(raw), expected))

    try:
        toks = [int(t) for t in raw]
    except ValueError as e:
        fail("non-integer token in output: %s" % e)

    pos = 0
    E = toks[pos:pos + N]; pos += N
    T = toks[pos:pos + R]; pos += R
    W = []
    for _r in range(R):
        W.append(toks[pos:pos + N]); pos += N
    return E, T, W


def validate_policy(N, R, S, TAX_DEN, E, T, W):
    if any(e < 0 for e in E):
        fail("negative endowment")
    if sum(E) != S:
        fail("endowments must sum to exactly S=%d, got %d" % (S, sum(E)))
    for r in range(R):
        if not (0 <= T[r] <= TAX_DEN):
            fail("tax rate T[%d]=%d out of range [0,%d]" % (r, T[r], TAX_DEN))
    for r in range(R):
        row = W[r]
        for w in row:
            if not (0 <= w <= W_MAX):
                fail("refill weight out of range [0,%d] at round %d" % (W_MAX, r))


def largest_remainder(weights, total):
    n = len(weights)
    sw = sum(weights)
    if sw == 0:
        base = total // n
        rem = total - base * n
        out = [base] * n
        for i in range(rem):
            out[i] += 1
        return out
    alloc = [total * w // sw for w in weights]
    rem = total - sum(alloc)
    order = sorted(range(n), key=lambda i: (-weights[i], i))
    for k in range(rem):
        alloc[order[k]] += 1
    return alloc


def simulate(N, R, v, ALPHA_NUM, ALPHA_DEN, TAX_DEN, E, T, W):
    wallet = list(E)
    total_relief = 0
    for r in range(R):
        bids = [min(v[r][i] * ALPHA_NUM // ALPHA_DEN, wallet[i]) for i in range(N)]
        mb = max(bids)
        if mb > 0:
            winner = bids.index(mb)  # first index attains max -> smallest-index tie-break
            total_relief += v[r][winner]
            bid = bids[winner]
            wallet[winner] -= bid
            others = [i for i in range(N) if i != winner]
            share = bid // (N - 1)
            rem = bid % (N - 1)
            for idx, i in enumerate(others):
                wallet[i] += share + (1 if idx < rem else 0)
        tax = [wallet[i] * T[r] // TAX_DEN for i in range(N)]
        pool = sum(tax)
        for i in range(N):
            wallet[i] -= tax[i]
        if pool > 0:
            alloc = largest_remainder(W[r], pool)
            for i in range(N):
                wallet[i] += alloc[i]
    return total_relief


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    N, R, ALPHA_NUM, ALPHA_DEN, S, TAX_DEN, v = read_instance(in_path)
    E, T, W = read_policy_tokens(out_path, N, R)
    validate_policy(N, R, S, TAX_DEN, E, T, W)

    F = simulate(N, R, v, ALPHA_NUM, ALPHA_DEN, TAX_DEN, E, T, W)

    # internal baseline: equal-split endowment, no tax, no refill
    E0 = largest_remainder([1] * N, S)
    T0 = [0] * R
    W0 = [[0] * N for _ in range(R)]
    B = simulate(N, R, v, ALPHA_NUM, ALPHA_DEN, TAX_DEN, E0, T0, W0)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
