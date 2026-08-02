#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- checker for polymer-sequence-design.

Scores a submitted monomer sequence against a target glass-transition
temperature computed from the *bond* (adjacent-pair / dyad) structure of the
sequence, not from its average composition. Prints the LAST line as
`... Ratio: <float in [0,1]>` per the harness contract.
"""
import sys, math

SIGMA = 10.0  # Tg-closeness falloff scale (Kelvin-like units)


def fail(msg):
    print(msg)
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    N = int(next(it)); K = int(next(it))
    tg = [int(next(it)) for _ in range(K)]
    M = [[int(next(it)) for _ in range(K)] for _ in range(K)]
    caps = [int(next(it)) for _ in range(K)]
    target = int(next(it))
    return N, K, tg, M, caps, target


def tg_pred(seq, M):
    n = len(seq)
    inv_sum = 0.0
    for k in range(n - 1):
        a, b = seq[k] - 1, seq[k + 1] - 1
        m = M[a][b]
        if m <= 0:
            return None
        inv_sum += 1.0 / m
    if inv_sum <= 0:
        return None
    return (n - 1) / inv_sum


def score_of(seq, M, target):
    tp = tg_pred(seq, M)
    if tp is None:
        return 0.0
    err = abs(tp - target)
    return math.exp(-(err * err) / (2.0 * SIGMA * SIGMA))


def baseline_sequence(N, K, caps, tg, target):
    """The checker's OWN trivial feasible construction: fill monomer types,
    in blocks, in order of how POORLY their pure Tg matches the target
    (worst-matching type first), up to each cap, until length N is reached.
    This never looks at the interaction matrix and never tries to balance
    composition against the target -- a stable, uninformed reference."""
    order = sorted(range(K), key=lambda i: -abs(tg[i] - target))
    seq = []
    for t in order:
        if len(seq) >= N:
            break
        take = min(caps[t], N - len(seq))
        seq.extend([t + 1] * take)
    return seq


def main():
    if len(sys.argv) < 3:
        fail("bad args")
    in_path, out_path = sys.argv[1], sys.argv[2]
    N, K, tg, M, caps, target = read_instance(in_path)

    with open(out_path) as f:
        raw = f.read().split()

    if len(raw) != N:
        fail(f"expected {N} tokens, got {len(raw)}")

    seq = []
    for tok in raw:
        try:
            v = int(tok)
        except ValueError:
            fail(f"non-integer token: {tok!r}")
        if v != v or math.isinf(v):
            fail("non-finite token")
        if v < 1 or v > K:
            fail(f"token {v} out of range [1,{K}]")
        seq.append(v)

    counts = [0] * K
    for v in seq:
        counts[v - 1] += 1
    for i in range(K):
        if counts[i] > caps[i]:
            fail(f"feedstock cap exceeded for type {i+1}: used {counts[i]} > cap {caps[i]}")

    F = score_of(seq, M, target)
    if F is None:
        fail("invalid dyad evaluation")

    base_seq = baseline_sequence(N, K, caps, tg, target)
    B = score_of(base_seq, M, target)
    B = max(1e-9, B)

    sc = min(1000.0, 100.0 * F / B)
    ratio = sc / 1000.0
    print(f"quality(F)={F:.6f} baseline(B)={B:.6f} Ratio: {ratio:.6f}")


if __name__ == "__main__":
    main()
