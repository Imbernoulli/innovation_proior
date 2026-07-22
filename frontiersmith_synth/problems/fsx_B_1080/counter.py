#!/usr/bin/env python3
"""counter.py <in> <out> <ans> -- deterministic scorer for Fixed-ROM Codeword Binding.
Prints exactly one final line: '... Ratio: <float>'.
"""
import sys


def fail(msg):
    print(msg)
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    with open(path, "r") as fh:
        toks = fh.read().split()
    if len(toks) < 1:
        raise ValueError("empty instance")
    pos = 0
    n = int(toks[pos]); pos += 1
    f = [int(toks[pos + i]) for i in range(n)]; pos += n
    L = [int(toks[pos + i]) for i in range(n)]; pos += n
    return n, f, L


def main():
    if len(sys.argv) < 3:
        fail("bad invocation")
    in_path, out_path = sys.argv[1], sys.argv[2]

    n, f, L = read_instance(in_path)

    try:
        with open(out_path, "r") as fh:
            raw_toks = fh.read().split()
    except Exception as e:
        fail(f"cannot read output: {e}")

    if len(raw_toks) != n:
        fail(f"expected {n} tokens, got {len(raw_toks)}")

    MAX_TOKEN_LEN = 32
    d = []
    for tok in raw_toks:
        if len(tok) > MAX_TOKEN_LEN:
            fail("token too long")
        try:
            v = int(tok)
        except ValueError:
            fail(f"non-integer token: {tok!r}")
        d.append(v)

    # feasibility: the submitted multiset of slot lengths must equal the
    # instance's fixed slot multiset exactly (every ROM slot used once).
    if sorted(d) != sorted(L):
        fail("submitted lengths are not a permutation of the fixed ROM slot multiset")

    # (redundant safety net; sorted(d)==sorted(L) already forces this since
    # every L_i >= 1 by construction, but keep an explicit finiteness/range
    # guard in case a future generator relaxes that invariant)
    for v in d:
        if v != v or v in (float("inf"), float("-inf")) or v < 1:
            fail("non-finite or non-positive length")

    F = sum(fi * di for fi, di in zip(f, d))

    # checker's own baseline: "important symbols deserve the biggest slot" --
    # bind the highest-frequency symbol to the LONGEST codeword, and so on.
    # By the rearrangement inequality this is the *worst* possible binding
    # for the fixed slot multiset, so it is a safe, generous normalizer.
    order = sorted(range(n), key=lambda i: (-f[i], i))
    slots_desc = sorted(L, reverse=True)
    b_d = [0] * n
    for rank, sym in enumerate(order):
        b_d[sym] = slots_desc[rank]
    B = sum(fi * di for fi, di in zip(f, b_d))

    if F <= 0:
        fail("non-positive objective")

    ratio = min(1.0, 0.1 * B / F)
    print(f"F={F} B={B}")
    print("Ratio: %.6f" % ratio)
    sys.exit(0)


if __name__ == "__main__":
    main()
