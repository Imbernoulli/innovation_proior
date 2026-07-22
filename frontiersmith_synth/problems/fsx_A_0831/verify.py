#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for the Abacus Cult's
Secret Radix problem.

Prints "... Ratio: <float in [0,1]>" on its last line and exits 0.

Re-derives the SAME hidden (b, a2, a1, a0) and the SAME held-out
(astronomically large) tributes from the test id (found on line 1 of <in>)
via derive_params(), which is kept byte-identical to gen.py's by hand (no
shared/importable module, so a sandboxed solution can never read the ground
truth). The held-out tributes themselves are never written to <in> or
shown anywhere -- only used internally here to score."""
import sys
import math
import random

B_MIN, B_MAX = 3, 40
X_MAX = 10 ** 6

B_SUB_MIN, B_SUB_MAX = 3, 40
A2_BOUND = 100
A1_BOUND = 1000
A0_BOUND = 100_000


def digitsum_base(x, b):
    s = 0
    while x > 0:
        s += x % b
        x //= b
    return s


def make_bignum(rng, ndigits):
    first = rng.randint(1, 9)
    rest = [str(rng.randint(0, 9)) for _ in range(ndigits - 1)]
    return int(str(first) + "".join(rest))


def derive_params(test_id):
    rng = random.Random(51000 + test_id)
    while True:
        b = rng.randint(B_MIN, B_MAX)
        if b != 10:
            break
    sign2 = rng.choice([-1, 1])
    a2 = sign2 * rng.randint(1, 5)
    a1 = rng.randint(-20, 20)
    a0 = rng.randint(-500, 500)

    K = 50 + 4 * test_id
    train_x = [rng.randint(1, X_MAX) for _ in range(K)]

    D_base = 5 + (3 * test_id) // 2
    M = 6
    held_x = []
    for _ in range(M):
        D = max(6, D_base + rng.randint(-2, 2))
        held_x.append(make_bignum(rng, D))

    return dict(b=b, a2=a2, a1=a1, a0=a0, K=K, train_x=train_x,
                M=M, held_x=held_x)


def compute_y(x, b, a2, a1, a0):
    s = digitsum_base(x, b)
    return a2 * s * s + a1 * s + a0


def simulate(xs, b, a2, a1, a0):
    return [compute_y(x, b, a2, a1, a0) for x in xs]


def mean_abs_err(a, b):
    tot = 0
    n = 0
    for u, v in zip(a, b):
        tot += abs(u - v)
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
    if len(header) != 2:
        fail("malformed input header")
    try:
        test_id = int(header[0])
    except ValueError:
        fail("bad test id")

    try:
        with open(out_path) as f:
            toks = f.read().split()
    except FileNotFoundError:
        fail("no output file")

    if len(toks) != 4:
        fail("expected exactly 4 tokens (b a2 a1 a0), got %d" % len(toks))

    parsed = []
    for tok in toks:
        try:
            v = int(tok)
        except ValueError:
            fail("token %r is not a plain integer" % tok)
        if not math.isfinite(v):
            fail("non-finite token")
        parsed.append(v)
    b_sub, a2_sub, a1_sub, a0_sub = parsed

    if b_sub < B_SUB_MIN or b_sub > B_SUB_MAX:
        fail("radix b out of range [%d,%d]" % (B_SUB_MIN, B_SUB_MAX))
    if abs(a2_sub) > A2_BOUND:
        fail("a2 out of range")
    if abs(a1_sub) > A1_BOUND:
        fail("a1 out of range")
    if abs(a0_sub) > A0_BOUND:
        fail("a0 out of range")

    params = derive_params(test_id)
    held_x = params["held_x"]

    held_true = simulate(held_x, params["b"], params["a2"], params["a1"],
                          params["a0"])
    held_pred = simulate(held_x, b_sub, a2_sub, a1_sub, a0_sub)
    held_flat = simulate(held_x, 3, 0, 0, 0)

    F = mean_abs_err(held_true, held_pred)
    B = mean_abs_err(held_true, held_flat)
    eps = B / 8.0 if B > 0 else 1.0
    sc = min(1000.0, 100.0 * (B + eps) / max(1e-9, F + eps))
    ratio = sc / 1000.0

    print("abacus-radix F=%.6f B=%.6f Ratio: %.6f" % (F, B, ratio))
    sys.exit(0)


if __name__ == "__main__":
    main()
