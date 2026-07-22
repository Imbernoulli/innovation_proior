#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE abacus-cult tribute ledger to stdout.

Deterministic: all randomness is seeded from testId only.

The cult's secret radix b and its quadratic blessing coefficients
(a2, a1, a0), plus the held-out (astronomically large) tributes, are NEVER
printed here -- only the logged (tribute, blessing) rows for moderate
tributes. verify.py independently re-derives the identical hidden law from
testId via the byte-identical derive_params() below (kept in sync by hand;
not imported from a shared module, so a sandboxed solution can never read
the ground truth)."""
import sys
import random

B_MIN, B_MAX = 3, 40
X_MAX = 10 ** 6


def digitsum_base(x, b):
    s = 0
    while x > 0:
        s += x % b
        x //= b
    return s


def make_bignum(rng, ndigits):
    """A deterministic random positive integer with exactly ndigits decimal digits."""
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

    K = 50 + 4 * test_id  # 54..90 logged tributes
    train_x = [rng.randint(1, X_MAX) for _ in range(K)]

    D_base = 5 + (3 * test_id) // 2  # held-out tributes grow beyond training scale
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


def main():
    test_id = int(sys.argv[1])
    p = derive_params(test_id)
    lines = [f"{test_id} {p['K']}"]
    for x in p["train_x"]:
        y = compute_y(x, p["b"], p["a2"], p["a1"], p["a0"])
        lines.append(f"{x} {y}")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
