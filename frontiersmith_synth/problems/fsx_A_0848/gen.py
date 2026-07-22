#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE derelict-airlock chatter log to stdout.

Deterministic: all randomness is seeded from testId only.

The hidden wiring (which of the four raw relay codes A/B/C/D plays the
INC / DEC / FLIP / NOOP role, the shared nudge magnitude, and the starting
polarity), plus the held-out (much longer, black-box) chatter bursts, are
NEVER printed here -- only the logged (burst, drift-reading) training rows.
verify.py independently re-derives the identical hidden wiring from testId
via the byte-identical derive_params() below (kept in sync by hand; not
imported from a shared module, so a sandboxed solution can never read the
ground truth)."""
import sys
import random

ALPHABET = "ABCD"
BASELINE = dict(inc="A", dec="B", flip="C", noop="D", u=1, p0=1)

U_MIN, U_MAX = 2, 9
L_TRAIN_MIN, L_TRAIN_MAX = 8, 20
NOISE_PROB = 1.0 / 6.0
NOISE_CHOICES = [-3, -2, -1, 1, 2, 3]


def simulate(s, inc, dec, flip, noop, u, p0):
    c = 0
    p = p0
    for ch in s:
        if ch == inc:
            c += p * u
        elif ch == dec:
            c -= p * u
        elif ch == flip:
            p = -p
        # noop: no effect
    return c


def derive_params(test_id):
    rng = random.Random(84800 + test_id)
    while True:
        roles = list(ALPHABET)
        rng.shuffle(roles)
        inc, dec, flip, noop = roles
        u = rng.randint(U_MIN, U_MAX)
        p0 = rng.choice([-1, 1])
        cand = dict(inc=inc, dec=dec, flip=flip, noop=noop, u=u, p0=p0)
        if cand != BASELINE:
            break

    K = 70 + 4 * test_id  # 74..110 logged bursts

    train_rows = []
    for _ in range(K):
        L = rng.randint(L_TRAIN_MIN, L_TRAIN_MAX)
        nflip = rng.choices([0, 1, 2], weights=[70, 25, 5])[0]
        nflip = min(nflip, L)
        flip_positions = set(rng.sample(range(L), nflip))
        chars = []
        for pos in range(L):
            if pos in flip_positions:
                chars.append(flip)
            else:
                chars.append(rng.choice([inc, dec, noop]))
        s = "".join(chars)
        y_true = simulate(s, inc, dec, flip, noop, u, p0)
        y_log = y_true
        if rng.random() < NOISE_PROB:
            y_log += rng.choice(NOISE_CHOICES)
        train_rows.append((s, y_log))

    # held-out extrapolation bursts (long, never shown): mix of typical
    # random shuffles and adversarial role-sorted (clustered) arrangements
    # that share the same role-multiset as a typical row but reorder it to
    # maximize the gap between an order-blind (count-only) predictor and
    # the true, order-sensitive counter reading.
    M = 10
    held = []
    for k in range(M):
        L = 100 + 30 * (test_id - 1) + 12 * k
        role_seq = [rng.choice(["inc", "dec", "flip", "noop"]) for _ in range(L)]
        sym = dict(inc=inc, dec=dec, flip=flip, noop=noop)
        if k % 2 == 1:
            order = {"inc": 0, "flip": 1, "dec": 2, "noop": 3}
            role_seq = sorted(role_seq, key=lambda r: order[r])
        s = "".join(sym[r] for r in role_seq)
        held.append(s)

    return dict(inc=inc, dec=dec, flip=flip, noop=noop, u=u, p0=p0, K=K,
                train_rows=train_rows, M=M, held=held)


def main():
    test_id = int(sys.argv[1])
    p = derive_params(test_id)
    lines = [f"{test_id} {p['K']}"]
    for s, y in p["train_rows"]:
        lines.append(f"{s} {y}")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
