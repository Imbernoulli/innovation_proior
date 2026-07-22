#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for "Airlock Splice",
the derelict-ship counter-wiring recovery problem.

Prints "... Ratio: <float in [0,1]>" on its last line and exits 0.

Re-derives the SAME hidden wiring (role assignment of A/B/C/D to
INC/DEC/FLIP/NOOP, the shared nudge magnitude, the starting polarity) and
the SAME held-out (much longer) chatter bursts from the test id (found on
line 1 of <in>) via derive_params(), which is kept byte-identical to
gen.py's by hand (no shared/importable module, so a sandboxed solution can
never read the ground truth). The held-out bursts themselves are never
written to <in> or shown anywhere -- only used internally here to score."""
import sys
import math
import random

ALPHABET = "ABCD"
BASELINE = dict(inc="A", dec="B", flip="C", noop="D", u=1, p0=1)

U_MIN, U_MAX = 2, 9
L_TRAIN_MIN, L_TRAIN_MAX = 8, 20
NOISE_PROB = 1.0 / 6.0
NOISE_CHOICES = [-3, -2, -1, 1, 2, 3]

U_SUB_MIN, U_SUB_MAX = 1, 1000


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

    K = 70 + 4 * test_id

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

    if len(toks) != 6:
        fail("expected exactly 6 tokens (inc dec flip noop u p0), got %d"
             % len(toks))

    inc_sub, dec_sub, flip_sub, noop_sub = toks[0], toks[1], toks[2], toks[3]
    for tok in (inc_sub, dec_sub, flip_sub, noop_sub):
        if len(tok) != 1 or tok not in ALPHABET:
            fail("role symbols must each be one of %s" % ALPHABET)
    if len({inc_sub, dec_sub, flip_sub, noop_sub}) != 4:
        fail("role symbols must be a permutation of %s (all distinct)"
             % ALPHABET)

    int_toks = []
    for tok in toks[4:]:
        try:
            v = int(tok)
        except ValueError:
            fail("token %r is not a plain integer" % tok)
        if not math.isfinite(v):
            fail("non-finite token")
        int_toks.append(v)
    u_sub, p0_sub = int_toks

    if u_sub < U_SUB_MIN or u_sub > U_SUB_MAX:
        fail("u out of range [%d,%d]" % (U_SUB_MIN, U_SUB_MAX))
    if p0_sub not in (-1, 1):
        fail("p0 must be -1 or 1")

    params = derive_params(test_id)
    held = params["held"]

    held_true = [simulate(s, params["inc"], params["dec"], params["flip"],
                           params["noop"], params["u"], params["p0"])
                 for s in held]
    held_pred = [simulate(s, inc_sub, dec_sub, flip_sub, noop_sub, u_sub,
                           p0_sub) for s in held]
    held_base = [simulate(s, BASELINE["inc"], BASELINE["dec"],
                           BASELINE["flip"], BASELINE["noop"],
                           BASELINE["u"], BASELINE["p0"]) for s in held]

    F = mean_abs_err(held_true, held_pred)
    B = mean_abs_err(held_true, held_base)
    eps = B / 8.0 if B > 0 else 1.0
    sc = min(1000.0, 100.0 * (B + eps) / max(1e-9, F + eps))
    ratio = sc / 1000.0

    print("airlock-splice F=%.6f B=%.6f Ratio: %.6f" % (F, B, ratio))
    sys.exit(0)


if __name__ == "__main__":
    main()
