#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for the lottery-machine-forensics task.

The instance id is NOT stored in <in> (only the revealed draws and the query indices are --
see gen.py). The checker re-identifies which of the N_CASES canonical instances it is looking
at by regenerating each candidate's hidden law and matching its revealed sequence against the
one actually given (so nothing beyond the puzzle's own data can be used to identify/shortcut
the instance), then scores the participant's Q predicted draws by the fraction of the 4 secret
primes (p_1..p_4, m = p_1*p_2*p_3*p_4) each guess agrees with the true draw modulo, averaged
over the Q queries, then rescaled to leave headroom above a perfect reconstruction.
"""
import sys
import re
import random

# ---- identical hidden-law generator to gen.py (duplicated on purpose: checkers must
#      regenerate ground truth independently, not import a groundtruth module) ----


def _sieve(limit):
    isc = [True] * (limit + 1)
    isc[0] = isc[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if isc[i]:
            for j in range(i * i, limit + 1, i):
                isc[j] = False
    return [i for i in range(2, limit + 1) if isc[i]]


_PRIMES = _sieve(4999)
_BANDS = [(101, 300), (300, 800), (800, 1800), (1800, 3200), (3200, 4999)]
N_REVEAL = 120
K_PRIMES = 4
_ANCHORS = [130, 900, 15000, 2_000_000, 900_000_000,
            400_000_000_000, 200_000_000_000_000, 900_000_000_000_000_000]
N_CASES = 10  # must match config.yaml subtasks[0].n_cases


def _crt(residues, moduli):
    x, M = 0, 1
    for r, mod in zip(residues, moduli):
        r = r % mod
        t = ((r - x) * pow(M, -1, mod)) % mod
        x = x + M * t
        M *= mod
    return x % M


def make_secret(t):
    rnd = random.Random(1_000_003 * t + 7)
    band = _BANDS[(t - 1) // 2]
    cand = [p for p in _PRIMES if band[0] <= p <= band[1]]
    primes = sorted(rnd.sample(cand, K_PRIMES))
    trap = (t % 2 == 0)
    trap_idx = rnd.randrange(K_PRIMES) if trap else -1
    a_list, c_list, x0_list = [], [], []
    for idx, p in enumerate(primes):
        if idx == trap_idx:
            a_list.append(1)
            c_list.append(0)
        else:
            a_list.append(rnd.randrange(2, p - 1))
            c_list.append(rnd.randrange(0, p))
        x0_list.append(rnd.randrange(0, p))
    m = 1
    for p in primes:
        m *= p
    a = _crt(a_list, primes)
    c = _crt(c_list, primes)
    x0 = _crt(x0_list, primes)
    ks = []
    for anc in _ANCHORS:
        j = rnd.randint(0, max(1, anc // 25))
        ks.append(anc + j)
    ks = sorted(set(ks))
    return primes, m, a, c, x0, trap_idx, ks


def gen_sequence(m, a, c, x0, n):
    xs = [x0 % m]
    for _ in range(n - 1):
        xs.append((a * xs[-1] + c) % m)
    return xs


def fast_forward(a, c, m, x0, k):
    """x_k for k>=1 (x_1=x0), via binary exponentiation of the affine monoid (a,c) mod m."""
    e = k - 1
    res_a, res_c = 1, 0
    cur_a, cur_c = a % m, c % m
    while e > 0:
        if e & 1:
            res_a, res_c = (cur_a * res_a) % m, (cur_a * res_c + cur_c) % m
        cur_a, cur_c = (cur_a * cur_a) % m, (cur_a * cur_c + cur_c) % m
        e >>= 1
    return (res_a * x0 + res_c) % m


TOKEN_RE = re.compile(r"^\d{1,30}$")


def fail(reason):
    print("infeasible: %s -- Ratio: 0.0" % reason)
    sys.exit(0)


def main():
    inf, outf = sys.argv[1], sys.argv[2]
    in_lines = open(inf).read().split("\n")
    n = int(in_lines[0].strip())
    revealed_xs = list(map(int, in_lines[1].split()))
    q = int(in_lines[2].strip())
    query_ks = list(map(int, in_lines[3].split()))
    if len(revealed_xs) != n or len(query_ks) != q:
        fail("malformed input")

    # identify which of the N_CASES canonical instances this is by matching the
    # regenerated revealed sequence + query list against what <in> actually contains.
    match = None
    for cand in range(1, N_CASES + 1):
        primes, m, a, c, x0, trap_idx, ks_expected = make_secret(cand)
        if ks_expected == query_ks and gen_sequence(m, a, c, x0, n) == revealed_xs:
            match = (primes, m, a, c, x0)
            break
    if match is None:
        fail("input does not match any regenerated instance")
    primes, m, a, c, x0 = match

    try:
        out_text = open(outf).read()
    except Exception:
        fail("cannot read output")
    tokens = out_text.split()
    if len(tokens) != q:
        fail("expected %d tokens, got %d" % (q, len(tokens)))
    guesses = []
    for tok in tokens:
        if not TOKEN_RE.match(tok):
            fail("non-numeric or malformed token %r" % tok)
        guesses.append(int(tok))

    total_credit = 0.0
    for k, g in zip(query_ks, guesses):
        true_val = fast_forward(a, c, m, x0, k)
        matches = sum(1 for p in primes if (g - true_val) % p == 0)
        total_credit += matches / K_PRIMES
    f_score = total_credit / q

    ratio = 0.10 + 0.80 * f_score
    ratio = max(0.0, min(1.0, ratio))
    print("f_score=%.6f -- Ratio: %.6f" % (f_score, ratio))
    sys.exit(0)


if __name__ == "__main__":
    main()
