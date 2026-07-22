#!/usr/bin/env python3
"""
gen.py <testId> -- march-64-automata bitsliced transition synthesis instance.

Plants a hidden GF(2)-affine transition rule under a HIDDEN state relabeling
(a random permutation pi of the S states onto b-bit codes, b = log2(S)).
Under pi, the transition is exactly

    code(next_state) = M . code(state)  xor  N . bits(symbol)  xor  c     (mod 2)

Publicly the FSM is only given in its natural 0..S-1 numbering (delta table),
so pi is invisible to the solver -- it must be (re)discovered by searching over
state-encodings. Harder cases additionally splice in a handful of point
perturbations so no single relabeling makes the WHOLE table gate-free (regime
change) -- so pi is a strong, but not perfect, encoding, and neither is any
other single relabeling.

An ACCEPTANCE CHECK (below) rejects any draw where the identity numbering
would already be about as cheap to compile as the planted encoding -- this
guarantees the "obvious" identity-numbering approach is genuinely punished on
every emitted instance, not just on average.

Deterministic: all randomness (including the retry loop) is seeded purely
from testId.
"""
import sys, random

# ladder: (S, K, n_perturb) by testId (1..10). n_perturb points are respliced
# after the affine plant so no encoding linearizes the WHOLE table (this also
# keeps the achievable gate-count away from its information-theoretic floor,
# which is needed so the score does not saturate).
LADDER = {
    1: (4, 4, 4),
    2: (4, 4, 4),
    3: (8, 2, 4),
    4: (8, 2, 4),
    5: (8, 4, 4),
    6: (8, 4, 4),
    7: (4, 4, 6),
    8: (8, 2, 6),
    9: (8, 4, 6),
    10: (8, 4, 8),
}

LANES = 64
MIN_GAP = 4          # required (identity ANF-cost) - (planted-encoding ANF-cost)
MAX_ATTEMPTS = 200


def bits_of(x, n):
    return [(x >> i) & 1 for i in range(n)]


def int_of(bits):
    v = 0
    for i, b in enumerate(bits):
        v |= (b & 1) << i
    return v


def matvec(mat, vec):
    # mat: rows list of column-bit-lists (row-major, mat[r][c]); vec: list of bits
    out = []
    for row in mat:
        acc = 0
        for c, mc in enumerate(row):
            acc ^= (mc & vec[c])
        out.append(acc)
    return out


def gen_matrix(rng, rows, cols):
    while True:
        m = [[rng.randint(0, 1) for _ in range(cols)] for _ in range(rows)]
        if any(any(r) for r in m):
            return m


# ---- same ANF-cost model used by solutions/greedy.py & strong.py, so the
# acceptance check measures exactly what a solver's circuit compiler would pay ----

def anf_transform(truth, n):
    a = truth[:]
    size = 1 << n
    for i in range(n):
        bit = 1 << i
        for x in range(size):
            if x & bit:
                a[x] ^= a[x ^ bit]
    return a


def bit_cost(anf, n):
    monomials = [mask for mask in range(1, 1 << n) if anf[mask]]
    cost = 0
    for mask in monomials:
        cost += max(0, bin(mask).count("1") - 1)
    if not monomials:
        cost += 2
    else:
        cost += max(0, len(monomials) - 1)
        if anf[0] == 1:
            cost += 1
    return cost


def encoding_cost(code_of, S, K, b, m, delta):
    n = b + m
    Y = [0] * (1 << n)
    for s in range(S):
        cs = code_of[s]
        for k in range(K):
            Y[cs | (k << b)] = code_of[delta[s][k]]
    tot = 0
    for j in range(b):
        truth = [(Y[x] >> j) & 1 for x in range(1 << n)]
        tot += bit_cost(anf_transform(truth, n), n)
    return tot


def try_build(rng, S, K, b, m, n_perturb):
    # hidden permutation state -> code (must not be identity)
    while True:
        pi = list(range(S))
        rng.shuffle(pi)
        if pi != list(range(S)):
            break
    inv_pi = [0] * S
    for s, code in enumerate(pi):
        inv_pi[code] = s

    M = gen_matrix(rng, b, b)
    N = gen_matrix(rng, b, m)
    c = [rng.randint(0, 1) for _ in range(b)]

    delta = [[0] * K for _ in range(S)]
    for s in range(S):
        cs = bits_of(pi[s], b)
        for k in range(K):
            ks = bits_of(k, m)
            newbits = matvec(M, cs)
            nb2 = matvec(N, ks)
            res = [newbits[i] ^ nb2[i] ^ c[i] for i in range(b)]
            newcode = int_of(res)
            delta[s][k] = inv_pi[newcode]

    if n_perturb > 0:
        pts = [(s, k) for s in range(S) for k in range(K)]
        rng.shuffle(pts)
        for (s, k) in pts[:n_perturb]:
            orig = delta[s][k]
            choices = [x for x in range(S) if x != orig]
            delta[s][k] = rng.choice(choices)

    return delta, pi


def main():
    testId = int(sys.argv[1])
    S, K, n_perturb = LADDER[max(1, min(10, testId))]

    b = (S - 1).bit_length()
    m = (K - 1).bit_length()
    assert (1 << b) == S and (1 << m) == K

    delta = None
    for attempt in range(MAX_ATTEMPTS):
        rng = random.Random((90173 * testId + 17) * 1000003 + attempt)
        cand_delta, pi = try_build(rng, S, K, b, m, n_perturb)
        identity_cost = encoding_cost(list(range(S)), S, K, b, m, cand_delta)
        planted_cost = encoding_cost(pi, S, K, b, m, cand_delta)
        if identity_cost - planted_cost >= MIN_GAP:
            delta = cand_delta
            lane_rng = rng
            break
    if delta is None:
        # exceedingly unlikely; fall back to the last candidate rather than crash
        delta = cand_delta
        lane_rng = rng

    # 64 lockstep lanes: cycle through every (state,symbol) domain point so the
    # checker's correctness check is exhaustive (exact-equivalence over the whole domain)
    domain = [(s, k) for s in range(S) for k in range(K)]
    lane_rng.shuffle(domain)
    lanes = [domain[i % len(domain)] for i in range(LANES)]

    out = []
    out.append(f"{S} {K}")
    for s in range(S):
        out.append(" ".join(str(x) for x in delta[s]))
    for (s, k) in lanes:
        out.append(f"{s} {k}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
