#!/usr/bin/env python3
"""
gen.py <testId> -- one instance of "Multiplying Almost Right to Save Power".

Instance = an approximate-multiplier dataflow: K multiply-accumulate (MAC) positions
grouped into G accumulation chains. Every position carries S sample operand pairs
(the "held-out" evaluation vectors baked into the instance itself). Each chain has an
absolute error budget B_g that any truncation/compensation configuration must respect
on EVERY sample.

testId 1..4  : all chains short (length 2..6)               -- no accumulation trap.
testId 5..10 : one or two LONG chains (length 70..170) mixed
               with short ones                                -- the trap: naive biased
               (floor) truncation accumulates linearly with chain length and blows the
               budget well before it would on a short chain, while round-to-nearest
               ("compensated") truncation accumulates only sub-linearly because the
               per-op errors cancel.

Budgets are calibrated directly from the realized sample data (not from a closed-form
bound) so feasibility is always data-exact:
  - short chains: budget = (floor-trunc, max depth) achieved error + margin
                  -> uniform floor truncation at max depth is always safe here.
  - long chains : budget = (round-trunc, max depth) achieved error + margin, and is
                  verified to be strictly BELOW the floor-trunc error at max depth
                  -> only compensation unlocks the deepest truncation on these chains.

All randomness is seeded deterministically from testId only.
"""
import sys
import random

TMAX = 6
AREA = [100, 88, 76, 64, 52, 40, 28]      # area_table[t], strictly decreasing, t=0..TMAX
COMP_EXTRA = 10                            # extra area for enabling compensation on one MAC
S = 8                                      # sample vectors per position
MINV, MAXV = 40, 200                       # operand value range


def trunc0(x, t):
    """Floor truncation: clear the low t bits (always rounds DOWN -> one-sided bias)."""
    return x if t == 0 else (x >> t) << t


def trunc1(x, t):
    """Round-to-nearest truncation to a multiple of 2^t (compensated -> symmetric error)."""
    return x if t == 0 else ((x + (1 << (t - 1))) >> t) << t


def chain_bias(samples, t, c):
    """Max (over the S samples) of the chain-level MAC error at depth t, mode c.
    c=0 (floor) errors are always >=0 (systematic bias); c=1 (rounded) errors can be
    either sign, so we take the max ABSOLUTE error."""
    best = 0
    for s in range(len(samples[0])):
        exact = 0
        approx = 0
        for pos in samples:
            a, b = pos[s]
            exact += a * b
            if c == 0:
                approx += trunc0(a, t) * trunc0(b, t)
            else:
                approx += trunc1(a, t) * trunc1(b, t)
        d = exact - approx
        best = max(best, d if c == 0 else abs(d))
    return best


def draw_chain(rng, length):
    return [[(rng.randint(MINV, MAXV), rng.randint(MINV, MAXV)) for _ in range(S)]
            for _ in range(length)]


def build_instance(testId):
    rng = random.Random(2026 * 1000 + testId * 97 + 7)

    if testId <= 4:
        G = rng.randint(4, 5)
        lengths = [rng.randint(2, 6) for _ in range(G)]
        is_long = [False] * G
    else:
        G = rng.randint(4, 5)
        n_long = 1 if testId <= 7 else 2
        n_long = min(n_long, G)
        long_idx = set(rng.sample(range(G), n_long))
        lengths = []
        is_long = []
        for gi in range(G):
            if gi in long_idx:
                lengths.append(rng.randint(70, 170))
                is_long.append(True)
            else:
                lengths.append(rng.randint(2, 6))
                is_long.append(False)

    chains = [draw_chain(rng, L) for L in lengths]

    budgets = []
    for gi, samples in enumerate(chains):
        b0 = chain_bias(samples, TMAX, 0)
        if not is_long[gi]:
            margin = max(3, b0 // 8) + 1
            Bg = b0 + margin
        else:
            b1 = chain_bias(samples, TMAX, 1)
            margin = max(3, b1 // 8) + 1
            Bg = b1 + margin
            attempts = 0
            while Bg >= b0 and attempts < 40:
                chains[gi] = draw_chain(rng, lengths[gi])
                samples = chains[gi]
                b0 = chain_bias(samples, TMAX, 0)
                b1 = chain_bias(samples, TMAX, 1)
                margin = max(3, b1 // 8) + 1
                Bg = b1 + margin
                attempts += 1
        budgets.append(Bg)

    return chains, budgets


def main():
    testId = int(sys.argv[1])
    chains, budgets = build_instance(testId)
    K = sum(len(c) for c in chains)
    G = len(chains)

    out = []
    out.append(f"{K} {G} {S} {TMAX} {COMP_EXTRA}")
    out.append(" ".join(str(x) for x in AREA))
    for L, Bg in zip((len(c) for c in chains), budgets):
        out.append(f"{L} {Bg}")
    for samples in chains:
        for pos in samples:
            toks = []
            for (a, b) in pos:
                toks.append(str(a))
                toks.append(str(b))
            out.append(" ".join(toks))

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
