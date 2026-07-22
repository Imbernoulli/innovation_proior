#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE bell-ringing ledger (TRAIN window) to stdout.

Temple scribes record, once per ceremony, the count of bell strikes in a running
ledger a(1), a(2), a(3), ...  The *clean* count obeys, for every line n>=3, ONE
of a few hidden short linear recurrences

        a(n) = A[n mod m] * a(n-1) + B[n mod m] * a(n-2) + C[n mod m]

where the ledger silently SWITCHES law according to  n mod m  for a hidden small
modulus m (which bell is rung first rotates through the ceremony cycle).  On any
finite window this looks like one long, messy law; only restricting to a residue
class n == r (mod m) exposes the clean short system behind class r.

A few percent of the LINES are copied down wrong by a tired scribe (corruptions):
the written number is replaced by a value far from the true count.  Those lines
poison any non-robust fit.

STDOUT prints ONLY:  a header "<N> <K> <testId>"  then the N observed (possibly
corrupted) training counts, one per line.  The hidden modulus m, the recurrence
coefficients, the corruption positions and the RNG seed are NEVER printed -- they
live only inside gen.py / verify.py, which the solver cannot read.  The held-out
lines a(N+1..N+K) are regenerated only inside verify.py.
"""
import sys, random


def hidden_build(t):
    """The hidden ledger for test id t.  Byte-identical in gen.py and verify.py.

    The clean count obeys a SHORT order-2 affine law selected by n mod m:
        a(n) = A[n mod m] * a(n-1) + B[n mod m] * a(n-2) + C[n mod m]
    for a hidden small modulus m.  On any finite window the interleaved laws look
    like one long messy law; restricting to a residue class exposes the clean
    small system.  Returns (N, K, m, laws, latent, observed) with 1-indexed lists
    of length N+K+1 (index 0 unused).  latent = clean counts; observed = latent
    with a few corrupted lines.  Lines 1,2 are seeds and are never corrupted."""
    rng = random.Random(918273117 + t * 2654435761)

    # ---- difficulty ladder: small/mild -> larger/adversarial ----
    if t <= 3:
        m = 2
        N = 120
        train_corr = 3
    else:
        m = 3
        N = 168
        train_corr = 3 + (t % 3)          # 3..5 wrong lines
    K = 50                                 # held-out lines to predict
    c_out = 6                              # corrupted held-out lines (irreducible)

    # ---- pick the per-class short laws (distinct dynamics) ----
    while True:
        laws = []
        used_ab = set()
        ok = True
        for r in range(m):
            A = rng.choice([1, 1, 2])
            B = rng.choice([-1, 0, 1])
            if (A, B) in used_ab:          # laws must differ in their dynamics
                ok = False
                break
            used_ab.add((A, B))
            C = rng.choice([x for x in range(-6, 7) if x != 0])
            laws.append((A, B, C))
        if not ok:
            continue
        a1 = rng.randint(3, 40)
        a2 = rng.randint(3, 40)
        latent = [0] * (N + K + 1)
        latent[1], latent[2] = a1, a2
        cap_ok = True
        for n in range(3, N + K + 1):
            A, B, C = laws[n % m]
            v = A * latent[n - 1] + B * latent[n - 2] + C
            latent[n] = v
            if abs(v) > 10 ** 9:
                cap_ok = False
                break
        if not cap_ok:
            continue
        mx = max(abs(latent[n]) for n in range(1, N + K + 1))
        if mx < 2000:                      # need real dynamic range
            continue
        break

    # ---- corruptions: a tired scribe writes some lines far from the truth ----
    observed = latent[:]
    pos_train = sorted(rng.sample(range(3, N + 1), train_corr))
    pos_out = sorted(rng.sample(range(N + 1, N + K + 1), c_out))
    for p in pos_train + pos_out:
        base = 1 + abs(latent[p])
        offset = base + rng.randint(0, 3 + abs(latent[p]))   # |off| >= 1+|latent|
        sign = rng.choice([-1, 1])
        observed[p] = latent[p] + sign * offset

    return N, K, m, laws, latent, observed


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    N, K, m, laws, latent, observed = hidden_build(t)
    out = ["%d %d %d" % (N, K, t)]
    for n in range(1, N + 1):
        out.append("%d" % observed[n])
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
