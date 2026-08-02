import sys, random


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def vmod(bits, M):
    v = 0
    for c in bits:
        v = (2 * v + (1 if c == '1' else 0)) % M
    return v


def find_candidate_modulus(samples, cap=24):
    """Recover the ground-truth (modulus, accept-map) directly from the
    (bits,label) sample set: the smallest modulus consistent with every
    given pair.  gen.py's self-check already guarantees this recovers the
    planted law exactly, so the checker needs no separate secret channel."""
    for Mc in range(2, cap + 1):
        resmap = {}
        ok = True
        for bits, label in samples:
            r = vmod(bits, Mc)
            if r in resmap:
                if resmap[r] != label:
                    ok = False
                    break
            else:
                resmap[r] = label
        if ok:
            return Mc, resmap
    return None, None


def main():
    if len(sys.argv) < 3:
        fail("bad invocation")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        toks = open(inf).read().split()
        it = 0
        seed = int(toks[it]); it += 1
        step_bound = int(toks[it]); it += 1
        n = int(toks[it]); it += 1
        samples = []
        for _ in range(n):
            bits = toks[it]; it += 1
            label = int(toks[it]); it += 1
            if any(c not in '01' for c in bits) or len(bits) == 0:
                fail("bad input")
            samples.append((bits, label))
    except Exception:
        fail("bad input")

    M1, resmap = find_candidate_modulus(samples, cap=24)
    if M1 is None or len(resmap) != M1:
        fail("internal ground-truth reconstruction failed (problem bug)")
    for r in range(M1):
        if r not in resmap:
            fail("internal ground-truth incomplete (problem bug)")
    h = [resmap[r] for r in range(M1)]

    # ---- internal baseline B: a provably-correct but non-minimal machine.
    # It is the true M1-state law lifted with an inert mod-KAUX counter that
    # neither affects the accept decision nor the law's own transitions, so
    # it is guaranteed to classify every string identically to the truth --
    # just with KAUX times as many (redundant) states. ----
    KAUX = 8
    B = M1 * KAUX

    # ---- held-out extrapolation set: deterministic, unseen by the solver ----
    rng = random.Random(90000 + seed)
    held = []
    for _ in range(400):
        L = rng.randint(1, 250)
        bits = ''.join(rng.choice('01') for _ in range(L))
        held.append((bits, h[vmod(bits, M1)]))
    for L in (1, 2, 3, 4, 249, 250):
        bits = ''.join(rng.choice('01') for _ in range(L))
        held.append((bits, h[vmod(bits, M1)]))

    # ---- parse participant machine ----
    try:
        otoks = open(outf).read().split()
    except Exception:
        fail("cannot read output")
    if not otoks:
        fail("empty output")
    oit = 0
    try:
        S = int(otoks[oit]); oit += 1
    except Exception:
        fail("bad state count")
    if S < 1 or S > 20000:
        fail("state count out of range")

    states = []
    try:
        for _ in range(S):
            t0 = otoks[oit]; oit += 1
            t1 = otoks[oit]; oit += 1
            tb = otoks[oit]; oit += 1
            parsed = []
            for t in (t0, t1, tb):
                if t == 'A' or t == 'R':
                    parsed.append(t)
                else:
                    iv = int(t)
                    if iv < 0 or iv >= S:
                        fail("transition target out of range")
                    parsed.append(iv)
            states.append(tuple(parsed))
    except IndexError:
        fail("truncated output")
    except ValueError:
        fail("non-finite or malformed token")
    if oit != len(otoks):
        fail("extra tokens in output")

    def simulate(bits):
        st = 0
        pos = 0
        steps = 0
        L = len(bits)
        while True:
            if steps >= step_bound:
                return None
            sym = bits[pos] if pos < L else '_'
            t0, t1, tb = states[st]
            tgt = t0 if sym == '0' else (t1 if sym == '1' else tb)
            steps += 1
            if tgt == 'A':
                return 1
            if tgt == 'R':
                return 0
            st = tgt
            pos += 1

    for bits, label in samples:
        r = simulate(bits)
        if r is None:
            fail("does not halt within the step bound on a visible pair")
        if r != label:
            fail("wrong classification on a visible pair %r" % bits[:40])

    for bits, label in held:
        r = simulate(bits)
        if r is None:
            fail("does not halt within the step bound on a held-out pair")
        if r != label:
            fail("wrong classification on a held-out pair %r" % bits[:40])

    F = S
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("M1=%d B=%d F=%d Ratio: %.6f" % (M1, B, F, sc / 1000.0))


if __name__ == "__main__":
    main()
