import sys, random

# ---------------------------------------------------------------------------
# Hidden law: accept(bits) = h[ value(bits) mod M1 ]   where value() reads the
# bitstring as a big-endian binary integer.  M1 and h are the ground truth the
# solver must recover (only sample pairs are ever shown to the solver).
# The difficulty ladder alternates a "local" regime (M1 a power of two <=8,
# so the last-4-bits / mod-16 view already determines mod-M1) with a "global"
# regime (M1 odd, coprime to 16, so no bounded suffix window can ever recover
# mod-M1 -- genuine full-history dependence).  testId 1..10 -> M1 below.
# ---------------------------------------------------------------------------
M1_TABLE = {1: 4, 2: 3, 3: 4, 4: 5, 5: 8, 6: 7, 7: 8, 8: 9, 9: 8, 10: 11}
NUM_SAMPLES = 600
SAMPLE_LEN_MAX = 60
STEP_BOUND = 280
PROBE_LEN_MAX = 250
N_PROBES = 300
KAUX = 8  # padding factor used by the checker's own internal baseline construction


def vmod(bits, M):
    v = 0
    for c in bits:
        v = (2 * v + (1 if c == '1' else 0)) % M
    return v


def rand_bits(rng, L):
    return ''.join(rng.choice('01') for _ in range(L))


def make_samples(rng, n, lmax):
    out = []
    for _ in range(n):
        L = rng.randint(1, lmax)
        out.append(rand_bits(rng, L))
    return out


# ---------------- greedy algorithm (fixed mod-16 / last-4-bit window) -------
def greedy_table(samples_labeled):
    buckets = [[] for _ in range(16)]
    for bits, label in samples_labeled:
        w = vmod(bits, 16)
        buckets[w].append(label)
    tbl = []
    for w in range(16):
        b = buckets[w]
        if not b:
            tbl.append(0)
        else:
            ones = sum(b)
            tbl.append(1 if 2 * ones >= len(b) else 0)
    return tbl


def greedy_predict(tbl, bits):
    w = vmod(bits, 16)
    return tbl[w]


# ---------------- strong algorithm (candidate-modulus consistency search) ---
def find_candidate_modulus(samples_labeled, cap=24):
    for Mc in range(2, cap + 1):
        resmap = {}
        ok = True
        for bits, label in samples_labeled:
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


def strong_predict(Mc, hmap, bits):
    r = vmod(bits, Mc)
    return hmap.get(r, 0)


def refine_size(Mc, h):
    """Moore partition refinement over the raw M-state residue automaton;
    returns the number of states after collapsing behaviourally-identical
    residues. Mirrors solutions/strong.py's refine() exactly, so gen.py can
    confirm the planted law is already irreducible (no accidental extra
    redundancy from a particular random h) before accepting an instance."""
    trans0 = [(2 * r) % Mc for r in range(Mc)]
    trans1 = [(2 * r + 1) % Mc for r in range(Mc)]
    part = [h[r] for r in range(Mc)]
    for _ in range(Mc + 1):
        idmap = {}
        newpart = [0] * Mc
        nextid = 0
        for r in range(Mc):
            key = (part[r], part[trans0[r]], part[trans1[r]])
            if key not in idmap:
                idmap[key] = nextid
                nextid += 1
            newpart[r] = idmap[key]
        if newpart == part:
            break
        part = newpart
    return len(set(part))


def main():
    seed = int(sys.argv[1])
    M1 = M1_TABLE[seed]

    nonce = 0
    while True:
        nonce += 1
        rng_h = random.Random(20000 + seed * 977 + nonce)
        h = [rng_h.randrange(2) for _ in range(M1)]
        if len(set(h)) == 1:
            h[0] = 1 - h[0]

        # -- self-check 0: the planted law must already be irreducible --
        # (a random h can accidentally create extra equivalent residues for
        # small M1; that would let 'strong' collapse below the intended
        # minimum and saturate the score, so reject and re-roll h instead).
        if refine_size(M1, h) != M1:
            continue

        rng_s = random.Random(10000 + seed * 613 + nonce)
        raw_samples = make_samples(rng_s, NUM_SAMPLES, SAMPLE_LEN_MAX)
        samples_labeled = [(b, h[vmod(b, M1)]) for b in raw_samples]

        # -- self-check 1: strong's candidate search must recover M1 exactly
        Mc, resmap = find_candidate_modulus(samples_labeled, cap=24)
        if Mc != M1:
            continue
        # every residue of the true law must be witnessed by the sample set
        if len(resmap) != M1:
            continue

        # -- self-check 2: greedy's fixed mod-16 window must behave as designed
        gtbl = greedy_table(samples_labeled)
        rng_p = random.Random(30000 + seed * 811 + nonce)
        probes = make_samples(rng_p, N_PROBES, PROBE_LEN_MAX)
        wrong = 0
        for bits in probes:
            truth = h[vmod(bits, M1)]
            if greedy_predict(gtbl, bits) != truth:
                wrong += 1
        is_trap = (M1 % 2 == 1)
        if is_trap:
            if wrong < int(0.15 * N_PROBES):
                continue  # trap must actually bite -- need a robust failure rate
        else:
            if wrong != 0:
                continue  # local regime: the mod-16 window must be exact

        break  # instance accepted

    lines = ["%d %d" % (seed, STEP_BOUND), str(NUM_SAMPLES)]
    for bits, label in samples_labeled:
        lines.append("%s %d" % (bits, label))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
