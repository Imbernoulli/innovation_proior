#!/usr/bin/env python3
"""gen.py <testId> -- polymer-sequence-design instance generator.
Deterministic: all randomness seeded from testId only.
"""
import sys, random

K = 3  # fixed number of monomer types

# per-case plan: (N, regime, strength, tight_cap)
#   regime  in {"neutral","plasticize","reinforce"}
#   strength in {"mild","strong"}
#   tight_cap: if True, clamp the cap of one type well below what naive
#              composition-matching would want (feedstock-limit trap)
PLAN = {
    1:  (36,  "neutral",    "mild",   False),
    2:  (48,  "plasticize", "mild",   False),
    3:  (60,  "reinforce",  "mild",   False),
    4:  (72,  "neutral",    "mild",   False),
    5:  (90,  "plasticize", "strong", False),
    6:  (108, "reinforce",  "strong", False),
    7:  (126, "neutral",    "mild",   False),
    8:  (150, "plasticize", "strong", True),
    9:  (180, "reinforce",  "strong", True),
    10: (210, "plasticize", "strong", True),
}


def round_robin(counts):
    """Deterministic maximal-spread interleave of a multiset with given per-type
    counts (classic fair-scheduling / Sainte-Lague style deficit scheduler)."""
    total = sum(counts)
    k = len(counts)
    remaining = list(counts)
    assigned = [0] * k
    seq = []
    for step in range(total):
        best_i, best_key = -1, None
        for i in range(k):
            if remaining[i] <= 0:
                continue
            frac = counts[i] / total
            key = frac * (step + 1) - assigned[i]
            if best_i == -1 or key > best_key + 1e-12:
                best_i, best_key = i, key
        seq.append(best_i + 1)
        assigned[best_i] += 1
        remaining[best_i] -= 1
    return seq


def block_seq(perm, counts):
    seq = []
    for t in perm:
        seq.extend([t + 1] * counts[t])
    return seq


def tg_pred(seq, M):
    n = len(seq)
    if n < 2:
        return M[seq[0] - 1][seq[0] - 1]
    inv_sum = 0.0
    for k in range(n - 1):
        a, b = seq[k] - 1, seq[k + 1] - 1
        inv_sum += 1.0 / M[a][b]
    return (n - 1) / inv_sum


def sample_composition(N, caps, rng, favor=None, favor_frac=0.5):
    """Random composition (n_1..n_K) with 0<=n_i<=caps[i], sum=N.
    If `favor` (a pair of type-indices) is given, prefer draws where those
    two types together hold roughly >= favor_frac of the chain (keeps the
    trap pair well represented so its dyad actually matters)."""
    best = None
    for _ in range(4000):
        c1 = rng.randint(0, N)
        c2 = rng.randint(0, N)
        lo, hi = min(c1, c2), max(c1, c2)
        n = [lo, hi - lo, N - hi]
        rng.shuffle(n)
        if not (all(0 <= n[i] <= caps[i] for i in range(K)) and sum(n) == N):
            continue
        if favor is None:
            return n
        fa, fb = favor
        frac = (n[fa] + n[fb]) / N
        if frac >= favor_frac:
            return n
        if best is None or frac > best[0]:
            best = (frac, n)
    if best is not None:
        return best[1]
    # fallback: proportional to caps, clipped to sum exactly N
    tot = sum(caps)
    n = [min(caps[i], N * caps[i] // max(1, tot)) for i in range(K)]
    while sum(n) < N:
        for i in range(K):
            if sum(n) >= N:
                break
            if n[i] < caps[i]:
                n[i] += 1
    while sum(n) > N:
        for i in range(K):
            if sum(n) <= N:
                break
            if n[i] > 0:
                n[i] -= 1
    return n


def baseline_error(N, K, caps, tg, target, M):
    """Mirrors the checker's own internal baseline construction: fill
    monomer types, in blocks, worst-pure-Tg-match first. Used here only to
    rejection-sample instances whose baseline lands in a well-calibrated
    error band (keeps the Ratio=min(1,F/(10B)) scoring well-conditioned
    instead of occasionally saturating or flatlining by sheer luck of the
    random draw)."""
    order = sorted(range(K), key=lambda i: -abs(tg[i] - target))
    seq = []
    for t in order:
        if len(seq) >= N:
            break
        take = min(caps[t], N - len(seq))
        seq.extend([t + 1] * take)
    return abs(tg_pred(seq, M) - target)


BASELINE_ERR_LO = 15.0
BASELINE_ERR_HI = 24.0


def build_instance(rng, N, regime, strength, tight_cap):
    # pure-component (diagonal) glass transitions, Kelvin-scale integers
    tg = rng.sample(range(250, 431, 5), K)

    # dyad interaction matrix M (symmetric); diagonal = pure component tg.
    # One randomly-chosen "trap pair" gets a large planted shift; the other
    # two off-diagonal pairs get only small neutral noise.
    M = [[0] * K for _ in range(K)]
    for i in range(K):
        M[i][i] = tg[i]
    pairs = [(0, 1), (0, 2), (1, 2)]
    trap_pair = pairs[rng.randrange(3)]

    shift_mag = {"mild": rng.randint(20, 35), "strong": rng.randint(55, 95)}[strength]
    if regime == "plasticize":
        trap_shift = -shift_mag
    elif regime == "reinforce":
        trap_shift = shift_mag
    else:
        trap_shift = rng.choice([-1, 1]) * rng.randint(3, 10)

    for (i, j) in pairs:
        base = (tg[i] + tg[j]) / 2.0
        if (i, j) == trap_pair:
            val = base + trap_shift
        else:
            val = base + rng.choice([-1, 1]) * rng.randint(3, 12)
        val = max(60.0, val)
        M[i][j] = M[j][i] = int(round(val))

    fa, fb = trap_pair
    other = [t for t in range(K) if t not in trap_pair][0]

    if regime == "neutral":
        # feedstock caps: generous, plenty of composition freedom
        caps = [int(rng.randint(int(0.55 * N), int(0.9 * N))) for _ in range(K)]
        tries = 0
        while sum(caps) < int(1.25 * N) and tries < 50:
            i = rng.randrange(K)
            caps[i] = min(N, caps[i] + max(1, N // 10))
            tries += 1
    else:
        # Squeeze the NON-trap type's supply so composition-only search
        # cannot dodge the trap pair by substituting in the bystander type:
        # any feasible composition must draw substantially on (fa,fb).
        caps = [0] * K
        caps[other] = int(rng.randint(int(0.20 * N), int(0.38 * N)))
        caps[fa] = int(rng.randint(int(0.55 * N), int(0.95 * N)))
        caps[fb] = int(rng.randint(int(0.55 * N), int(0.95 * N)))
        tries = 0
        while sum(caps) < int(1.2 * N) and tries < 50:
            i = rng.choice([fa, fb])
            caps[i] = min(N, caps[i] + max(1, N // 10))
            tries += 1

    if tight_cap:
        # squeeze the cap of one of the trap-pair types well below what an
        # uninformed composition-only search would want to allocate to it,
        # forcing a real feedstock trade-off.
        squeeze_t = fa
        caps[squeeze_t] = max(int(0.12 * N), min(caps[squeeze_t], int(0.30 * N)))
        while sum(caps) < N:  # keep the instance feasible
            caps[fb] = min(N, caps[fb] + max(1, N // 8))

    # ---- plant a reachable target: build a deliberate (composition, arrangement)
    # ---- pair matching the regime and read off its true Tg as the target.
    favor_frac = 0.55 if regime != "neutral" else None
    n_counts = sample_composition(N, caps, rng,
                                   favor=(fa, fb) if regime != "neutral" else None,
                                   favor_frac=favor_frac if favor_frac else 0.5)

    if regime == "plasticize":
        # isolate the trap pair: put the non-trap type in the middle block
        perm = [fa, other, fb]
        plant = block_seq(perm, n_counts)
    elif regime == "reinforce":
        plant = round_robin(n_counts)
    else:
        multiset = []
        for t in range(K):
            multiset.extend([t + 1] * n_counts[t])
        rng.shuffle(multiset)
        plant = multiset

    target = int(round(tg_pred(plant, M)))
    target = max(80, min(600, target))

    berr = baseline_error(N, K, caps, tg, target, M)
    debug = dict(regime=regime, strength=strength, trap_pair=trap_pair,
                 trap_shift=trap_shift, plant_counts=n_counts, caps=caps,
                 berr=berr)
    return N, tg, M, caps, target, berr, debug


def main():
    testId = int(sys.argv[1])
    N, regime, strength, tight_cap = PLAN[testId]
    rng = random.Random(1_000_003 * testId + 17)

    best = None  # (dist_to_band_center, N, tg, M, caps, target, debug)
    band_mid = (BASELINE_ERR_LO + BASELINE_ERR_HI) / 2.0
    for attempt in range(4000):
        N_, tg, M, caps, target, berr, debug = build_instance(rng, N, regime, strength, tight_cap)
        if BASELINE_ERR_LO <= berr <= BASELINE_ERR_HI:
            best = (0.0, N_, tg, M, caps, target, debug)
            break
        dist = abs(berr - band_mid)
        if best is None or dist < best[0]:
            best = (dist, N_, tg, M, caps, target, debug)

    _, N, tg, M, caps, target, debug = best

    if "--debug" in sys.argv:
        print(f"DEBUG {debug}", file=sys.stderr)

    out = []
    out.append(f"{N} {K}")
    out.append(" ".join(str(x) for x in tg))
    for i in range(K):
        out.append(" ".join(str(x) for x in M[i]))
    out.append(" ".join(str(x) for x in caps))
    out.append(str(target))
    print("\n".join(out))


if __name__ == "__main__":
    main()
