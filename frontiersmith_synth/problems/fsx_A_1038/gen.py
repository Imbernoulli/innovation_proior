import sys, math, cmath, random

P_ALPHABET = 8

# ladder table: N candidate sites, d = planted null-divisor (hidden from the solver),
# L = comb multiplicity so K = d*L is the emitter budget, T = number of harbors to light,
# Qtrap = protected bearings planted right next to a lit harbor (the trap),
# Qsoft = protected bearings planted elsewhere (generous threshold).
TABLE = {
    1:  dict(N=12, d=3, L=3, T=2, Qtrap=0, Qsoft=2),
    2:  dict(N=15, d=3, L=3, T=2, Qtrap=0, Qsoft=2),
    3:  dict(N=20, d=4, L=3, T=2, Qtrap=0, Qsoft=2),
    4:  dict(N=21, d=3, L=3, T=2, Qtrap=0, Qsoft=2),
    5:  dict(N=24, d=3, L=3, T=2, Qtrap=2, Qsoft=1),
    6:  dict(N=28, d=4, L=3, T=2, Qtrap=0, Qsoft=2),
    7:  dict(N=30, d=6, L=4, T=3, Qtrap=0, Qsoft=3),
    8:  dict(N=35, d=5, L=4, T=3, Qtrap=0, Qsoft=3),
    9:  dict(N=40, d=8, L=4, T=3, Qtrap=4, Qsoft=1),
    10: dict(N=42, d=6, L=4, T=3, Qtrap=4, Qsoft=1),
}


def intensity(N, P, chosen, j):
    s = 0j
    for (i, p) in chosen:
        s += cmath.exp(2j * math.pi * (p / P + i * j / N))
    return abs(s) ** 2


def phase_conj(i, j, N, P):
    ang = -2 * math.pi * i * j / N
    return round(ang / (2 * math.pi / P)) % P


def single_lobe(N, K, P, jstar):
    K = min(K, N)
    return [(i, phase_conj(i, jstar, N, P)) for i in range(K)]


def multi_lobe(N, K, P, targets):
    T = len(targets)
    K = min(K, N)
    base, rem = K // T, K % T
    chosen, pos = [], 0
    for gi in range(T):
        size = base + (1 if gi < rem else 0)
        jt = targets[gi]
        for k in range(size):
            i = pos + k
            chosen.append((i, phase_conj(i, jt, N, P)))
        pos += size
    return chosen


def main():
    t = int(sys.argv[1])
    cfg = TABLE[t]
    N, d, L, T = cfg['N'], cfg['d'], cfg['L'], cfg['T']
    g = N // d
    K = d * L
    P = P_ALPHABET

    m_pool = list(range(1, g))
    targets = None
    for attempt in range(4000):
        rng = random.Random(500000 + t * 9973 + attempt)
        ms = sorted(rng.sample(m_pool, min(T, len(m_pool))))
        cand = [d * m for m in ms]
        sch = single_lobe(N, K, P, cand[0])
        mch = multi_lobe(N, K, P, cand)
        if min(intensity(N, P, sch, j) for j in cand) < 1.0:
            continue
        if min(intensity(N, P, mch, j) for j in cand) < 3.0:
            continue
        targets = cand
        break
    if targets is None:
        raise RuntimeError("gen: could not place targets")
    sch = single_lobe(N, K, P, targets[0])
    mch = multi_lobe(N, K, P, targets)

    used = set(targets)
    protected = []
    thresholds = {}
    Qtrap, Qsoft = cfg['Qtrap'], cfg['Qsoft']

    if Qtrap > 0:
        cands = []
        for tb in targets:
            for off in range(1, d):
                for sign in (1, -1):
                    q = (tb + sign * off) % N
                    if q in used or q % d == 0:
                        continue
                    cands.append(q)
        seen = set(); uniq = []
        for q in cands:
            if q not in seen:
                seen.add(q); uniq.append(q)
        # rank/threshold against the multi-lobe leak specifically (that is exactly
        # what the "obvious" phase-conjugate greedy submits)
        uniq.sort(key=lambda q: -intensity(N, P, mch, q))
        for q in uniq[:Qtrap]:
            leak = intensity(N, P, mch, q)
            thresholds[q] = max(1.0, 0.35 * leak)
            protected.append(q); used.add(q)

    if Qsoft > 0:
        rng2 = random.Random(600000 + t)
        tries = 0
        while len(protected) < Qtrap + Qsoft and tries < 2000:
            tries += 1
            q = rng2.randrange(N)
            if q in used or q % d == 0:
                continue
            leak = intensity(N, P, mch, q)
            thresholds[q] = max(4.0, 3.0 * leak)
            protected.append(q); used.add(q)

    out = [f"{N} {K} {P}", str(len(targets)), " ".join(map(str, targets)), str(len(protected))]
    for q in protected:
        out.append(f"{q} {thresholds[q]:.6f}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
