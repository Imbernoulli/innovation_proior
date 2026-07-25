import sys
import random


def nonprimitive_period(rng, q, m):
    """Random period of length m with no smaller period (so pair-level shortcuts
    cannot compress a tandem repeat of it cheaply)."""
    while True:
        w = [rng.randrange(q) for _ in range(m)]
        ok = True
        for d in range(1, m):
            if m % d == 0 and w == w[:d] * (m // d):
                ok = False
                break
        # also avoid near-periodicity with period m-1 overlap tricks: require
        # first and last letters differ so boundary pair != internal pairs
        if ok and w[0] != w[-1]:
            return w


def make_case(t):
    rng = random.Random(987654321 + t * 10007)

    # parameter ladder (tuned so greedy RePair lands well below lookahead strong)
    table = {
        1:  dict(q=5, K=10, c=2, L=8,  groups=3, mlo=3, mhi=4, klo=40, khi=55,
                 decoy=0, noise=0.15),
        2:  dict(q=5, K=10, c=3, L=8,  groups=4, mlo=3, mhi=5, klo=45, khi=60,
                 decoy=0, noise=0.20),
        3:  dict(q=4, K=8,  c=3, L=10, groups=3, mlo=4, mhi=5, klo=55, khi=75,
                 decoy=8, noise=0.15),
        4:  dict(q=4, K=8,  c=4, L=10, groups=4, mlo=4, mhi=6, klo=55, khi=80,
                 decoy=10, noise=0.15),
        5:  dict(q=4, K=7,  c=4, L=10, groups=5, mlo=5, mhi=6, klo=60, khi=85,
                 decoy=12, noise=0.12),
        6:  dict(q=4, K=7,  c=4, L=12, groups=6, mlo=5, mhi=6, klo=65, khi=90,
                 decoy=14, noise=0.12),
        7:  dict(q=4, K=8,  c=5, L=12, groups=6, mlo=5, mhi=7, klo=65, khi=90,
                 decoy=14, noise=0.12),
        8:  dict(q=5, K=8,  c=4, L=12, groups=6, mlo=5, mhi=7, klo=70, khi=95,
                 decoy=14, noise=0.22),
        9:  dict(q=4, K=9,  c=4, L=12, groups=8, mlo=5, mhi=7, klo=70, khi=100,
                 decoy=16, noise=0.12),
        10: dict(q=4, K=10, c=4, L=12, groups=8, mlo=6, mhi=8, klo=75, khi=105,
                 decoy=16, noise=0.15),
    }
    p = table[t]
    q, K, c, L = p["q"], p["K"], p["c"], p["L"]

    segments = []   # list of token lists
    decoys = []     # pairs (a,b) whose count we inflate in the noise

    # --- trap tandem-repeat groups ---
    # Each group: (w)^k. The boundary pair (w[-1], w[0]) straddles copies;
    # replacing it first misaligns every copy of the repeat. We plant extra
    # occurrences of that boundary pair inside noise so the most-frequent-pair
    # heuristic commits to it first. With a tight rule budget K and per-rule
    # overhead c, pair-at-a-time compression of many medium repeats also runs
    # out of rules long before the whole-period rules do.
    shared_decoy = None
    for g in range(p["groups"]):
        m = rng.randint(p["mlo"], p["mhi"])
        w = nonprimitive_period(rng, q, m)
        k = rng.randint(p["klo"], p["khi"])
        segments.append(w * k)
        if p["decoy"] > 0:
            if t >= 6 and shared_decoy is None and rng.random() < 0.7:
                shared_decoy = (w[-1], w[0])   # one decoy shared across groups
            decoys.append((w[-1], w[0]) if shared_decoy is None else shared_decoy)

    # --- noise segments with planted decoy-pair occurrences ---
    rep_len = sum(len(s) for s in segments)
    noise_len = int(rep_len * p["noise"] / max(1e-9, 1.0 - p["noise"]))
    noise_len = max(noise_len, 40)
    # scatter decoy occurrences inside the noise
    chunks = []
    di = 0
    placed = 0
    target_decoys = sum(p["decoy"] for _ in decoys) if decoys else 0
    body = []
    while len(body) < noise_len:
        if decoys and placed < target_decoys and rng.random() < 0.35:
            a, b = decoys[di % len(decoys)]
            di += 1
            placed += 1
            # embed a,b guarded by separators different from a and b so we do
            # not accidentally build (a,b)* runs or extend a period
            guard = (a + 1 + rng.randrange(max(1, q - 1))) % q
            if guard == b:
                guard = (guard + 1) % q
            body.append(a)
            body.append(b)
            body.append(guard)
            body.append(rng.randrange(q))
        else:
            body.append(rng.randrange(q))
    # split noise into 2-3 segments so it interleaves with the repeats
    ncuts = rng.randint(1, 2)
    cuts = sorted(rng.randrange(1, len(body)) for _ in range(ncuts))
    prev = 0
    for cp in cuts + [len(body)]:
        segments.append(body[prev:cp])
        prev = cp

    rng.shuffle(segments)
    S = [t2 for s in segments for t2 in s]
    n = len(S)
    return n, K, c, L, q, S


def main():
    t = int(sys.argv[1])
    n, K, c, L, q, S = make_case(t)
    print(n, K, c, L, q)
    print("".join(chr(97 + x) for x in S))


main()
