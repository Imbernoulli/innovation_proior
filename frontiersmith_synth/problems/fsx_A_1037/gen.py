import sys, random


def main():
    i = int(sys.argv[1])
    rng = random.Random(913131 + 7919 * i)

    # difficulty ladder: small -> large, K's sacrifice-fraction shrinks (harder to defend)
    if i <= 2:
        N = rng.randint(8, 10); M = rng.randint(5, 6)
    elif i <= 4:
        N = rng.randint(12, 16); M = rng.randint(6, 7)
    elif i <= 6:
        N = rng.randint(18, 22); M = rng.randint(7, 8)
    elif i <= 8:
        N = rng.randint(24, 28); M = rng.randint(9, 10)
    else:
        N = rng.randint(30, 36); M = rng.randint(10, 12)

    values = [rng.randint(15, 999) for _ in range(N)]

    # two machines are given large ("cheap to concentrate on") capacity, planted at a
    # random pair of indices so no fixed position can be hardcoded by a solution;
    # the rest are deliberately scarce.
    idxs = list(range(1, M + 1))
    rng.shuffle(idxs)
    p1, p2 = idxs[0], idxs[1]
    cap = [0] * (M + 1)
    for m in range(1, M + 1):
        if m in (p1, p2):
            lo = max(6, N // 2)
            hi = max(lo + 2, N)
            cap[m] = rng.randint(lo, hi)
        else:
            cap[m] = rng.randint(2, 4)

    R = int(round(N * rng.uniform(1.15, 1.6)))
    R = max(R, N)
    R = min(R, 2 * N - 1)

    P2 = M * (M - 1) // 2
    fracs = [0.28, 0.24, 0.22, 0.19, 0.17, 0.15, 0.13, 0.11, 0.09, 0.07]
    K = max(2, round(P2 * fracs[i - 1]))
    K = min(K, P2)

    lines = [
        "%d %d %d %d" % (N, M, R, K),
        " ".join(map(str, values)),
        " ".join(str(cap[m]) for m in range(1, M + 1)),
    ]
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
