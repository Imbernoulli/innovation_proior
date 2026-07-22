# TIER: greedy
# The obvious approach: sample many random words (couriers wandering), then greedily keep the k
# words that add the most new coverage. It never recovers the hidden district partition, so its
# random words concentrate in a few districts near the seed and churn within them.
import sys, random

def main():
    d = sys.stdin.read().split()
    idx = 0
    n = int(d[idx]); idx += 1
    m = int(d[idx]); idx += 1
    k = int(d[idx]); idx += 1
    L = int(d[idx]); idx += 1
    t = int(d[idx]); idx += 1
    Gens = []
    for _ in range(m):
        Gens.append([int(d[idx + i]) for i in range(n)]); idx += n
    S = [int(d[idx + i]) for i in range(t)]; idx += t

    Ginv = []
    for G in Gens:
        H = [0] * n
        for x in range(n):
            H[G[x]] = x
        Ginv.append(H)

    def img(word):
        s = set()
        for p in S:
            cur = p
            for tok in word:
                cur = Gens[tok - 1][cur] if tok > 0 else Ginv[-tok - 1][cur]
            s.add(cur)
        return s

    rng = random.Random(20240576)
    pool = []
    pool.append(([], frozenset(S)))  # identity
    NPOOL = 600
    toks = [x for x in range(-m, m + 1) if x != 0]
    for _ in range(NPOOL):
        ln = rng.randint(1, L)
        w = [rng.choice(toks) for _ in range(ln)]
        pool.append((w, frozenset(img(w))))

    # greedy max-coverage selection of k words
    chosen = []
    covered = set()
    used = [False] * len(pool)
    for _ in range(k):
        best = -1; bg = -1
        for i, (w, im) in enumerate(pool):
            if used[i]:
                continue
            g = len(im - covered)
            if g > bg:
                bg = g; best = i
        if best < 0 or bg <= 0:
            break
        used[best] = True
        chosen.append(pool[best][0])
        covered |= pool[best][1]

    out = []
    for w in chosen:
        out.append(" ".join(map(str, w)) if w else "0")
    if not out:
        out = ["0"]
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
