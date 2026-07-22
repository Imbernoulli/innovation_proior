# TIER: strong
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); m = int(next(it)); cap = int(next(it)); K = int(next(it))
    pairs = []
    for _ in range(K):
        a = int(next(it)); b = int(next(it))
        pairs.append((a, b))

    # -- Phase 1: a minimal binary-splitting core (bits = ceil(log2 n) probes) so
    #    every single fault keeps a unique fingerprint; this part is unavoidable
    #    and identical in spirit to the textbook recipe. --
    bits = max(1, (n - 1).bit_length())
    base_rows = min(bits, m)
    fp = [j & ((1 << base_rows) - 1) for j in range(n)]  # fp[j] = code(j), base_rows-wide

    # Hypothesis space H = the n singles + the K PUBLISHED doubles (exactly the
    # adversary's sweep -- nothing else). fp of a pair = OR of its two members.
    total_h = n + K
    hyp_fp = fp[:]  # singles first
    comp_to_hyps = [[j] for j in range(n)]
    for idx, (a, b) in enumerate(pairs):
        h = n + idx
        hyp_fp.append(fp[a] | fp[b])
        comp_to_hyps[a].append(h)
        comp_to_hyps[b].append(h)

    # -- Phase 2: the insight. A FULL 2-disjunct matrix (separating every possible
    #    pair of components) needs far more than the leftover budget. Instead spend
    #    the remaining T = m - base_rows probes buying disjunctness ONLY where the
    #    published sweep actually collides: greedily pick, one probe at a time, the
    #    single pivot component whose membership probe splits the CURRENT collision
    #    buckets of H (singles+published pairs) the most (max reduction of
    #    sum(bucket_size^2), i.e. max entropy gain), and stop once no split helps. --
    T = max(0, m - base_rows)
    used = [False] * n
    extra_pivots = []

    remaining_rounds = T
    while remaining_rounds > 0:
        bucket_count = {}
        for f in hyp_fp:
            bucket_count[f] = bucket_count.get(f, 0) + 1

        best_p, best_gain = -1, 0
        for p in range(n):
            if used[p]:
                continue
            with_p = {}
            for h in comp_to_hyps[p]:
                f = hyp_fp[h]
                with_p[f] = with_p.get(f, 0) + 1
            gain = 0
            for f, wp in with_p.items():
                total = bucket_count[f]
                gain += 2 * wp * (total - wp)
            if gain > best_gain:
                best_gain, best_p = gain, p

        if best_p < 0 or best_gain <= 0:
            break

        used[best_p] = True
        extra_pivots.append(best_p)
        for h in range(total_h):
            hyp_fp[h] <<= 1
        for h in comp_to_hyps[best_p]:
            hyp_fp[h] |= 1
        remaining_rounds -= 1

    rows = []
    for i in range(base_rows):
        bitv = 1 << i
        rows.append("".join("1" if (j & bitv) else "0" for j in range(n)))
    for p in extra_pivots:
        row = ["0"] * n
        row[p] = "1"
        rows.append("".join(row))

    print(len(rows))
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
