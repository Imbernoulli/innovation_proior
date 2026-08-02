# TIER: strong
"""Insight: the checker's Tg model is a bond/dyad-frequency-weighted harmonic
mean of the interaction matrix M -- it depends on the RUN-LENGTH structure
(blocky vs. alternating), not just on the average composition. So instead of
only picking a composition and reacting it randomly, search over compositions
AND, for each one, evaluate the two structurally-extreme arrangements a
sequence of monomers can take:
  - fully segregated blocks (every ordering of the K block boundaries --
    exploits that a non-adjacent pair in block order NEVER forms a bond, so
    a costly/plasticizing interaction can be made to simply not occur), and
  - maximal alternation (a deficit/fair-scheduling interleave that pushes
    cross-type bonds as high as composition allows -- exploits a
    reinforcing interaction).
Keep whichever (composition, arrangement) pair lands closest to the target,
after a local refinement pass around the best coarse grid point.
"""
import sys, itertools


def tg_pred(seq, M):
    n = len(seq)
    inv_sum = 0.0
    for k in range(n - 1):
        a, b = seq[k] - 1, seq[k + 1] - 1
        inv_sum += 1.0 / M[a][b]
    if inv_sum <= 0:
        return None
    return (n - 1) / inv_sum


def round_robin(counts):
    total = sum(counts)
    k = len(counts)
    if total == 0:
        return []
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


def best_arrangement(counts, M, target):
    """Try every block permutation plus the round-robin interleave; return
    (err, seq) for the best of these constructions on this composition."""
    K = len(counts)
    best = None
    for perm in itertools.permutations(range(K)):
        seq = block_seq(perm, counts)
        tp = tg_pred(seq, M)
        if tp is None:
            continue
        err = abs(tp - target)
        if best is None or err < best[0]:
            best = (err, seq)
    rr = round_robin(counts)
    tp = tg_pred(rr, M)
    if tp is not None:
        err = abs(tp - target)
        if best is None or err < best[0]:
            best = (err, rr)
    return best


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); K = int(next(it))
    tg = [int(next(it)) for _ in range(K)]
    M = [[int(next(it)) for _ in range(K)] for _ in range(K)]
    caps = [int(next(it)) for _ in range(K)]
    target = int(next(it))

    step = max(1, N // 24)
    best_overall = None  # (err, seq)

    lo1 = max(0, N - caps[1] - caps[2])
    hi1 = min(caps[0], N)
    n1_vals = sorted(set(list(range(lo1, hi1 + 1, step)) + [lo1, hi1]))
    for n1 in n1_vals:
        rem1 = N - n1
        lo2 = max(0, rem1 - caps[2])
        hi2 = min(caps[1], rem1)
        if lo2 > hi2:
            continue
        n2_vals = sorted(set(list(range(lo2, hi2 + 1, step)) + [lo2, hi2]))
        for n2 in n2_vals:
            n3 = rem1 - n2
            if n3 < 0 or n3 > caps[2]:
                continue
            counts = [n1, n2, n3]
            res = best_arrangement(counts, M, target)
            if res is None:
                continue
            if best_overall is None or res[0] < best_overall[0]:
                best_overall = (res[0], res[1], counts)

    if best_overall is None:
        # fallback: fill caps to reach N, single block
        n = [0] * K
        rem = N
        for i in range(K):
            take = min(caps[i], rem)
            n[i] = take
            rem -= take
        seq = block_seq(list(range(K)), n)
        print(" ".join(map(str, seq)))
        return

    # local refine: fine-grained search (step=1) around the best coarse point
    _, _, base_counts = best_overall
    n1c = base_counts[0]
    for n1 in range(max(lo1, n1c - step), min(hi1, n1c + step) + 1):
        rem1 = N - n1
        lo2 = max(0, rem1 - caps[2])
        hi2 = min(caps[1], rem1)
        n2c = base_counts[1] if abs(n1 - n1c) <= step else lo2
        for n2 in range(max(lo2, n2c - step), min(hi2, n2c + step) + 1):
            n3 = rem1 - n2
            if n3 < 0 or n3 > caps[2]:
                continue
            counts = [n1, n2, n3]
            res = best_arrangement(counts, M, target)
            if res is None:
                continue
            if res[0] < best_overall[0]:
                best_overall = (res[0], res[1], counts)

    _, seq, _ = best_overall
    print(" ".join(map(str, seq)))


if __name__ == "__main__":
    main()
